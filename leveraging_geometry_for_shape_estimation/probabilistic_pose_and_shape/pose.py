import cv2
import numpy as np
import sys
import os
import json
from torchvision import models
from tqdm import tqdm
from pytorch3d.renderer import look_at_view_transform
from math import ceil
import torch
from pytorch3d.io import load_obj
from scipy.spatial.transform import Rotation as scipy_rot

from leveraging_geometry_for_shape_estimation.keypoint_matching.get_matches_3d import load_information_depth_camera,create_pixel_bearing,pb_and_depth_to_wc
from probabilistic_formulation.utilities import create_all_possible_combinations,get_uvuv_p_from_superpoint,create_all_possible_combinations_uvuv_p_together
from probabilistic_formulation.factors import factor_3d_consistency,factor_depth_positive,factor_world_coordinates_valid,factor_3d_consistency_no_depth,factor_super_point_match



def init_Rs(tilts,elevs,azims):
    "expect tuples for each with (start,stop,n_steps) in degrees"

    tilts = np.linspace(tilts[0],tilts[1],tilts[2])
        # TODO: this used to be negative 
    elevs = np.linspace(elevs[0],elevs[1],elevs[2])
    azims = np.linspace(azims[0],azims[1],azims[2])
    Rs = np.zeros((azims.shape[0]*elevs.shape[0]*tilts.shape[0],3,3))
    counter = 0
    for tilt in tilts:
        for azim in azims:
            for elev in elevs:
                Rs[counter] = scipy_rot.from_euler('zyx',[tilt,azim,elev], degrees=True).as_matrix()
                counter += 1
            
    return Rs


def init_Ts(xs,ys,zs):

    xs = np.linspace(xs[0],xs[1],xs[2])
    ys = np.linspace(ys[0],ys[1],ys[2])
    zs = np.linspace(zs[0],zs[1],zs[2])
    x, y, z = np.meshgrid(xs,ys,zs, indexing='ij')
    Ts = np.stack([x.flatten(),y.flatten(),z.flatten()], axis=1)

    return Ts


def get_R_limits(azim,elev,probabilistic_shape_config):
    tilt_center = 0
    azim_center = 180 - float(azim)
    elev_center = - float(elev)

    tilts = (tilt_center - probabilistic_shape_config["tilt"]["range"],tilt_center + probabilistic_shape_config["tilt"]["range"],probabilistic_shape_config["tilt"]["steps"])
    elevs = (elev_center - probabilistic_shape_config["elev"]["range"],elev_center + probabilistic_shape_config["elev"]["range"],probabilistic_shape_config["elev"]["steps"])
    azims = (azim_center - probabilistic_shape_config["azim"]["range"],azim_center + probabilistic_shape_config["azim"]["range"],probabilistic_shape_config["azim"]["steps"])

    return tilts,elevs,azims

def get_T_limits(f,img_size,sensor_width,probabilistic_shape_config,bbox,gt_z):

    img_size = np.array(img_size)

    bbox = np.array(bbox)
    bbox_center = (bbox[0:2] + bbox[2:4]) / 2
    bbox_size = bbox[2:4] - bbox[0:2]

    T_lower_upper = np.zeros((2,2))
    for i,delta in enumerate(np.array([[probabilistic_shape_config['x']["range"],probabilistic_shape_config['y']["range"]],[-probabilistic_shape_config['x']["range"],-probabilistic_shape_config['y']["range"]]])):
        target_pixel = delta * bbox_size + bbox_center
        pb_xy = - (target_pixel - img_size/2) * sensor_width / img_size
        T_lower_upper[i] = pb_xy * gt_z / f
    
    xs = (T_lower_upper[0][0],T_lower_upper[1][0],probabilistic_shape_config['x']["steps"])
    ys = (T_lower_upper[0][1],T_lower_upper[1][1],probabilistic_shape_config['y']["steps"])


    # zs = (gt_z-probabilistic_shape_config['z']["range"],gt_z+probabilistic_shape_config['z']["range"],probabilistic_shape_config['z']["steps"])
    if probabilistic_shape_config["gt_z"] == "True":
        zs = (gt_z,gt_z,1)
    elif probabilistic_shape_config["gt_z"] == "False":
        zs = (0,probabilistic_shape_config['z']["range"]*f,probabilistic_shape_config['z']["steps"])

    return (xs,ys,zs)


def get_pb_real_grid(w,h,f,sensor_width,device):
    # get pixel bearing
    if w >= h:
        fov = 2 * np.arctan((sensor_width/2)/f)
    elif w < h:
        fov = 2 * np.arctan((sensor_width/2 * h/w)/f)
    P_proj = torch.Tensor([[[1/np.tan(fov/2.), 0.0000, 0.0000, 0.0000],
                            [0.0000, 1/np.tan(fov/2.), 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 1.0000],
                            [0.0000, 0.0000, 1.0000, 0.0000]]])
                            
    pb_x,pb_y,pb_z = create_pixel_bearing(max(w,h),max(w,h),P_proj,device)

    B_not_normalised = torch.stack([pb_x,pb_y,pb_z],dim=2)
    # crop
    if w >= h:
        B_not_normalised = B_not_normalised[int((w-h)/2):int((w+h)/2),:,:]
    elif w < h:
        B_not_normalised = B_not_normalised[:,int((h-w)/2):int((h+w)/2),:]
    B_normalised = B_not_normalised/torch.linalg.norm(B_not_normalised,dim=2).unsqueeze(2).repeat(1,1,3)

    return B_normalised

def check_gt_pose_in_limits(xs,ys,zs,tilts,elevs,azims,gt_T,gt_R):
    # x = gt_T[0] > xs[0] and gt_T[0] < xs[1]
    # y = gt_T[1] > ys[0] and gt_T[1] < ys[1]
    x = gt_T[0] > xs[0] - 0.000001 and gt_T[0] < xs[1] + 0.000001
    y = gt_T[1] > ys[0] - 0.000001 and gt_T[1] < ys[1] + 0.000001
    z = gt_T[2] > zs[0] - 0.000001 and gt_T[2] < zs[1] + 0.000001

    rot = list(scipy_rot.from_matrix(gt_R).as_euler('zyx', degrees=True))

    # t = rot[1] > tilts[0] and rot[1] < tilts[1]
    # e = rot[2] > elevs[0] and rot[2] < elevs[1]
    # a = rot[0] > azims[0] and rot[0] < azims[1]

    t = rot[0] > tilts[0] and rot[0] < tilts[1]
    a = rot[1] > azims[0] and rot[1] < azims[1]
    e = rot[2] > elevs[0] and rot[2] < elevs[1]

    all_correct = x and y and z and t and e and a
    return all_correct

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_angle(m1,m2):

    m = np.matmul(np.array(m1).T,np.array(m2))

    value = (np.trace(m) - 1 )/ 2

    clipped_value = np.clip(value,-0.9999999,0.999999)

    angle = np.arccos(clipped_value)

    return angle * 180 / np.pi 

def get_nearest_pose_to_gt_all_R(xs,ys,zs,Rs,gt_T,gt_R):
    xs = np.linspace(xs[0],xs[1],xs[2])
    ys = np.linspace(ys[0],ys[1],ys[2])
    zs = np.linspace(zs[0],zs[1],zs[2])

    best_T = []
    coords = [xs,ys,zs]
    for i in range(3):
        best_T.append(find_nearest(coords[i],gt_T[i]))

    angles = []
    for i in range(Rs.shape[0]):
        angle = get_angle(Rs[i],gt_R)
        angles.append(angle)

    min_index = np.argmin(angles)
    best_angle = angles[min_index]
    best_R = Rs[min_index]
    return best_T,best_R.tolist(),best_angle

def get_nearest_pose_to_gt(xs,ys,zs,tilts,elevs,azims,gt_T,gt_R):
    xs = np.linspace(xs[0],xs[1],xs[2])
    ys = np.linspace(ys[0],ys[1],ys[2])
    zs = np.linspace(zs[0],zs[1],zs[2])

    gt_rot = list(scipy_rot.from_matrix(gt_R).as_euler('zyx', degrees=True))
    # gt_rot_as_angles = (gt_rot[1],gt_rot[2],gt_rot[0])
    gt_rot_as_angles = (gt_rot[0],gt_rot[1],gt_rot[2])
    
    tilts = np.linspace(tilts[0],tilts[1],tilts[2])
    elevs = np.linspace(elevs[0],elevs[1],elevs[2])
    azims = np.linspace(azims[0],azims[1],azims[2])

    best_T = []
    coords = [xs,ys,zs]
    for i in range(3):
        best_T.append(find_nearest(coords[i],gt_T[i]))

    best_R = []
    # coords = [tilts,elevs,azims]
    coords = [tilts,azims,elevs]
    for i in range(3):
        best_R.append(find_nearest(coords[i],gt_rot_as_angles[i]))


    return best_T,best_R

def create_pose_info_dict(pred_R,pred_T,n_indices,max_factor,gt_pose_in_limits,gt_R,gt_T,best_R_possible,best_T_possible,xs,ys,zs,tilts,elevs,azims):
    pose_information = {}
    pose_information["predicted_r"] = pred_R.tolist()
    pose_information["predicted_t"] = pred_T.tolist()
    pose_information["indices"] = list(range(n_indices))
    pose_information["factor"] = max_factor
    pose_information["in_limits"] = str(gt_pose_in_limits)

    pose_information["gt_R"] = gt_R
    pose_information["gt_T"] = gt_T
    rot = list(scipy_rot.from_matrix(gt_R).as_euler('zyx', degrees=True))
    pose_information["gt_angles"] = [rot[0],rot[1],rot[2]]
    pose_information["best_R_possible"] = best_R_possible
    pose_information["best_T_possible"] = best_T_possible
    pose_information["limits"] = {} 

    pose_information["limits"]["xs"] = xs
    pose_information["limits"]["ys"] = ys
    pose_information["limits"]["zs"] = zs

    pose_information["limits"]["tilts"] = tilts
    pose_information["limits"]["elevs"] = elevs
    pose_information["limits"]["azims"] = azims
    return pose_information


def get_pose_for_folder(global_config):

    target_folder = global_config["general"]["target_folder"]
    models_folder_read = global_config["general"]["models_folder_read"]
    top_n_retrieval = global_config["keypoints"]["matching"]["top_n_retrieval"]


    pose_config = global_config["pose_and_shape"]["pose"]

    device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))
    torch.cuda.set_device(device)

    for name in tqdm(os.listdir(target_folder + '/cropped_and_masked')):

        # if not 'bed_0114' in name:
        #     continue 
        # if not 'sofa_0149' in name:
        #     continue
        
        with open(target_folder + '/nn_infos/' + name.split('.')[0] + '.json','r') as f:
            retrieval_list = json.load(f)["nearest_neighbours"]

        with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
            gt_infos = json.load(f)

        with open(target_folder + '/segmentation_infos/' + name.split('.')[0] + '.json','r') as f:
            segmentation_infos = json.load(f)

        
        for i in range(top_n_retrieval):
        # for i in range(1):
            output_path = target_folder + '/poses/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json'
            # if os.path.exists(output_path):
            #     continue

            elev = retrieval_list[i]["elev"]
            azim = retrieval_list[i]["azim"]

            with open(target_folder + '/matches_orig_img_size/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json','r') as f:
                matches_orig_img_size = json.load(f)


            # convert pixel to pixel bearing
            f = gt_infos["focal_length"]
            w = gt_infos["img_size"][0]
            h = gt_infos["img_size"][1]
            # w,h,f = 1024, 768 ,31.9477089690625
            W = global_config["models"]["img_size"]
            sensor_width = pose_config["sensor_width"]

            
            depth_path = models_folder_read + '/models/depth/' + retrieval_list[i]["path"].replace('.png','.npy')
            # depth_path = "/data/cvfs/fml35/derivative_datasets/pix3d_new/own_data/depth/256/bed/IKEA_MALM_3/elev_015_azim_158.0.npy"
            depth = np.load(depth_path)
            M = depth > 0

            P_proj = load_information_depth_camera(fov=60.)
            pb_x,pb_y,pb_z = create_pixel_bearing(W,W,P_proj,device)
            X_p = pb_and_depth_to_wc(pb_x,pb_y,pb_z,depth,elev,azim,M)

           
            # get infos
            uv = torch.Tensor(matches_orig_img_size["pixels_real_orig_size"]).long()
            uv_p = torch.Tensor(matches_orig_img_size["pixels_rendered"]).long()
            uv_uv_p = torch.cat((uv,uv_p),dim=1)
            B = get_pb_real_grid(w,h,f,sensor_width)


            # superpoint matches
            # n_super = 6
            # superpoint_real = torch.Tensor([[420,901],[504,885],[236 ,544],[708 ,345],[131 ,539],[598, 329],[189 ,119],[309 ,145],[378, 150]])
            # superpoint_rendered = torch.Tensor([[147, 222],[168, 220],[115, 148],[197,  87],[ 94, 148],[166,  93],[ 98 , 59],[120,  63],[138, 64]])
            # uv = torch.Tensor(superpoint_real[:n_super]).long()
            # uv_p = torch.Tensor(superpoint_rendered[:n_super]).long()
            # uv_uv_p = torch.cat((uv,uv_p),dim=1)




            # initialise Rs
            # tilts = (-3,2,7)
            # elevs = (0,-60,11)
            # azims = (0,50,11)
            tilts,elevs,azims = get_R_limits(azim,elev,global_config["pose_and_shape_probabilistic"]["pose"])
            # print(tilts,elevs,azims)
            # rot = list(scipy_rot.from_matrix(gt_infos["rot_mat"]).as_euler('zyx', degrees=True))

            # tilts = (rot[1],rot[1],1)
            # elevs = (rot[2],rot[2],1)
            # azims = (rot[0],rot[0],1)
            Rs = init_Rs(tilts,elevs,azims)

            # initialise Ts
            # xs = (-0.5,0.5,10)
            # ys = (-0.5,0.5,10)
            # zs = (1.1955-0.1,1.1955+0.1,5)
            # xs = (-0.3,0.3,10)
            # ys = (-0.3,0.3,10)
            # zs = (0.7, 1.3,10)
            # xs = (gt_infos["trans_mat"][0],gt_infos["trans_mat"][0],1)
            # ys = (gt_infos["trans_mat"][1],gt_infos["trans_mat"][1],1)
            # zs = (gt_infos["trans_mat"][2],gt_infos["trans_mat"][2],1)
            
            bbox = segmentation_infos["predictions"]["bbox"]
            gt_z = gt_infos["trans_mat"][2]
            xs,ys,zs = get_T_limits(f,[w,h],sensor_width,global_config["pose_and_shape_probabilistic"]["pose"],bbox,gt_z)
            Ts = init_Ts(xs,ys,zs)
                # true_azim = 180 - rot[0]

            gt_pose_in_limits = check_gt_pose_in_limits(xs,ys,zs,tilts,elevs,azims,gt_infos["trans_mat"],gt_infos["rot_mat"])
            best_T_possible,best_R_possible = get_nearest_pose_to_gt(xs,ys,zs,tilts,elevs,azims,gt_infos["trans_mat"],gt_infos["rot_mat"])

            # compute factors
            factors = []
            Ts = torch.Tensor(Ts)
            for i in tqdm(range(len(Rs))):
                R = torch.Tensor(Rs[i]).unsqueeze(0)
                uv_uv_p_batch,R_batch,T_batch = create_all_possible_combinations_uvuv_p_together(uv_uv_p,R,Ts)
                
                uv_batch = uv_uv_p_batch[:,:2]
                uv_p_batch = uv_uv_p_batch[:,2:4]

                factors_all_T = factor_3d_consistency_no_depth(uv_batch,uv_p_batch,R_batch,T_batch,B,X_p) * factor_world_coordinates_valid(uv_p_batch,M)
                factors += torch.sum(factors_all_T.reshape(uv.shape[0],Ts.shape[0]),dim=0).tolist()

            # print(factors)
            best_index = factors.index(max(factors))

            best_T_index = best_index % len(Ts)
            best_R_index = best_index // len(Ts)

            T = Ts[best_T_index]
            R = Rs[best_R_index]
            max_factor = max(factors)
            # print('Max value', max_factor)
            # print('best T', T)
            # print('T_gt',gt_infos["trans_mat"])
            # print('best R', R)
            # print('R_gt',gt_infos["rot_mat"])

            n_indices = len(matches_orig_img_size["pixels_real_orig_size"])
            pose_information = create_pose_info_dict(R,T,n_indices,max_factor,gt_pose_in_limits,gt_infos["rot_mat"],gt_infos["trans_mat"],best_R_possible,best_T_possible,xs,ys,zs,tilts,elevs,azims)

       
            with open(output_path,'w') as f:
                json.dump(pose_information,f,indent=4)

            save_path = output_path.replace('/poses/','/factors/').replace('.json','.npz')
            np.savez(save_path,factors=factors,Rs=Rs,Ts=Ts)





def main():
    np.random.seed(1)
    torch.manual_seed(0)

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["pose_and_shape_probabilistic"]["use_probabilistic"] == "True":
        get_pose_for_folder(global_config)
    

    



if __name__ == '__main__':
    main()

