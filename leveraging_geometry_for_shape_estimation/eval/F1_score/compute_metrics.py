
import torch
from tqdm import tqdm
import os
import json
import cv2
import numpy as np
from numpy.lib.utils import info
import torch
import os
import sys
import json

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform,FoVPerspectiveCameras,RasterizationSettings, MeshRenderer, MeshRasterizer,SoftPhongShader,Textures)
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj
from scipy.spatial.transform import Rotation as scipy_rot

from leveraging_geometry_for_shape_estimation.utilities.dicts import open_json_precomputed_or_current,determine_base_dir,load_json

from metrics import compare_meshes

def stretch_3d_coordinates(world_coordinates,planes,stretching):
    print('world_coordinates.shape',world_coordinates.shape)
    print('planes',planes.shape)
    print('stretchign',stretching.shape)
    for i in range(planes.shape[0]):
        n = planes[i,:3]
        d = planes[i,3]

        y = torch.sign(torch.matmul(world_coordinates,n) - d)

        sign = torch.unsqueeze(y,dim=-1)
        sign = sign.repeat((1,1,3))
        n = n.unsqueeze(0).unsqueeze(0).repeat((world_coordinates.shape[0],world_coordinates.shape[1],1))
        tau = stretching[:,i].clone().unsqueeze(1).unsqueeze(2).repeat((1,world_coordinates.shape[1],3))
        world_coordinates = world_coordinates + tau / 2 * sign * n

    return world_coordinates


def get_distance(x1,x2):
    return np.sum((np.array(x1) - np.array(x2))**2)**0.5

def get_all_distances(t_gt,t_pred):
    abs_dist = np.abs(np.array(t_gt) - np.array(t_pred))
    length = np.sum(abs_dist**2)**0.5

    magnitude_gt = np.sum(np.array(t_gt)**2)**0.5
    normalised_length = length/magnitude_gt
    normalised_dist = abs_dist/magnitude_gt

    return length,abs_dist,normalised_length,normalised_dist

def get_total_angle(m1,m2):

    m = np.matmul(np.array(m1).T,np.array(m2))

    value = (np.trace(m) - 1 )/ 2

    clipped_value = np.clip(value,-0.9999999,0.999999)

    angle = np.arccos(clipped_value)

    return angle * 180 / np.pi

def get_all_angles(m1,m2):

    # elev_1,tilt_1,azim_1 = list(scipy_rot.from_matrix(m1).as_euler('zyx', degrees=True))
    # elev_2,tilt_2,azim_2 = list(scipy_rot.from_matrix(m2).as_euler('zyx', degrees=True))
    angles_1 = scipy_rot.from_matrix(m1).as_euler('zyx', degrees=True)
    angles_2 = scipy_rot.from_matrix(m2).as_euler('zyx', degrees=True)
    diff_angles = np.abs(angles_1 - angles_2)
    two_angles_each = np.vstack((diff_angles,360-diff_angles))
    min_angles = np.min(two_angles_each,axis=0)
    diff_tilt,diff_azim,diff_elev = min_angles
    # diff_elev,diff_tilt,diff_azim = np.abs(elev_1 - elev_2),np.abs(tilt_1 - tilt_2),np.abs(azim_1 - azim_2)
    total_diff = get_total_angle(m1,m2)
    return diff_tilt,diff_azim,diff_elev,total_diff
    # pose_information["gt_angles"] = [rot[1],rot[2],rot[0]] = tilt azim elev

def get_angles_and_dists(r_gt,r_pred,t_gt,t_pred):
    diff_tilt,diff_azim,diff_elev,total_diff = get_all_angles(r_gt,r_pred)
    length,abs_dist,normalised_length,normalised_dist = get_all_distances(t_gt,t_pred)

    metric_dict = {}
    metric_dict["total_angle_diff"] = total_diff
    metric_dict["diff_tilt"] = diff_tilt
    metric_dict["diff_azim"] = diff_azim
    metric_dict["diff_elev"] = diff_elev
    metric_dict["diff_absolute_length"] = length
    metric_dict["diff_absolute_distances"] = abs_dist.tolist()
    metric_dict["diff_normalised_length"] = normalised_length
    metric_dict["diff_normalised_distances"] = normalised_dist.tolist()
    return metric_dict


def convert_metrics_to_json(shape_metrics):

    new_dict = {}
    for metric in shape_metrics:
        new_dict[metric] = shape_metrics[metric][0].item()

    new_dict['F1'] = shape_metrics['F1@0.300000'][0].item()

    return new_dict

def F1_from_prediction(R_gt,T_gt,gt_obj,predicted_obj,R_total,T_total,device,pose_path,visualisations,wc_predicted_depth,scaling_factor):

    # load and position ground truth object
    bs = R_total.shape[0]

    gt_vertices_origin,gt_faces,gt_properties = gt_obj
    R_gt = torch.Tensor(R_gt).to(device) #.inverse().to(device)
    T_gt = torch.Tensor(T_gt).to(device)
    gt_vertices = torch.transpose(torch.matmul(R_gt,torch.transpose(gt_vertices_origin,0,1)),0,1) + T_gt
    textures_gt = Textures(verts_rgb=torch.ones((1,gt_vertices.shape[0],3),device=device))
    gt_mesh = Meshes(verts=[gt_vertices], faces=[gt_faces[0]],textures=textures_gt)
    
    # load and position predicted object
    pred_vertices_origin,pred_faces,pred_properties = predicted_obj
    pred_vertices = pred_vertices_origin.repeat(bs,1,1)
    pred_faces = pred_faces[0].repeat(bs,1,1)

    # pred_vertices_no_z = torch.transpose(torch.matmul(R_total,torch.transpose(pred_vertices_origin,0,1)),0,1) + T_total
    # textures_predicted = Textures(verts_rgb=torch.ones((1,pred_vertices_no_z.shape[0],3),device=device))
    # pred_mesh_no_z = Meshes(verts=[pred_vertices_no_z], faces=[pred_faces[0]], textures=textures_predicted)

    # T_predicted_gt_z = torch.cat((T_total[:2],T_gt[2:]))
    pred_vertices_no_z = torch.transpose(torch.matmul(R_total,torch.transpose(pred_vertices,-1,-2)),-1,-2) + T_total
    textures_predicted = Textures(verts_rgb=torch.ones((bs,pred_vertices_no_z.shape[1],3),device=device))

    pred_mesh_no_z = Meshes(verts=pred_vertices_no_z, faces=pred_faces, textures=textures_predicted)

    meshes_path = None
    # visualisations = False
    if visualisations:
        meshes_path = pose_path + '/meshes'
        os.mkdir(meshes_path)
        save_obj(meshes_path + '/predicted_before_transform.obj',pred_vertices_origin,pred_faces[0])
        save_obj(meshes_path + '/predicted.obj',pred_vertices_no_z[0],pred_faces[0])
        save_obj(meshes_path + '/gt.obj',gt_vertices,gt_faces[0])
        save_obj(meshes_path + '/gt_scaled.obj',gt_vertices*scaling_factor,gt_faces[0])
        # print('saving ply')
        # save_ply(meshes_path + '/predicted_depth.ply',wc_predicted_depth.reshape(-1,3).cpu())
        # print('saved ply')

    shape_metrics_no_z = compare_meshes(pred_mesh_no_z, gt_mesh, meshes_path,visualisations, reduce=False, thresholds=[0.3,0.5,0.7])



    pred_vertices_gt_pose = torch.transpose(torch.matmul(R_gt,torch.transpose(pred_vertices_origin,0,1)),0,1) + T_gt
    # pred_vertices_gt_z = torch.transpose(torch.matmul(R_total,torch.transpose(pred_vertices_origin,0,1)),0,1) + T_predicted_gt_z
    textures_predicted = Textures(verts_rgb=torch.ones((1,pred_vertices_no_z.shape[1],3),device=device))


    pred_mesh_no_z = Meshes(verts=[pred_vertices_no_z[0]], faces=[pred_faces[0]], textures=textures_predicted)

    pred_mesh_gt_pose = Meshes(verts=[pred_vertices_gt_pose], faces=[pred_faces[0]], textures=textures_predicted)



    # shape_metrics_gt_z = compare_meshes(pred_mesh_gt_z, gt_mesh, reduce=False, thresholds=[0.3])

    # return pred_mesh_no_z, pred_mesh_gt_z, gt_mesh, shape_metrics_no_z, shape_metrics_gt_z

    return pred_mesh_no_z, pred_mesh_gt_pose, shape_metrics_no_z


def F1_from_prediction_shape(R_gt,T_gt,gt_obj,predicted_obj,R_total,T_total,predicted_stretching,planes,device,pose_path,visualisations,wc_predicted_depth,scaling_factor):

    # load and position ground truth object
    bs = R_total.shape[0]

    gt_vertices_origin,gt_faces,gt_properties = gt_obj
    R_gt = torch.Tensor(R_gt).to(device) #.inverse().to(device)
    T_gt = torch.Tensor(T_gt).to(device)
    gt_vertices = torch.transpose(torch.matmul(R_gt,torch.transpose(gt_vertices_origin,0,1)),0,1) + T_gt
    textures_gt = Textures(verts_rgb=torch.ones((1,gt_vertices.shape[0],3),device=device))
    gt_mesh = Meshes(verts=[gt_vertices], faces=[gt_faces[0]],textures=textures_gt)
    
    # load and position predicted object
    pred_vertices_origin,pred_faces,pred_properties = predicted_obj
    pred_vertices = pred_vertices_origin.repeat(bs,1,1)
    pred_faces = pred_faces[0].repeat(bs,1,1)



    pred_vertices = stretch_3d_coordinates(pred_vertices,planes,predicted_stretching)


    pred_vertices_no_z = torch.transpose(torch.matmul(R_total,torch.transpose(pred_vertices,-1,-2)),-1,-2) + T_total
    # pred_vertices_no_z = pred_vertices
    textures_predicted = Textures(verts_rgb=torch.ones((bs,pred_vertices_no_z.shape[0],3),device=device))
    pred_mesh_no_z = Meshes(verts=pred_vertices_no_z, faces=pred_faces, textures=textures_predicted)

    meshes_path = None
    if visualisations:
        meshes_path = pose_path + '/meshes'
        os.mkdir(meshes_path)
        save_obj(meshes_path + '/predicted_before_transform.obj',pred_vertices_origin,pred_faces[0])
        save_obj(meshes_path + '/predicted.obj',pred_vertices_no_z[0],pred_faces[0])
        save_obj(meshes_path + '/gt.obj',gt_vertices,gt_faces[0])
        save_obj(meshes_path + '/gt_scaled.obj',gt_vertices*scaling_factor,gt_faces[0])
        # print('saving ply')
        # save_ply(meshes_path + '/predicted_depth.ply',wc_predicted_depth.reshape(-1,3).cpu())
        # print('saved ply')
        # visualisations = False
        

    shape_metrics_no_z = compare_meshes(pred_mesh_no_z, gt_mesh, meshes_path,visualisations, reduce=False, thresholds=[0.3,0.4,0.5,0.7])

    pred_vertices_gt_pose = torch.transpose(torch.matmul(R_gt,torch.transpose(pred_vertices_origin,0,1)),0,1) + T_gt
    # pred_vertices_gt_z = torch.transpose(torch.matmul(R_total,torch.transpose(pred_vertices_origin,0,1)),0,1) + T_predicted_gt_z
    textures_predicted = Textures(verts_rgb=torch.ones((1,pred_vertices_no_z.shape[1],3),device=device))


    pred_mesh_no_z = Meshes(verts=[pred_vertices_no_z[0]], faces=[pred_faces[0]], textures=textures_predicted)

    pred_mesh_gt_pose = Meshes(verts=[pred_vertices_gt_pose], faces=[pred_faces[0]], textures=textures_predicted)


    return pred_mesh_no_z, pred_mesh_gt_pose, shape_metrics_no_z



def get_score_for_folder(global_config):

    target_folder = global_config["general"]["target_folder"]
    image_folder = global_config["general"]["image_folder"]
    models_folder_read = global_config["general"]["models_folder_read"]
    top_n_retrieval = global_config["keypoints"]["matching"]["top_n_retrieval"]


    pose_config = global_config["pose_and_shape"]["pose"]

    device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))
    torch.cuda.set_device(device)


    for name in tqdm(os.listdir(target_folder + '/selected_nn')):

        # with open(target_folder + '/selected_nn/' + name,'r') as f:
        #     selected = json.load(f)

        # with open(target_folder + '/nn_infos/' + name,'r') as f:
        #     retrieval_list = json.load(f)["nearest_neighbours"]

        # with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
        #     gt_infos = json.load(f)
        selected = load_json(target_folder + '/selected_nn/' + name)
        retrieval_list = open_json_precomputed_or_current('/nn_infos/' + name,global_config,'retrieval')["nearest_neighbours"]
        gt_infos = open_json_precomputed_or_current('/gt_infos/' + name.rsplit('_',1)[0] + '.json',global_config,'segmentation')


        if gt_infos["objects"] == []:
            continue


        number_nn = selected["selected_nn"]
        name_pose = name.split('.')[0] + '_' + str(number_nn).zfill(3) + '_' + str(selected["selected_orientation"]).zfill(2) + '.json'

        out_path = target_folder + '/metrics/' + name_pose
        # if os.path.exists(out_path):
        #     continue

        pose_info = load_json(target_folder + '/poses/' + name_pose)
        bbox_overlap = open_json_precomputed_or_current('/bbox_overlap/' + name.split('.')[0] + '.json',global_config,'segmentation')


        # predicted obj
        # model_path_pred = models_folder_read + "/models/remeshed/" + retrieval_list[i]["model"].replace('model/','')
        # number_nn = int(name.rsplit('_',2)[1].split('_')[0]) #.split('.')[0])
        number_nn = selected["selected_nn"]
        model_path_pred = global_config["dataset"]["dir_path"] + retrieval_list[number_nn]["model"]
        R_pred = torch.Tensor(pose_info["predicted_r"]).to(device).unsqueeze(0)
        T_pred = torch.Tensor(pose_info["predicted_t"]).to(device).unsqueeze(0)
        R_gt = gt_infos["objects"][bbox_overlap['index_gt_objects']]["rot_mat"]
        T_gt = gt_infos["objects"][bbox_overlap['index_gt_objects']]["trans_mat"]

        
        predicted_obj = load_obj(model_path_pred,device=device,create_texture_atlas=False, load_textures=False)
        


        model_path_gt = global_config["dataset"]["dir_path"] + gt_infos["objects"][bbox_overlap['index_gt_objects']]["model"]
        gt_obj = load_obj(model_path_gt,device=device,create_texture_atlas=False, load_textures=False)


        if global_config["pose_and_shape"]["shape"]["optimise_shape"] == "False":
            _,_,shape_metrics = F1_from_prediction(R_gt,T_gt,gt_obj,predicted_obj,R_pred,T_pred,device,None,False,None,None)
        elif global_config["pose_and_shape"]["shape"]["optimise_shape"] == "True":
            print('Unsqueeze ?')
            predicted_stretching = torch.Tensor(pose_info["predicted_stretching"]).to(device).unsqueeze(0).to(device)
            planes = torch.Tensor(global_config["pose_and_shape"]["shape"]["planes"]).to(device)
            _,_,shape_metrics = F1_from_prediction_shape(R_gt,T_gt,gt_obj,predicted_obj,R_pred,T_pred,predicted_stretching,planes,device,None,False,None,None)
    

        shape_metrics = convert_metrics_to_json(shape_metrics)
        # shape_metrics = {}

        pose_metrics = get_angles_and_dists(R_gt,pose_info["predicted_r"],T_gt,pose_info["predicted_t"])
        
        combined_metrics = {**shape_metrics, **pose_metrics}

        with open(out_path,'w') as f:
            json.dump(combined_metrics,f,indent=4)





def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)


    get_score_for_folder(global_config)

if __name__ == '__main__':
    print('Compute metrics')
    main()