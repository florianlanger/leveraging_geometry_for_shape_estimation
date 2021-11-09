import torch
import numpy as np
import pyopengv
import inspect
from pytorch3d.renderer import look_at_view_transform
import cv2
import time
from scipy.spatial.transform import Rotation as scipy_rot
from scipy.optimize import minimize
import sys



def estimate_pose_shape(pixels_real_original_image,real_bearings,world_coordinates,pose_config,scipy_lbfgs_options,bounds,w,h,f,planes,gt_infos,retrieval_infos):

    n = pixels_real_original_image.shape[0]
    pixels = np.zeros_like(pixels_real_original_image).astype(np.float32)
    pixels[:,0] = pixels_real_original_image[:,1]
    pixels[:,1] = pixels_real_original_image[:,0]


    real_bearings = real_bearings.astype(np.float64)
    world_coordinates = world_coordinates.astype(np.float64)

    
    fov =  2 * np.arctan(pose_config["sensor_width"]/(2*f))

    n_stretching = planes.shape[0]

    # options = {}
    # options["maxiter"]=100000
    # options["maxfun"]=15000
    # options["disp"] = False

    parameter_history = []
    def call_back_store(params):
        parameter_history.append(list(params) + [gt_z])

    options = scipy_lbfgs_options
    if options["finite_diff_rel_step"] == "None":
        options["finite_diff_rel_step"] = None

    # options["maxfev"] = 1

    if pose_config["use_gt_z"] == "False":
        x0 = np.zeros(6+n_stretching,dtype=float)
        init_T = [0.0,0.0,1.0]
        x0[n_stretching:n_stretching+3] = init_T
        args = (world_coordinates,pixels_real_original_image,w,h,fov,planes)
        result = minimize(objective_function,x0,args=args,method='L-BFGS-B',options=options)

        stretching = result["x"][0:n_stretching]
        T = np.expand_dims(result["x"][n_stretching:n_stretching+3],axis=1)
        R = scipy_rot.from_euler('zyx',result["x"][n_stretching+3:n_stretching+6]).as_matrix()

    elif pose_config["use_gt_z"] == "True":
        
        init_R = initialise_R(retrieval_infos)
        init_T, T_lower_bound, T_upper_bound = initialise_T_and_bounds(gt_infos,pose_config,bounds)
        x0 = initialise_x0(n_stretching,init_T,init_R)

        # x0 = np.array([ 0.07944504, 0.11434241 ,-0.18376836 , 0.03123633, -0.12866615,  0.,-2.61314048  ,-0.2945095])
        gt_z = np.array(gt_infos["trans_mat"])[2]
        parameter_history.append(list(x0) + [gt_z])
        args = (world_coordinates,pixels_real_original_image,w,h,fov,planes,gt_z)
        bounds = get_bounds(bounds,T_lower_bound, T_upper_bound,n_stretching,init_R)
        result = minimize(objective_function_gt_z,x0,args=args,method='L-BFGS-B',bounds=bounds,options=options,callback=call_back_store)


        parameter_history_standard = np.zeros((100,6+n_stretching))

        n_history = min(100,len(parameter_history))
        # n_history = len(parameter_history)

        parameter_history_standard[:n_history,:] =  np.array(parameter_history[:n_history])


        # x_out = x0
        x_out = result["x"]


        stretching = x_out[0:n_stretching]
        
        T = np.zeros(3)
        T[:2] = x_out[n_stretching:n_stretching+2]
        T[2] = gt_z
        T = np.expand_dims(T,axis=1)

        R = scipy_rot.from_euler('zyx',x_out[n_stretching+2:n_stretching+5]).as_matrix()

        # print('numpy: T = {}, R = {} , Stretching = {}'.format(T,R,stretching))

        # R,T,stretching = pytorch_lbfgs(x0,args,n_stretching,device)
    
    output = np.concatenate([R,T],axis=1)
    output = [output]
    

    return output,stretching, parameter_history_standard


def objective_function(x,wc,pixel,w,h,fov,planes,gt_z=None):
    """ x has dim 7: tau,t1,t2,t3,r1,r2,r3,"""
    n_stretch = planes.shape[0]
    tau = x[0:n_stretch]
    T = x[n_stretch:n_stretch+3]

    if gt_z != None:
        T[2] = gt_z
    R = scipy_rot.from_euler('zyx', x[n_stretch+3:n_stretch+6], degrees=False).as_matrix()
    reprojected_pixel = transform_wc(wc,w,h,fov,tau,planes,R,T)
    reprojection_error = np.sqrt(np.sum((reprojected_pixel - pixel)**2))



    return reprojection_error

def objective_function_gt_z(x,wc,pixel,w,h,fov,planes,gt_z):
    """ x has dim 7: tau,t1,t2,t3,r1,r2,r3,"""
    n_stretch = planes.shape[0]
    tau = x[0:n_stretch]
    T = np.zeros(3)
    T[:2] = x[n_stretch:n_stretch+2]
    T[2] = gt_z
    x[n_stretch+2] = 0
    R = scipy_rot.from_euler('zyx', x[n_stretch+2:n_stretch+5], degrees=False).as_matrix()

    # print('R,T,wc',R,T,wc)
    reprojected_pixel = transform_wc(wc,w,h,fov,tau,planes,R,T)
    # print('reprojected_pixel',reprojected_pixel)
    reprojection_error = np.sum(np.sqrt(np.sum((reprojected_pixel - pixel)**2,axis=1)),axis=0)
    # print(reprojection_error)
    # print(dfd)
    return reprojection_error

def bounds_as_output(bounds,which,n_stretching):
    if which == 'lower':
        index = 0
    elif which == 'upper':
        index = 1

    x0 = np.zeros(n_stretching + 5)
    for i in range(n_stretching + 5):
        x0[i] = bounds[i][index]
    return x0


def get_bounds(bounds,T_low, T_up,n_stretching,init_R):
    b_stretch = bounds["stretching"]
    stretch_bounds = [tuple([-b_stretch,b_stretch])] * n_stretching
    T_bounds = [tuple([T_low[0],T_up[0]]),tuple([T_low[1],T_up[1]])]
    r_b = np.array(bounds["rotations"]) * np.pi/180.
    R_bounds = [tuple([init_R[0] - r_b[0],init_R[0] + r_b[0]]),tuple([init_R[1]-r_b[1],init_R[1]+r_b[1]]),tuple([init_R[2]-r_b[2],init_R[2]+r_b[2]])]
    bounds = stretch_bounds + T_bounds + R_bounds
    return bounds

def initialise_R(retrieval_infos):
    x = 0
    y = np.pi - float(retrieval_infos["azim"]) / 180 * np.pi
    z = - float(retrieval_infos["elev"]) /180 * np.pi
    init_R = np.array([x,y,z])
    return init_R


def initialise_T_and_bounds(gt_infos,pose_config,bounds):
    sensor_width = pose_config["sensor_width"]
    f = gt_infos["focal_length"]

    img_size = np.array(gt_infos["img_size"])
    print('Should not use gt bbox here!!!')
    bbox = np.array(gt_infos["bbox"])
    bbox_center = (bbox[0:2] + bbox[2:4]) / 2
    bbox_size = bbox[2:4] - bbox[0:2]

    T_init_lower_upper = np.zeros((3,2))
    bound = bounds["delta_T"]
    for i,delta in enumerate(np.array([[0.,0.],[+bound,+bound],[-bound,-bound]])):
        target_pixel = delta * bbox_size + bbox_center
        pb_xy = - (target_pixel - img_size/2) * sensor_width / img_size
        T_init_lower_upper[i] = pb_xy * gt_infos['trans_mat'][2] / f

    return T_init_lower_upper

def initialise_x0(n_stretching,init_T,init_R):
    x0 = np.zeros(5+n_stretching,dtype=float)
    x0[n_stretching:n_stretching+2] = init_T
    x0[n_stretching+2:n_stretching+5] = init_R
    return x0


def transform_wc(wc,w,h,fov,tau,planes,R,T):
    """world coordinates, image width, image height, stretch parameter, plane normal, plane distance from origin, R rotation matrix, T tranlsation vector"""

    for i in range(planes.shape[0]):
        n = planes[i,:3]
        d = planes[i,3]
        y = np.inner(n,wc) - d
        sign = np.repeat(np.expand_dims(np.sign(y),1),3,axis=1)
        wc = wc + tau[i] / 2 * sign * n


    cc = np.matmul(R,wc.T).T + T
    

    sensor_width = 32
    f = (sensor_width /2) / np.tan(fov/2)

    pb = cc / (cc[:,2:]/f)
    # mask = (camera_coordinates[:,:,2] > 0)
    
    px = - pb[:,0] * w/sensor_width + w/2
    py = - pb[:,1] * w/sensor_width + h/2




    pixel = np.zeros((wc.shape[0],2))
    pixel[:,0] = py
    pixel[:,1] = px
    return pixel



def single_poses_loop(global_config,all_4_indices,world_coordinates_matches_all,pixels_real_original_image_all,real_bearings_all,indices_valid_matches_all,w,h,f,device,planes,gt_infos,retrieval_infos,n_keypoints_pose):
    pose_config = global_config["pose_and_shape"]["pose"]
    scipy_lbfgs_options = global_config["pose_and_shape"]["shape"]["scipy_lbfgs"]
    bounds = global_config["pose_and_shape"]["shape"]["bounds"]

    all_predicted_r = torch.zeros((pose_config["max_poses_to_check"]+100,3,3))
    all_predicted_t = torch.zeros((pose_config["max_poses_to_check"]+100,1,3))
    all_predicted_stretching = torch.zeros((pose_config["max_poses_to_check"]+100,planes.shape[0]))
    all_world_coordinates_matches_for_poses = torch.zeros((pose_config["max_poses_to_check"]+100,n_keypoints_pose,3))
    all_pixels_real_original_image_for_poses = torch.zeros((pose_config["max_poses_to_check"]+100,n_keypoints_pose,2))
    all_indices = np.zeros((pose_config["max_poses_to_check"]+100,n_keypoints_pose),dtype=int)
    all_param_history = np.zeros((pose_config["max_poses_to_check"]+100,100,6+planes.shape[0]))


    counter_poses = 0
    for iter_quad,subset_4_indices in enumerate(all_4_indices):

        world_coordinates_matches = world_coordinates_matches_all[subset_4_indices]
        pixels_real_original_image = pixels_real_original_image_all[subset_4_indices]
        real_bearings = real_bearings_all[subset_4_indices]
        indices_valid_matches = indices_valid_matches_all[subset_4_indices]


        ransac_threshold = None
        output,stretching,parameter_history = estimate_pose_shape(pixels_real_original_image,real_bearings,world_coordinates_matches,pose_config,scipy_lbfgs_options,bounds,w,h,f,planes,gt_infos,retrieval_infos)
        # output = estimate_pose(pixels_real_original_image,real_bearings,world_coordinates_matches,n_matches,device,intrinsic_camera_matrix,ransac_threshold,config)

        for n_pose in range(len(output)):
            if np.isnan(output[n_pose]).any():
                continue
            
            else:
                all_predicted_r[counter_poses] = torch.Tensor(output[n_pose][:,:3])
                all_predicted_t[counter_poses,0] = torch.Tensor(output[n_pose][:,3])
                all_predicted_stretching[counter_poses] = torch.Tensor(stretching)
                all_indices[counter_poses] = subset_4_indices
                all_param_history[counter_poses] = parameter_history
                
                if pose_config["choose_best_based_on"] ==  "keypoints" or pose_config["choose_best_based_on"] ==  "combined":
                    all_world_coordinates_matches_for_poses[counter_poses] = torch.Tensor(world_coordinates_matches)
                    all_pixels_real_original_image_for_poses[counter_poses] = torch.Tensor(pixels_real_original_image)
                
                counter_poses += 1


        if counter_poses >= pose_config["max_poses_to_check"]:
            break

    all_predicted_r = all_predicted_r[:counter_poses].to(device).to(torch.float32)
    all_predicted_t = all_predicted_t[:counter_poses].to(device).to(torch.float32)
    all_predicted_stretching = all_predicted_stretching[:counter_poses].to(device).to(torch.float32)
    all_world_coordinates_matches_for_poses = all_world_coordinates_matches_for_poses[:counter_poses].to(device).to(torch.float32)
    all_pixels_real_original_image_for_poses = all_pixels_real_original_image_for_poses[:counter_poses].to(device).to(torch.float32)
    all_indices = all_indices[:counter_poses]

    return all_predicted_r,all_predicted_t,all_predicted_stretching,all_world_coordinates_matches_for_poses,all_pixels_real_original_image_for_poses,all_indices,all_param_history

