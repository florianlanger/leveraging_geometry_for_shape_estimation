import numpy as np
from scipy.spatial.transform import Rotation as scipy_rot
from scipy.optimize import minimize
import sys



def estimate_pose_shape(pixels_real_original_image,world_coordinates,pose_config,scipy_lbfgs_options,bounds,w,h,f,gt_infos,R_pred):

    pixels = np.zeros_like(pixels_real_original_image).astype(np.float32)
    pixels[:,0] = pixels_real_original_image[:,1]
    pixels[:,1] = pixels_real_original_image[:,0]

    world_coordinates = world_coordinates.astype(np.float64)

    
    fov =  2 * np.arctan(pose_config["sensor_width"]/(2*f))

    parameter_history = []
    def call_back_store(params):
        parameter_history.append(list(params) + [gt_z])

    options = scipy_lbfgs_options
    if options["finite_diff_rel_step"] == "None":
        options["finite_diff_rel_step"] = None

    if pose_config["use_gt_z"] == "False":
        assert False == True, "not implemented"
        # x0 = np.zeros(3,dtype=float)
        # init_T = [0.0,0.0,1.0]
        # x0[n_stretching:n_stretching+3] = init_T
        # args = (world_coordinates,pixels_real_original_image,w,h,fov,planes)
        # result = minimize(objective_function,x0,args=args,method='L-BFGS-B',options=options)

        # stretching = result["x"][0:n_stretching]
        # T = np.expand_dims(result["x"][n_stretching:n_stretching+3],axis=1)
        # R = scipy_rot.from_euler('zyx',result["x"][n_stretching+3:n_stretching+6]).as_matrix()

    elif pose_config["use_gt_z"] == "True":
        init_T, T_lower_bound, T_upper_bound = initialise_T_and_bounds(gt_infos,pose_config,bounds)
        x0 = init_T.copy()

        gt_z = np.array(gt_infos["trans_mat"])[2]
        parameter_history.append(list(x0) + [gt_z])
        args = (world_coordinates,pixels_real_original_image,w,h,fov,R_pred,gt_z)
        bounds = get_bounds(T_lower_bound, T_upper_bound)
        result = minimize(objective_function_gt_z,x0,args=args,method='L-BFGS-B',bounds=bounds,options=options,callback=call_back_store)


        parameter_history_standard = np.zeros((100,3))

        n_history = min(100,len(parameter_history))
        parameter_history_standard[:n_history,:] =  np.array(parameter_history[:n_history])
        x_out = result["x"]
        reprojection_dist = result["fun"]

        T = np.zeros(3)
        T[:2] = x_out
        T[2] = gt_z
    

    return T,reprojection_dist,parameter_history_standard


def objective_function(x,wc,pixel,w,h,fov,R):
    """ x has dim 3: t1,t2,t3"""
    T = x.copy()
    reprojected_pixel = transform_wc(wc,w,h,fov,R,T)
    reprojection_error = np.sum(np.sqrt(np.sum((reprojected_pixel - pixel)**2,axis=1)),axis=0)
    return reprojection_error

def objective_function_gt_z(x,wc,pixel,w,h,fov,R,gt_z):
    """ x has dim 2: t1,t2"""
    T = np.zeros(3)
    T[:2] = x
    T[2] = gt_z
    # print('TODO:check that reprojection uses correct matrix multiplication of R')
    reprojected_pixel = transform_wc(wc,w,h,fov,R,T)
    reprojection_error = np.sum(np.sqrt(np.sum((reprojected_pixel - pixel)**2,axis=1)),axis=0)

    return reprojection_error

def get_bounds(T_low, T_up):
    T_bounds = [tuple([T_low[0],T_up[0]]),tuple([T_low[1],T_up[1]])]
    return T_bounds

def initialise_T_and_bounds(gt_infos,pose_config,bounds):
    sensor_width = pose_config["sensor_width"]
    f = gt_infos["focal_length"]

    img_size = np.array(gt_infos["img_size"])
    # print(TODO:'Should not use gt bbox here!!!')
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


def transform_wc(wc,w,h,fov,R,T):
    """world coordinates, image width, image height, stretch parameter, plane normal, plane distance from origin, R rotation matrix, T tranlsation vector"""

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



def single_poses_loop(global_config,all_4_indices,world_coordinates_matches_all,pixels_real_original_image_all,w,h,f,gt_infos,n_keypoints_pose,R):
    pose_config = global_config["pose_and_shape"]["pose"]
    scipy_lbfgs_options = global_config["pose_and_shape"]["shape"]["scipy_lbfgs"]
    bounds = global_config["pose_and_shape"]["shape"]["bounds"]

    all_predicted_t = np.zeros((pose_config["max_poses_to_check"]+100,3))
    all_indices = np.zeros((pose_config["max_poses_to_check"]+100,n_keypoints_pose),dtype=int)
    all_param_history = np.zeros((pose_config["max_poses_to_check"]+100,100,3))
    all_reprojection_dists = np.zeros((pose_config["max_poses_to_check"]+100))


    counter_poses = 0
    for iter_quad,subset_4_indices in enumerate(all_4_indices):

        world_coordinates_matches = world_coordinates_matches_all[subset_4_indices]
        pixels_real_original_image = pixels_real_original_image_all[subset_4_indices]

        T,reprojection_dist,parameter_history = estimate_pose_shape(pixels_real_original_image,world_coordinates_matches,pose_config,scipy_lbfgs_options,bounds,w,h,f,gt_infos,R)

        if np.isnan(T).any():
            continue
        
        else:
            all_predicted_t[counter_poses] = T
            all_reprojection_dists[counter_poses] = reprojection_dist
            all_indices[counter_poses] = subset_4_indices
            all_param_history[counter_poses] = parameter_history
            counter_poses += 1


        if counter_poses >= pose_config["max_poses_to_check"]:
            break

    all_predicted_t = all_predicted_t[:counter_poses]
    all_reprojection_dists = all_reprojection_dists[:counter_poses]
    all_indices = all_indices[:counter_poses]
    all_param_history = all_param_history[:counter_poses]

    return all_predicted_t,all_reprojection_dists,all_indices,all_param_history

