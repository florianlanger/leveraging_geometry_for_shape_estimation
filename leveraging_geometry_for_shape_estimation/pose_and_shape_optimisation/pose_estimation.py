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

def estimate_pose(pixels_real_original_image,real_bearings,world_coordinates,n_matches,device,intrinsic_camera_matrix,ransac_threshold,config):

    n = pixels_real_original_image.shape[0]
    pixels = np.zeros_like(pixels_real_original_image).astype(np.float32)
    pixels[:,0] = pixels_real_original_image[:,1]
    pixels[:,1] = pixels_real_original_image[:,0]

    if ransac_threshold is not None:
        min_n_matches = 4
    elif ransac_threshold is None:
        min_n_matches = 3

    if n < min_n_matches:
        repeat_world_coordinates = world_coordinates[:min_n_matches-n,:]
        repeat_pixels = pixels[:min_n_matches-n,:]
        repeat_real_bearings = real_bearings[:min_n_matches-n,:]
        
        world_coordinates = np.concatenate((world_coordinates,repeat_world_coordinates),axis=0)
        pixels = np.concatenate((pixels,repeat_pixels),axis=0)
        real_bearings = np.concatenate((real_bearings,repeat_real_bearings),axis=0)


    real_bearings = real_bearings.astype(np.float64)
    world_coordinates = world_coordinates.astype(np.float64)

    if ransac_threshold is not None:
        output = pyopengv.absolute_pose_ransac(real_bearings, world_coordinates, config["absolute_pose_algorithm"],ransac_threshold,iterations=50000)
        # output = pyopengv.absolute_pose_lmeds(real_bearings, world_coordinates, "upnp",ransac_threshold,iterations=10000)
        # output = pyopengv.absolute_pose_ransac(real_bearings, world_coordinates, "p3p_kneip"), ransac_threshold,iterations=10000)
        output = [output]

    elif ransac_threshold is None:
        if config["absolute_pose_algorithm"] == 'p3p_kneip':
            output = pyopengv.absolute_pose_p3p_kneip(real_bearings, world_coordinates)
        elif config["absolute_pose_algorithm"] == 'upnp':
            output = pyopengv.absolute_pose_upnp(real_bearings, world_coordinates)

    # for single_matrix in output:
    #     if np.isnan(single_matrix).any():
    #         output = [np.array([[1.,0,0,0],[0.,1.,0,0],[0.,0.0,1.,0.]])]
    #         break


    for i,single_matrix in enumerate(output):
        if np.isnan(single_matrix).any():
            output[i] = np.array([[1.,0,0,0],[0.,1.,0,0],[0.,0.0,1.,0.]])

    # print('Uncomment to keep valid predictions')
    
    return output