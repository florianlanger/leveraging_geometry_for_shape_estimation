import torch
from scipy.spatial.transform import Rotation as scipy_rot
from pytorch3d.renderer import look_at_view_transform
import numpy as np
import os
import json
import sys

def create_rotations(elev,azim,dist,path_transformations):

    # elev = torch.cat([torch.zeros((number_views,)),15 * torch.ones((number_views,)),30 * torch.ones((number_views,)),45 * torch.ones((number_views,))])
    elev_repeat = torch.repeat_interleave(torch.Tensor(elev),len(azim))
    # azim = torch.linspace(0,337.5,16).repeat(4)
    azim_repeat = torch.Tensor(azim).repeat(len(elev))


    R, T = look_at_view_transform(dist=dist, elev=elev_repeat, azim=azim_repeat)

    np.savez(path_transformations  + 'R_T_torch.npz', R=R, T=T)


    pi = 3.14159265
    R_new = np.zeros((64,3))
    for i in range(64):
        single_rot = R[i].numpy()


        # rot = scipy_rot.from_euler('zyx',single_rot,degrees=False)

        rot = scipy_rot.from_matrix(single_rot)
                
        start_rot = scipy_rot.from_euler('zyx',[pi,-pi/2,pi],degrees=False)      
        final_rot = scipy_rot.from_euler('y',pi/2,degrees=False)
        total_rot = final_rot * rot * start_rot
        R_new[i] = total_rot.as_euler('zyx')
    
    np.savez(path_transformations  + 'R_T_blender.npz', R=R_new, T=T)


def create_rotations_debug(elev,azim,dist,path_transformations):

    # elev = torch.cat([torch.zeros((number_views,)),15 * torch.ones((number_views,)),30 * torch.ones((number_views,)),45 * torch.ones((number_views,))])
    elev_repeat = torch.repeat_interleave(torch.Tensor(elev),len(azim))
    # azim = torch.linspace(0,337.5,16).repeat(4)
    azim_repeat = torch.Tensor(azim).repeat(len(elev))


    R, T = look_at_view_transform(dist=dist, elev=elev_repeat, azim=azim_repeat)



    pi = 3.14159265
    R_new = np.zeros((64,3,3))
    for i in range(64):
        print(i)
        single_rot = R[i].numpy()

        rot = scipy_rot.from_matrix(single_rot)
                
        start_rot = scipy_rot.from_euler('zyx',[pi,-pi/2,pi],degrees=False)      
        final_rot = scipy_rot.from_euler('y',pi/2,degrees=False)
        total_rot = final_rot * rot * start_rot
        angles = total_rot.as_euler('zyx')
        changed_angles = [angles[2],angles[1],angles[0]]
        R_new[i] = rot.from_euler('zyx',changed_angles).as_matrix()

    np.savez(path_transformations  + 'R_T_blender_same_pix3d.npz', R=R_new, T=T)


if __name__ == '__main__':

    global_info = sys.argv[1] + '/global_information.json'

    with open(global_info,'r') as f:
        global_config = json.load(f)
        

    elev = global_config["models"]["elev"]
    azim = global_config["models"]["azim"]
    dist = global_config["models"]["dist"]
    path_transformations = global_config["general"]["target_folder"] + '/models/rotations/'
    # path_transformations = '/scratch/fml35/experiments/exp_024_debug/models/rotations/'
    create_rotations(elev,azim,dist,path_transformations)



    # R_T_old = np.load('/data/cornucopia/fml35/experiments/test_output_all_1/models/rotations/R_T_blender.npz')
    # R_T_new = np.load('/home/mifs/fml35/code/shape/retrieval_plus_keypoints/R_T_saved_for_blender.npz')

    # for i in range(64):
    #     if not (np.abs(R_T_old["R"][i] - R_T_new["R"][i]) < 0.0000001 ).all():
    #         print(i)
    #         print(R_T_old["R"][i])
    #         print(R_T_new["R"][i])

    # assert (np.abs(R_T_old["T"] - R_T_new["T"]) < 0.00001 ).all()

    #     2
    # [-3.14159265 -0.78539817  3.14159265]
    # [ 3.14159265 -0.78539817  3.14159265]
    # 3
    # [-3.14159265 -1.17809726  3.14159265]
    # [ 3.14159265 -1.17809726  3.14159265]


    
