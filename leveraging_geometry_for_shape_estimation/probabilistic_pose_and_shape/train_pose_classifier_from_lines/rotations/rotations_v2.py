from re import T
from time import time
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
from datetime import datetime

from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.pose import init_Rs,init_Ts,get_pb_real_grid,get_R_limits
from leveraging_geometry_for_shape_estimation.utilities.dicts import load_json
from leveraging_geometry_for_shape_estimation.pose_and_shape_optimisation.select_best_v2 import get_angle
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.rotations.factors import get_factor_reproject_lines_multiple_R_v2
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.rotations.visualise_rotations import visualise


def find_R_closest_query(R,R_query):

    transform_Rs  = [scipy_rot.from_euler('zyx',[0,0,0], degrees=True).as_matrix(),
                        scipy_rot.from_euler('zyx',[0,90,0], degrees=True).as_matrix(),
                        scipy_rot.from_euler('zyx',[0,180,0], degrees=True).as_matrix(),
                        scipy_rot.from_euler('zyx',[0,270,0], degrees=True).as_matrix()]

    rotated_Rs = [np.matmul(R,transform_R) for transform_R in transform_Rs]
    angles = [get_angle(rotated_R,R_query) for rotated_R in rotated_Rs]
    best_R = rotated_Rs[np.argmin(angles)]
    min_angle = np.min(angles)

    rotated_Rs = [r_rot.tolist() for r_rot in rotated_Rs]

    return best_R,rotated_Rs,min_angle


def analyse_min_angles(both_angles,total_counter,path_output_results):
    text = ''
    for key in both_angles:
        text += str(key) + '\n'
        text += 'For roca misleading as dont take symmetry into account\n'
        angles = np.array(both_angles[key])
        text += 'N total: ' + str(total_counter) + '\n'
        text += 'N angles: ' + str(len(angles)) + '\n'
        text += 'Percentage angles less than 5 degrees:' + str(np.sum(angles<5)/len(angles)) + '\n'
        text += 'Percentage angles less than 10 degrees:' + str(np.sum(angles<10)/len(angles))+ '\n'
        text += 'Percentage angles less than 20 degrees:' + str(np.sum(angles<20)/len(angles))+ '\n'

    print(text)
    with open(path_output_results,'w') as f:
        f.write(text)

def get_roca_rot(q):
    q = [q[1],q[2],q[3],q[0]]
    R = scipy_rot.from_quat(q).as_matrix()
    invert = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    R = np.matmul(invert,R)
    return R


def get_pose_for_folder():
    print('always use both lines')

    name = 'all_lines_both_stages'

    read_folder = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val'
    path_roca_detection_with_gt = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/all_detection_infos_fixed_category.json'
    path_output = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/'
    sw = 2
    device = torch.device("cuda:0")

    run_name = '{}_{}'.format(datetime.now().strftime("date_%Y_%m_%d_time_%H_%M_%S"),name)

    path_output_file = path_output + 'rotation_results/{}.json'.format(run_name)
    path_output_results = path_output + 'rotation_accuracies/{}.txt'.format(run_name)

    dir_path_shapenet = '/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/'
    lines_3D = np.array([[0,0.,0,1.,0.0,0],[0,0.,0,0.,1.0,0],[0,0.,0,0.,0.0,1.]])

    torch.cuda.set_device(device)

    # rot_limits = {"tilt":{"range": 20,"steps": 20},"elev":{"range": 45,"steps": 45},"azim":{"range": 45,"steps": 45}}
    rot_limits = {"tilt":{"range": 20,"steps": 20},"elev":{"range": 45,"steps": 45},"azim":{"range": 0,"steps": 1}}
    tilts,elevs,azims = get_R_limits(0,30,rot_limits)

    Rs,angles = init_Rs(tilts,elevs,azims)
    # np.save(out_folder + '/rotations_for_lines.npy',Rs)

    # Rs = np.load(out_folder + '/rotations_for_lines.npy')
    print('loading roca detection')
    with open(path_roca_detection_with_gt,'r') as f:
        roca_detection_with_gt = json.load(f)
    print('loading roca detection done')
    # have two step procedure, first use all lines from image to find up axis (as have visualised currently),
    # then use only lines in bbox (load from filtered) to determine azimuthal
    counter_non_existent = 0
    min_angles = {'roca':[] ,'gt':[]}
    total_counter = 0
    for gt_name in tqdm(sorted(roca_detection_with_gt)[:100000]):

        for i in range(len(roca_detection_with_gt[gt_name])):
            total_counter += 1
            roca_detection_with_gt[gt_name][i]['rotations_from_lines'] = {'rotation_found':False}
            t1 = time()
    
            gt_infos = roca_detection_with_gt[gt_name][i]["associated_gt_infos"]
            if gt_infos["matched_to_gt_object"] == False:
                continue

            # if 'scene0011_00-001700_02' not in roca_detection_with_gt[gt_name][i]["detection"]:
            #     continue

            # convert pixel to pixel bearing
            f = gt_infos["focal_length"]
            w = gt_infos["img_size"][0]
            h = gt_infos["img_size"][1]

            # get infos
            B = get_pb_real_grid(w,h,f,sw,device)

            # compute factors
            factors = []

            lines_2D = torch.Tensor(np.load(read_folder + '/lines_2d_cropped/' + gt_name.split('.')[0] + '.npy')).long()

            if not os.path.exists(read_folder + '/lines_2d_filtered/' + roca_detection_with_gt[gt_name][i]["detection"] + '.npy'):
                counter_non_existent += 1
                # print(read_folder + '/lines_2d_filtered/' + roca_detection_with_gt[gt_name][i]["detection"] + '.npy')
                continue

            lines_2D_filtered = torch.Tensor(np.load(read_folder + '/lines_2d_filtered/' + roca_detection_with_gt[gt_name][i]["detection"] + '.npy')).long()
            if lines_2D_filtered.shape[0] == 0:
                lines_2D_filtered = lines_2D

            lines_2D_filtered = lines_2D
            
            if len(lines_2D.shape) > 1:

                mask = ~(lines_2D[:,:2] == lines_2D[:,2:4]).all(dim=1)
                lines_2D = lines_2D[mask]
                line_dirs_3D = torch.Tensor([[1.00,  0.0000,  0.0000],[ 0.0000,  1.00,  0.0000],[ 0.,  0.0000,  1.00]])

                # 1. best tilt and elevation
                t2 = time()
                factors_batch,factor_all_2d_lines,all_masks = get_factor_reproject_lines_multiple_R_v2(torch.Tensor(Rs).to(device),line_dirs_3D.to(device),lines_2D.to(device),B.to(device),mask_elements=True)
                t3 = time()
                factors = factors_batch.cpu().tolist()
                best_R_index = factors.index(max(factors))
                best_R = Rs[best_R_index]
                best_angles = angles[best_R_index]

                factors_2d_lines_best_R = factor_all_2d_lines[best_R_index].cpu()
                mask_best_R = all_masks[best_R_index].cpu()

                infos_stage_1 = {"lines_2D": lines_2D[mask_best_R].numpy().tolist(),"factors":factors_2d_lines_best_R[mask_best_R].numpy().tolist(),"best_R":best_R.tolist(),"best_angles":best_angles.tolist()}

                # 2. best azim only from filtered lines

                tilts = (best_angles[0],best_angles[0],1)
                azims = (180-45,180+45,45)
                elevs = (best_angles[2],best_angles[2],1)

                new_Rs,new_angles = init_Rs(tilts,elevs,azims)
                t4 = time()
                factors_batch,factor_all_2d_lines,all_masks = get_factor_reproject_lines_multiple_R_v2(torch.Tensor(new_Rs).to(device),line_dirs_3D.to(device),lines_2D_filtered.to(device),B.to(device))
                t5 = time()
                factors = factors_batch.cpu().tolist()
                best_R_index = factors.index(max(factors))
                best_R = new_Rs[best_R_index]
                factors_2d_lines_best_R = factor_all_2d_lines[best_R_index].cpu()
                mask_best_R = all_masks[best_R_index].cpu()
                selected_R_gt,rotated_Rs,min_angle_gt = find_R_closest_query(best_R,gt_infos['rot_mat'])
                selected_R_roca,_,min_angle_roca = find_R_closest_query(best_R,get_roca_rot(roca_detection_with_gt[gt_name][i]['q']))
                infos_stage_2 = {"lines_2D": lines_2D_filtered[mask_best_R].numpy().tolist(),"factors":factors_2d_lines_best_R[mask_best_R].numpy().tolist(),
                "selected_R_gt":selected_R_gt.tolist(),"selected_R_roca":selected_R_roca.tolist(),'all_Rs':rotated_Rs,"min_angle_gt":min_angle_gt,"min_angle_roca":min_angle_roca}

                roca_detection_with_gt[gt_name][i]['rotations_from_lines'] = {'rotation_found':True,"stage_1":infos_stage_1,"stage_2":infos_stage_2}
                min_angles['roca'].append(min_angle_roca)
                min_angles['gt'].append(min_angle_gt)

                # for z,rot in enumerate(rotated_Rs):
                #     out_path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/rotation_vis/debug_upside_down_chair_changed_order_rot/{}_{}_4_rotations.png'.format(roca_detection_with_gt[gt_name][i]["detection"],str(z).zfill(3)) 
                #     visualise(read_folder,gt_name,gt_infos,dir_path_shapenet,rot,sw,device,lines_3D,np.array(infos_stage_2['lines_2D']),f,np.array(infos_stage_2['factors']),out_path)


        t6 = time()

    analyse_min_angles(min_angles,total_counter,path_output_results)
    # print('non existent ',counter_non_existent)
    with open(path_output_file,'w') as f:
        json.dump(roca_detection_with_gt,f,indent=4)

        # print('t1:',t2-t1)
        # print('t2:',t3-t2)
        # print('t3:',t4-t3)
        # print('t4:',t5-t4)
        # print('t5:',t6-t5)
        # print('t6:',t7-t6) 
        # print('----')


def main():
    np.random.seed(1)
    torch.manual_seed(0)

    get_pose_for_folder()
    


if __name__ == '__main__':
    print('Rotation from lines')
    main()

