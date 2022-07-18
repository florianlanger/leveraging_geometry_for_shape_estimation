from requests import get
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import sys
import json
# from leveraging_geometry_for_shape_estimation.visualise_3d.visualise_predictions_from_csvs_v3 import visualise
import torch
from torchaudio import list_audio_backends
from tqdm import tqdm
from scipy.spatial.transform import Rotation as scipy_rot
import quaternion
from pytorch3d.structures.meshes import Meshes,join_meshes_as_scene

from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.visualisation.visualisation_points_and_normals import convert_K,load_4by4_from_txt
from leveraging_geometry_for_shape_estimation.vis_pose.vis_pose import load_mesh,render_mesh,overlay_rendered_image,render_mesh_from_calibration,just_rendered_image, overlay_rendered_image_v2
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.ground_plane import get_model_to_infos_scannet_just_id

def vis_render(original_image,list_infos,gt_name,model_to_infos,COLOR_BY_CLASS):

        device = torch.device("cuda:0")

        dir_path_calibrations = '/scratches/octopus_2/fml35/datasets/scannet/scannet_frames_25k/'
        dir_shapes = '/scratches/octopus_2/fml35/datasets/shapenet_v2/ShapeNetRenamed/model/'

        K_path = dir_path_calibrations + gt_name.split('-')[0] + '/intrinsics_color.txt'
        K = load_4by4_from_txt(K_path)
        width,height = original_image.shape[1],original_image.shape[0]
        K = convert_K(torch.Tensor(K),width,height)

        # change because different convention
        list_meshes = []
        for infos in list_infos:
            q = [infos['q'][1],infos['q'][2],infos['q'][3],infos['q'][0]]
            # factor = np.array(model_to_infos[infos['model_id']]['bbox']) * 2
            # infos['s'] = (infos['s'] / factor).tolist()
            infos['r'] = scipy_rot.from_quat(q).as_matrix()
            category = model_to_infos[infos['model_id']]['category']
            # category = category.replace('bookshelf','bookcase')
            full_path_model = dir_shapes + category + '/' + infos['model_id'] + '/model_normalized.obj'
            mesh = load_mesh(full_path_model,infos['r'],infos['t'],infos['s'],device,color=COLOR_BY_CLASS[category])
            list_meshes.append(mesh)

        scene = join_meshes_as_scene(list_meshes).to(device)
        # print(scene.device)
        rendered_image = render_mesh_from_calibration(width,height,K,scene,device)
        resize = (480,360)
        # rendered_image = cv2.resize(rendered_image,resize)
        # original_image = cv2.resize(original_image,resize)
        out_image = overlay_rendered_image_v2(original_image,rendered_image)
        out_image = cv2.resize(out_image,resize)

        theta = np.pi/16
        r_cam = torch.tensor([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
        rendered_image_geometry = render_mesh_from_calibration(width,height,K,scene,device,t_cam = torch.tensor([0,-0.5,1.8]),r_cam=r_cam)
        rendered_image_geometry = just_rendered_image(rendered_image_geometry)
        rendered_image_geometry = cv2.resize(rendered_image_geometry,resize)

        return out_image,rendered_image_geometry


def get_predictions_own(model_to_infos,COLOR_BY_CLASS):
    pred_path = '/scratch/fml35/experiments/regress_T/runs_10_T_and_R_and_S/date_2022_06_16_time_17_46_25_five_refinements_depth_aug_points_in_bbox_3d_coords/predictions/epoch_000150/translation_pred_scale_pred_rotation_roca_init_retrieval_roca_all_images_False/our_single_predictions.json'
    # pred_path = '/scratch/fml35/experiments/regress_T/runs_10_T_and_R_and_S/date_2022_06_21_time_10_32_21_single_refinement_points_bbox_random_points_no_canny/predictions/epoch_000250/translation_pred_scale_pred_rotation_roca_init_retrieval_roca_all_images_True/our_single_predictions.json'
    out_dir = '/scratch/fml35/experiments/regress_T/vis_own/date_2022_06_16_time_17_46_25_five_refinements_depth_aug_points_in_bbox_3d_coords_epoch_000150_translation_pred_scale_pred_rotation_roca_init_retrieval_roca_all_images_False'
    
    with open(pred_path) as f:
        predictions = json.load(f)
        
    os.mkdir(out_dir)

    # out_dir = pred_path.split('.')[0] + '_visualised'

    img_dir = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/images/'

    gt_images_already_visualised = []

    for detection_name in tqdm(sorted(predictions)):
        gt_name = detection_name.rsplit('_',1)[0]
        if gt_name in gt_images_already_visualised:
            continue

        original_image = cv2.imread(img_dir + gt_name + '.jpg')

        list_infos = get_list_infos(predictions,gt_name,model_to_infos)

        out_render,render_geometry = vis_render(original_image,list_infos,gt_name,model_to_infos,COLOR_BY_CLASS)
        cv2.imwrite(out_dir + '/' + gt_name + '.png',out_render)
        cv2.imwrite(out_dir + '/' + gt_name + '_geometry.png',render_geometry)

        gt_images_already_visualised.append(gt_name)

def get_predictions_roca(model_to_infos,COLOR_BY_CLASS):

    out_dir = '/scratch/fml35/experiments/regress_T/vis_roca/results_pi_over_16_1_8'
    # os.mkdir(out_dir)

    with open('/scratch2/fml35/results/ROCA/per_frame_best_no_null_correct_category_names.json','r') as f:
        roca_single_frame = json.load(f)

    img_dir = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/images/'

    for name in tqdm(sorted(roca_single_frame)):
        gt_name = name.split('/')[0] + '-' + name.split('/')[2].split('.')[0]
        original_image = cv2.imread(img_dir + gt_name + '.jpg')

        list_infos = roca_single_frame[name]
        for i in range(len(list_infos)):
            list_infos[i]['model_id'] = list_infos[i]["scene_cad_id"][1]

            invert = np.array([[-1,0,0],[0,-1,0],[0,0,1.]])
            list_infos[i]['t'] = np.matmul(invert,list_infos[i]['t'])

            q = list_infos[i]['q']
            q = np.quaternion(q[0],q[1],q[2],q[3])
            R = quaternion.as_rotation_matrix(q)
            R = np.matmul(invert,R)
            q = quaternion.from_rotation_matrix(R)
            list_infos[i]['q'] = quaternion.as_float_array(q)

        out_render,render_geometry = vis_render(original_image,list_infos,gt_name,model_to_infos,COLOR_BY_CLASS)
        cv2.imwrite(out_dir + '/' + gt_name + '.png',out_render)
        cv2.imwrite(out_dir + '/' + gt_name + '_geometry.png',render_geometry)


def get_list_infos_gt(list_raw_infos):
    out_list = []
    for i in range(len(list_raw_infos)):
        if list_raw_infos[i]["associated_gt_infos"]["matched_to_gt_object"] == False:
            continue
        out_list.append(list_raw_infos[i]["associated_gt_infos"])
        out_list[-1]['model_id'] = out_list[-1]['model'].split('/')[2]
        out_list[-1]['t'] = out_list[-1]['trans_mat']
        out_list[-1]['s'] = out_list[-1]['scaling']

        q = scipy_rot.from_matrix(out_list[-1]['rot_mat']).as_quat()
        out_list[-1]['q'] = [q[3],q[0],q[1],q[2]]
    
    return out_list

def get_predictions_gt(model_to_infos,COLOR_BY_CLASS):

    out_dir = '/scratch/fml35/experiments/regress_T/vis_gt/results_pi_over_16_1_8'
    # os.mkdir(out_dir)

    with open('/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/all_detection_infos_all_images.json','r') as f:
        gt_infos = json.load(f)

    img_dir = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/images/'

    for name in tqdm(sorted(gt_infos)):
        gt_name = name.split('.')[0]
        original_image = cv2.imread(img_dir + gt_name + '.jpg')

        list_infos = get_list_infos_gt(gt_infos[name])

        if list_infos == []:
            out_render = cv2.resize(original_image,(480,360))
            render_geometry = np.ones((360,480,3))*255
        else:
            out_render,render_geometry = vis_render(original_image,list_infos,gt_name,model_to_infos,COLOR_BY_CLASS)
        cv2.imwrite(out_dir + '/' + gt_name + '.png',out_render)
        cv2.imwrite(out_dir + '/' + gt_name + '_geometry.png',render_geometry)




def get_list_infos(predictions,gt_name,model_to_infos):


    list_infos = []
    for detection_name in sorted(predictions):
        if detection_name.rsplit('_',1)[0] == gt_name:

            single_object_infos = predictions[detection_name]
            factor = np.array(model_to_infos[single_object_infos['model_id']]['bbox']) * 2
            single_object_infos['s'] = (single_object_infos['s'] / factor).tolist()
            list_infos.append(predictions[detection_name])
    return list_infos


def main():
    model_to_infos = get_model_to_infos_scannet_just_id()

    COLOR_BY_CLASS = {
        'bin': np.array([210, 43, 16]) / 255,
        'bathtub': np.array([176, 71, 241]) / 255,
        'bed': np.array([204, 204, 255]) / 255,
        'bookshelf': np.array([255, 191, 0]) / 255,
        'cabinet': np.array([255, 127, 80]) / 255,
        'chair': np.array([44, 131, 242]) / 255,
        'display': np.array([212, 172, 23]) / 255,
        'sofa': np.array([237, 129, 241]) / 255,
        'table': np.array([32, 195, 182]) / 255
    }
    COLOR_BY_CLASS = {k: np.array(COLOR_BY_CLASS[k][::-1]) for k in COLOR_BY_CLASS}
    get_predictions_own(model_to_infos,COLOR_BY_CLASS)
    # get_predictions_roca(model_to_infos,COLOR_BY_CLASS)
    # get_predictions_gt(model_to_infos,COLOR_BY_CLASS)
    

if __name__ == '__main__':
    main()