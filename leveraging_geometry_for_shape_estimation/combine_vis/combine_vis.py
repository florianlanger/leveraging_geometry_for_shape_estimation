from numpy.core.fromnumeric import resize
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

from leveraging_geometry_for_shape_estimation.pose_and_shape_optimisation.pose_selection import compute_rendered_pixel,compute_rendered_pixel_shape,stretch_3d_coordinates
from leveraging_geometry_for_shape_estimation.utilities.dicts import open_json_precomputed_or_current,determine_base_dir

def resize_img(img,size):
    padded_img = np.zeros((size[0],size[1],3),dtype=np.uint8)
    h,w,_ = img.shape
    aspect_orig = float(h)/w
    aspect_new = float(size[0])/size[1]

    if aspect_new >= aspect_orig: 
        new_w = size[1]
        new_h = int(np.round(h * size[1]/w))
        resized_img = cv2.resize(img,(new_w,new_h))
        padded_img[int((size[0]-new_h)/2):int((size[0]+new_h)/2),:,:] = resized_img
    else:
        new_h = size[0]
        new_w = int(np.round(w * size[0]/h))
        resized_img = cv2.resize(img,(new_w,new_h))
        padded_img[:,int((size[1]-new_w)/2):int((size[1]+new_w)/2),:] = resized_img
    return padded_img

def load_image(path,size=(256,256)):
    if os.path.isfile(path):
        img = cv2.imread(path)
        img = resize_img(img,size=size)
    else:
        img = np.zeros((size[0],size[1],3),dtype=np.uint8)
    return img

def get_pose_for_folder(global_config):

    target_folder = global_config["general"]["target_folder"]
    image_folder = global_config["general"]["image_folder"]
    models_folder_read = global_config["general"]["models_folder_read"]
    top_n_retrieval = global_config["keypoints"]["matching"]["top_n_retrieval"]

    with open(target_folder + '/global_stats/visualisation_images.json','r') as f:
        visualisation_list = json.load(f)


    for name in tqdm(os.listdir(target_folder + '/poses_vis')):
        name = name.replace('.png','.json')
        
        gt_infos = open_json_precomputed_or_current('/gt_infos/' + name.rsplit('_',3)[0] + '.json',global_config,'segmentation')
        # with open(target_folder + '/gt_infos/' + name.rsplit('_',3)[0] + '.json','r') as f:
            # gt_infos = json.load(f)

        if gt_infos["img"] in visualisation_list:

            for i in range(top_n_retrieval):

                    out_path = target_folder + '/combined_vis/' + name.split('.')[0] + '.png'
                    # if os.path.exists(out_path):
                    #     continue

                    just_name = name.rsplit('_',3)[0] + '.png'
                    detection = name.rsplit('_',2)[0] + '.png'
                    retrieval = name.rsplit('_',1)[0] + '.png'
                    orientation = name.split('.')[0] + '.png'

                    seg = load_image(determine_base_dir(global_config,'segmentation') + '/segmentation_vis/' + detection)
                    matches = load_image(determine_base_dir(global_config,'retrieval') + '/matches_vis/' + retrieval,(256,512))

                    # matches_quality = load_image(target_folder + '/matches_quality_vis/' + retrieval)
                    matches_quality = load_image(determine_base_dir(global_config,'retrieval') + '/T_lines_vis/' + orientation.replace('.png','_closest_gt.png'))
                    keypoints = load_image(determine_base_dir(global_config,'retrieval') + '/keypoints_vis/' + detection) #.replace('.png','.' + file_ending))
                    lines = load_image(determine_base_dir(global_config,'lines') + '/lines_2d_filtered_vis/' + detection)
                    factors_lines = load_image(determine_base_dir(global_config,'R') + '/factors_lines_vis/' + orientation)
                    lines_T = load_image(target_folder + '/T_lines_vis/' + orientation.replace('.png','_selected.png'))
                    poses = load_image(target_folder + '/poses_vis/' + orientation)

                    # combined_top = cv2.hconcat([seg,keypoints,matches])
                    # combined_bottom = cv2.hconcat([matches_quality,lines,factors_lines,poses])
                    combined_top = cv2.hconcat([seg,matches,matches_quality])
                    combined_bottom = cv2.hconcat([lines,factors_lines,lines_T,poses])
                    combined = cv2.vconcat([combined_top,combined_bottom])
                    

                    cv2.imwrite(out_path,combined)

                    if os.path.exists(target_folder + '/metrics/' + name):
                        with open(target_folder + '/metrics/' + name,'r') as f:
                            metrics = json.load(f)

                        if 'F1' in metrics:
                            F1 = str(int(np.round(metrics["F1"]))).zfill(3)
                            angle  = str(int(np.round(metrics["total_angle_diff"]))).zfill(3)
                            out_path_2 = target_folder + '/combined_vis_metrics_name/' + name.split('.')[0] + '_' + str(i).zfill(3) + '_F1_' + F1 + '_angle_' + angle + '.png'
                            cv2.imwrite(out_path_2,combined)







def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["general"]["visualise"] == "True":
        get_pose_for_folder(global_config)

if __name__ == '__main__':
    main()