from platform import dist
import re
from cv2 import threshold
import numpy as np
import cv2
import os
import sys
import json
from tqdm import tqdm
import torch

from leveraging_geometry_for_shape_estimation.utilities.write_on_images import draw_text_block
from leveraging_geometry_for_shape_estimation.pose_and_shape_optimisation.select_best_v2 import get_angle

from probabilistic_formulation.factors.factors_T.compare_lines import sample_points_from_lines
from probabilistic_formulation.utilities import create_all_possible_combinations_2, create_all_possible_combinations_2_dimension_1

def draw_bbox(img,bbox,color=[0,0,255]):
    x_start, y_start, x_end, y_end = np.round(bbox).astype(int)

    cv2.line(img, (x_start, y_start), (x_end, y_start), color, 3)
    cv2.line(img, (x_start, y_start), (x_start, y_end), color, 3)
    cv2.line(img, (x_end, y_start), (x_end, y_end), color, 3)
    cv2.line(img, (x_start, y_end), (x_end, y_end), color, 3)



def visualise_lines(original_lines,filtered_lines,no_duplicates,bbox,expanded_bbox,img,vis_path,threshold_pixel_bbox,threshold_pixel_remove_duplicates,absolute_increase_bbox):

    for line in original_lines:
        y_start, x_start, y_end, x_end = [int(val) for val in line]
        cv2.line(img, (x_start, y_start), (x_end, y_end), [255,0,0], 2)

    for line in filtered_lines:
        y_start, x_start, y_end, x_end = [int(val) for val in line]
        cv2.line(img, (x_start, y_start), (x_end, y_end), [0,255,255], 2)

    for line in no_duplicates:
        y_start, x_start, y_end, x_end = [int(val) for val in line]
        cv2.line(img, (x_start, y_start), (x_end, y_end), [0,255,0], 2)

    draw_bbox(img,bbox)
    draw_bbox(img,expanded_bbox,color=[0,255,255])

    top_left_corner = [20,20]

    text = ['lines original: ' + str(original_lines.shape[0]),
            'lines filtered: ' + str(filtered_lines.shape[0]),
            'lines no duplicates: ' + str(no_duplicates.shape[0]),
            'threshold_pixel_bbox: ' + str(np.round(threshold_pixel_bbox,4)),
            'threshold_pixel_remove_duplicates_end_points: ' + str(threshold_pixel_remove_duplicates),
            'absolute_increase_bbox: ' + str(absolute_increase_bbox)]

    draw_text_block(img,text,top_left_corner,font_scale=2)

    cv2.imwrite(vis_path, img)



def expand_bbox(bbox,absolute_increase_bbox):
    expanded_bbox = [bbox[0] - absolute_increase_bbox,bbox[1] - absolute_increase_bbox,bbox[2] + absolute_increase_bbox,bbox[3] + absolute_increase_bbox]
    return expanded_bbox

def get_correct_orientation(name,exp_folder):

    rotations = []
    print(name)

    for k in range(4):

        with open(exp_folder + '/poses/' + name.split('.')[0] + '_' + str(0).zfill(3) + '_' + str(k).zfill(2) + '.json','r') as f:
            poses = json.load(f)

        rotations.append(poses["predicted_r"])
    correct_orientation = int(np.argmin([get_angle(np.array(poses["gt_R"]),rotation) for rotation in rotations]))

    orientation_name = name.split('.')[0] + '_' + str(0).zfill(3) + '_' + str(correct_orientation).zfill(2) + '.json'
    return orientation_name

def get_name(name,correct_lines):

    for k in range(4):

        orientation_name = name.split('.')[0] + '_' + str(0).zfill(3) + '_' + str(k).zfill(2) + '.json'
        if orientation_name in correct_lines:
            return orientation_name

def main():
    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["line_detection"]["use_correct_lines"] == True:

        with open(global_config["general"]["target_folder"] + '/global_stats/visualisation_images.json','r') as f:
            visualisation_list = json.load(f)

        exp_correct_lines = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_203_new_lines_from_2d_all_vis'

        with open(exp_correct_lines + '/global_stats/count_2d_lines_correct/T_lines_vis_quality_2d_lines_9_absolute_threshold_filtered.json','r') as f:
            correct_lines = json.load(f)
            
        target_folder = global_config["general"]["target_folder"]

        remove_duplicates_params = {"percent_line_overlap": 0.97, "relative_threshold":0.005,"points_per_line":100}
        
        for name in tqdm(sorted(os.listdir(target_folder + '/segmentation_infos'))):

            with open(target_folder + '/segmentation_infos/' + name,'r') as file:
                segmentation_infos = json.load(file)
            
            img_name = segmentation_infos["img"]

            visualise = img_name in visualisation_list
            input_path_img = target_folder +  '/images/' + img_name
            output_path_lines = target_folder + '/lines_2d_filtered/' + name.split('.')[0] + '.npy'
            vis_path = target_folder + '/lines_2d_filtered_vis/' + name.split('.')[0] + '.png'


            correct_orientation_name = get_name(name,correct_lines)
            # lines = np.load(input_path_lines).astype(float)
            lines = correct_lines[correct_orientation_name]['correct_lines']
            img = cv2.imread(input_path_img)
            h,w,_ = img.shape
            threshold_pixel_bbox = np.max([h,w]) * global_config["line_detection"]["threshold_for_filtering_percent_whole_image"]
            threshold_pixel_remove_duplicates = np.max([h,w]) * global_config["line_detection"]["threshold_for_removing_duplicates"]
            absolute_increase_bbox = np.max([h,w]) * global_config["line_detection"]["increase_bbox_percent_whole_image"]
            remove_duplicates_params["pixel_threshold"] = np.max([h,w]) * remove_duplicates_params["relative_threshold"]

            bbox = segmentation_infos["predictions"]["bbox"]
            expanded_bbox = expand_bbox(bbox,absolute_increase_bbox)


            lines = np.round(lines).astype(int)
            np.save(output_path_lines,lines)
            # print(no_duplicates)

            if visualise:
                visualise_lines(lines,lines,lines,bbox,expanded_bbox,img,vis_path,threshold_pixel_bbox,threshold_pixel_remove_duplicates,absolute_increase_bbox)



    

if __name__ == '__main__':
    print('Get correct lines')
    main()