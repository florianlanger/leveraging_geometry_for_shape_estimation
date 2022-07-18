from platform import dist
import re
import numpy as np
import cv2
import os
import sys
import json
from tqdm import tqdm
import torch

def draw_bbox(img,bbox,color=[0,0,255]):
    x_start, y_start, x_end, y_end = np.round(bbox).astype(int)

    cv2.line(img, (x_start, y_start), (x_end, y_start), color, 3)
    cv2.line(img, (x_start, y_start), (x_start, y_end), color, 3)
    cv2.line(img, (x_end, y_start), (x_end, y_end), color, 3)
    cv2.line(img, (x_start, y_end), (x_end, y_end), color, 3)



def visualise_lines(original_lines,filtered_lines,bbox,expanded_bbox,img,vis_path):

    for line in original_lines:
        y_start, x_start, y_end, x_end = [int(val) for val in line]
        cv2.line(img, (x_start, y_start), (x_end, y_end), [255,0,0], 2)

    for line in filtered_lines:
        y_start, x_start, y_end, x_end = [int(val) for val in line]
        cv2.line(img, (x_start, y_start), (x_end, y_end), [0,255,0], 2)

    draw_bbox(img,bbox)
    draw_bbox(img,expanded_bbox,color=[0,255,255])

    cv2.imwrite(vis_path, img)


def filter_lines(lines,bbox,threshold):

    filtered_lines = []
    for line in lines:
        line = torch.from_numpy(line)
        space = torch.linspace(0,1,100).unsqueeze(1).tile(1,2)
        sampled_points_line = line[:2].unsqueeze(0).tile(100,1) + (line[2:4] - line[:2]).unsqueeze(0).tile(100,1) * space
        percent = check_points_in_bbox(sampled_points_line,bbox)
        if percent > threshold:
            filtered_lines.append(line.numpy())
    return np.array(filtered_lines)



def check_points_in_bbox(points,bbox):
    # return ratio of points in bbox
    bbox = torch.Tensor(bbox)
    n = points.shape[0]
    lower = torch.flip(bbox[0:2],dims=(0,)).unsqueeze(0).tile(n,1)
    upper = torch.flip(bbox[2:4],dims=(0,)).unsqueeze(0).tile(n,1)

    greater_lower = torch.all(points >= lower,dim=1)
    smaller_upper = torch.all(points <= upper,dim=1)
    combined = torch.bitwise_and(greater_lower,smaller_upper)
    return torch.sum(combined) / n


def expand_bbox(bbox,relative_increase):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    increase_width = width * relative_increase
    increase_height = height * relative_increase

    expanded_bbox = [bbox[0] - increase_width,bbox[1] - increase_height,bbox[2] + increase_width,bbox[3] + increase_height]
    return expanded_bbox

def main():
    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    threshold = global_config["line_detection"]["threshold_for_filtering"]

    with open(global_config["general"]["target_folder"] + '/global_stats/visualisation_images.json','r') as f:
        visualisation_list = json.load(f)

    target_folder = global_config["general"]["target_folder"]
    
    for name in tqdm(os.listdir(target_folder + '/segmentation_infos')):

        with open(target_folder + '/segmentation_infos/' + name,'r') as file:
            segmentation_infos = json.load(file)
        
        img_name = segmentation_infos["img"]

        visualise = img_name in visualisation_list
        input_path_img = target_folder +  '/images/' + img_name
        input_path_lines = target_folder + '/lines_2d_cropped/' + img_name.split('.')[0] + '.npy'
        output_path_lines = target_folder + '/lines_2d_filtered/' + name.split('.')[0] + '.npy'
        vis_path = target_folder + '/lines_2d_filtered_vis/' + name.split('.')[0] + '.png'


        lines = np.load(input_path_lines)
        img = cv2.imread(input_path_img)
        h,w,_ = img.shape

        bbox = segmentation_infos["predictions"]["bbox"]
        expanded_bbox = expand_bbox(bbox,global_config["line_detection"]["relative_increase_bbox"])

        filtered_lines = filter_lines(lines,expanded_bbox,threshold)

        np.save(output_path_lines,filtered_lines)

        if visualise:
            visualise_lines(lines,filtered_lines,bbox,expanded_bbox,img,vis_path)




    

if __name__ == '__main__':
    print('Filter Lines')
    main()