from platform import dist
import numpy as np
import cv2
import os
import sys
import json
from tqdm import tqdm
import torch

def visualise_points(points,points_filtered,bbox,img,vis_path):
    size = int(np.round(min(img.shape[:2]) / 100))

    for point in points:
        cv2.circle(img, (point[0], point[1]), size, [255,0,0], -1)
    for point in points_filtered:
        cv2.circle(img, (point[0], point[1]), size, [0,255,0], -1)


    x_start, y_start, x_end, y_end = np.round(bbox).astype(int)

    cv2.line(img, (x_start, y_start), (x_end, y_start), [0,0,255], 3)
    cv2.line(img, (x_start, y_start), (x_start, y_end), [0,0,255], 3)
    cv2.line(img, (x_end, y_start), (x_end, y_end), [0,0,255], 3)
    cv2.line(img, (x_start, y_end), (x_end, y_end), [0,0,255], 3)

    cv2.imwrite(vis_path, img)

def filter_points_in_bbox(points,bbox,percent_expand):
    w_bbox = bbox[2] - bbox[0]
    h_bbox = bbox[3] - bbox[1]
    expanded_bbox = [bbox[0] - w_bbox*percent_expand,bbox[1] - h_bbox*percent_expand,bbox[2] + w_bbox*percent_expand,bbox[3] + h_bbox*percent_expand]
    expanded_bbox = np.array(expanded_bbox)
    lower = np.tile(expanded_bbox[0:2],(points.shape[0],1))
    upper = np.tile(expanded_bbox[2:4],(points.shape[0],1))
    greater_lower = np.all(points[:,:2] >= lower,axis=1)
    smaller_upper = np.all(points[:,:2] <= upper,axis=1)
    mask = greater_lower & smaller_upper
    return points[mask,:]


def main():
    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    with open(global_config["general"]["target_folder"] + '/global_stats/visualisation_images.json','r') as f:
        visualisation_list = json.load(f)

    target_folder = global_config["general"]["target_folder"]
    percent_expand = global_config["keypoints"]["detection"]["percent_expand_bbox_filter"]
    
    for name in tqdm(os.listdir(target_folder + '/cropped_and_masked')):

        with open(target_folder + '/segmentation_infos/' + name.split('.')[0] + '.json','r') as file:
            segmentation_infos = json.load(file)
        
        img_name = segmentation_infos["img"]

        visualise = img_name in visualisation_list
        input_path_img = target_folder +  '/images/' + img_name
        input_path_keypoints = target_folder + '/keypoints/' + img_name.split('.')[0] + '.npy'
        output_path_keypoints = target_folder + '/keypoints_filtered/' + name.split('.')[0] + '.npy'
        vis_path = target_folder + '/keypoints_filtered_vis/' + name.split('.')[0] + '.png'


        points = np.load(input_path_keypoints)
        img = cv2.imread(input_path_img)
        h,w,_ = img.shape

        # drop confidnece scors
        points = np.round(points[:,:2]).astype(int)
        filtered_points = filter_points_in_bbox(points,segmentation_infos["predictions"]["bbox"],percent_expand)

        clipped_points = filtered_points*0

        clipped_points[:,0] = np.clip(filtered_points[:,0],0,w-1)
        clipped_points[:,1] = np.clip(filtered_points[:,1],0,h-1)

        # assert np.all(clipped_points == filtered_points)

        np.save(output_path_keypoints,clipped_points)

        if visualise:
            visualise_points(points,clipped_points,segmentation_infos["predictions"]["bbox"],img,vis_path)




    

if __name__ == '__main__':
    print('Filter Points')
    main()