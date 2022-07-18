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


def filter_lines(lines,bbox,threshold_pixel):
 
    filtered_lines = []
    for line in lines:
        line = torch.from_numpy(line)
        space = torch.linspace(0,1,100).unsqueeze(1).tile(1,2)
        sampled_points_line = line[:2].unsqueeze(0).tile(100,1) + (line[2:4] - line[:2]).unsqueeze(0).tile(100,1) * space
        percent = check_points_in_bbox(sampled_points_line,bbox)
        n_pixel = torch.linalg.norm(line[:2] - line[2:4]) * percent
        if n_pixel > threshold_pixel:
            filtered_lines.append(line.numpy())
    return np.array(filtered_lines)


def check_already_exists(line,existing_lines,pixel_threshold):
    already_exists = False
    for test_line in existing_lines:
        dist1 = np.linalg.norm(np.array(line[:2]) - np.array(test_line[:2]))
        dist2 = np.linalg.norm(np.array(line[2:4]) - np.array(test_line[2:4]))
        if dist1 < pixel_threshold and dist2 < pixel_threshold:
            already_exists = True
            break
        # check reverse
        dist1 = np.linalg.norm(np.array(line[:2]) - np.array(test_line[2:4]))
        dist2 = np.linalg.norm(np.array(line[2:4]) - np.array(test_line[:2]))
        if dist1 < pixel_threshold and dist2 < pixel_threshold:
            already_exists = True
            break
    return already_exists

def incorporate_new_line(line,existing_lines,remove_duplicates_params):

    percent_line_overlap = remove_duplicates_params["percent_line_overlap"]

    already_exists = False
    n_percent_1,n_percent_2 = compare_line_to_existing_lines(line,existing_lines,remove_duplicates_params["pixel_threshold"],remove_duplicates_params["points_per_line"]) 

    for i in range(n_percent_1.shape[0]):
        value1 = n_percent_1[i]
        value2 = n_percent_2[i]
        if value1 < percent_line_overlap and value2 < percent_line_overlap:
            continue
        elif value1 >= percent_line_overlap and value2 < percent_line_overlap:
            # replace existing line with new line
            existing_lines[i] = line
            already_exists = True
            break
        elif value1 < percent_line_overlap and value2 >= percent_line_overlap:
            # do nothing but mark that line already exists
            already_exists = True
            break
        elif value1 >= percent_line_overlap and value2 >= percent_line_overlap:
            # do nothing but mark that line already exists
            already_exists = True
            break

    if already_exists == False:
        existing_lines.append(line)

    return existing_lines

    

def compare_line_to_existing_lines(line,existing_lines,pixel_threshold,points_per_line):

    existing_lines = torch.Tensor(existing_lines)
    test_line = torch.Tensor(line).unsqueeze(0).repeat(existing_lines.shape[0],1)
    # print(existing_lines.shape)
    # print(test_line.shape)

    samples_1 = sample_points_from_lines(existing_lines,points_per_line)
    samples_2 = sample_points_from_lines(test_line,points_per_line)

    assert samples_1.shape == (existing_lines.shape[0],points_per_line,2),(samples_1.shape,[existing_lines.shape[0],points_per_line,2])
    assert samples_2.shape == (existing_lines.shape[0],points_per_line,2),(samples_2.shape,[existing_lines.shape[0],points_per_line,2])

    batched_1,batched_2 = create_all_possible_combinations_2_dimension_1(samples_1,samples_2)

    assert batched_1.shape == (existing_lines.shape[0],points_per_line**2,2),(batched_1.shape,(existing_lines.shape[0],points_per_line**2,2))

    dists = torch.linalg.norm(batched_1 - batched_2,dim=2)
    assert dists.shape == (existing_lines.shape[0],points_per_line**2),(dists.shape, (existing_lines.shape[0],points_per_line**2))
    dists = dists.view(existing_lines.shape[0],points_per_line,points_per_line)

    min_dists_1,_ = torch.min(dists,dim=2)
    min_dists_2,_ = torch.min(dists,dim=1)

    assert min_dists_1.shape == (existing_lines.shape[0],points_per_line)
    assert min_dists_2.shape == (existing_lines.shape[0],points_per_line)

    n_smaller_1 = torch.sum(min_dists_1 < pixel_threshold,dim=1)
    n_smaller_2 = torch.sum(min_dists_2 < pixel_threshold,dim=1)

    n_percent_1 = n_smaller_1 / points_per_line
    n_percent_2 = n_smaller_2 / points_per_line

    return n_percent_1,n_percent_2

def compare_lines_to_lines(lines1,lines2,pixel_threshold,points_per_line):
    assert lines1.shape == lines2.shape

    n_lines = lines1.shape[0]

    samples_1 = sample_points_from_lines(lines1,points_per_line)
    samples_2 = sample_points_from_lines(lines2,points_per_line)

    assert samples_1.shape == (n_lines,points_per_line,2),(samples_1.shape,[n_lines,points_per_line,2])
    assert samples_2.shape == (n_lines,points_per_line,2),(samples_2.shape,[n_lines,points_per_line,2])

    batched_1,batched_2 = create_all_possible_combinations_2_dimension_1(samples_1,samples_2)

    assert batched_1.shape == (n_lines,points_per_line**2,2),(batched_1.shape,(n_lines,points_per_line**2,2))

    dists = torch.linalg.norm(batched_1 - batched_2,dim=2)
    assert dists.shape == (n_lines,points_per_line**2),(dists.shape, (n_lines,points_per_line**2))
    dists = dists.view(n_lines,points_per_line,points_per_line)

    min_dists_1,_ = torch.min(dists,dim=2)
    min_dists_2,_ = torch.min(dists,dim=1)

    assert min_dists_1.shape == (n_lines,points_per_line)
    assert min_dists_2.shape == (n_lines,points_per_line)

    n_smaller_1 = torch.sum(min_dists_1 < pixel_threshold,dim=1)
    n_smaller_2 = torch.sum(min_dists_2 < pixel_threshold,dim=1)

    n_percent_1 = n_smaller_1 / points_per_line
    n_percent_2 = n_smaller_2 / points_per_line

    return n_percent_1,n_percent_2
    


def remove_duplicates(lines,threshold_pixel):
    no_duplicates = []
    for line in lines:
        already_exists = check_already_exists(line,no_duplicates,threshold_pixel)
        if not already_exists:
            no_duplicates.append(line)
    return np.array(no_duplicates)

def remove_duplicates_v2(lines,remove_duplicates_params):
    no_duplicates = [lines[0]]
    for line in lines[1:]:
        no_duplicates = incorporate_new_line(line,no_duplicates,remove_duplicates_params)
    return np.array(no_duplicates)

def remove_duplicates_v3(lines,remove_duplicates_params,device):

    if lines.shape[0] == 0:
        return lines

    lines_torch = torch.Tensor(lines).to(device)
    lines1,lines2 = create_all_possible_combinations_2(lines_torch,lines_torch)
    n_percent_1,n_percent_2 = compare_lines_to_lines(lines1,lines2,remove_duplicates_params["pixel_threshold"],remove_duplicates_params["points_per_line"])
    percent_line_overlap = remove_duplicates_params["percent_line_overlap"]

    n_percent_1 = n_percent_1.cpu().view(lines.shape[0],lines.shape[0])
    n_percent_2 = n_percent_2.cpu().view(lines.shape[0],lines.shape[0])

    n_percent_1 = n_percent_1.fill_diagonal_(0)
    n_percent_2 = n_percent_2.fill_diagonal_(0)

    assert (n_percent_1 == torch.transpose(n_percent_2,dim0=1,dim1=0)).all()

    # print(n_percent_1)
    # print(n_percent_2)

    already_added = []
    lines_no_duplicates = []
 
    for i in range(n_percent_1.shape[0]):
        # print(n_percent_1[i])
        if i in already_added:
            continue
        # print(n_percent_1[i])
        value1 = torch.max(n_percent_1[i])
        value2 = torch.max(n_percent_2[i])
        index2 = torch.argmax(n_percent_2[i])
        if value1 < percent_line_overlap and value2 < percent_line_overlap:
            if i not in already_added:
                lines_no_duplicates.append(lines[i])
                already_added.append(i)

        elif value1 >= percent_line_overlap and value2 < percent_line_overlap:
            if index2 not in already_added:
                lines_no_duplicates.append(lines[index2])
                already_added.append(index2)

        elif value1 < percent_line_overlap and value2 >= percent_line_overlap:
            if i not in already_added:
                lines_no_duplicates.append(lines[i])
                already_added.append(i)


        elif value1 >= percent_line_overlap and value2 >= percent_line_overlap:
            if i not in already_added:
                lines_no_duplicates.append(lines[i])
                already_added.append(i)

    # print(lines_no_duplicates)
    return np.array(lines_no_duplicates)




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


def expand_bbox(bbox,absolute_increase_bbox):
    expanded_bbox = [bbox[0] - absolute_increase_bbox,bbox[1] - absolute_increase_bbox,bbox[2] + absolute_increase_bbox,bbox[3] + absolute_increase_bbox]
    return expanded_bbox

def main():

    device = torch.device('cuda:0')

    params = {"increase_bbox_percent_whole_image": 0.05,"threshold_for_filtering_percent_whole_image": 0.04,"threshold_for_removing_duplicates": 0.02}

    target_folder = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val'

    remove_duplicates_params = {"percent_line_overlap": 0.97, "relative_threshold":0.005,"points_per_line":100}

    path_roca_detection_with_gt = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/all_detection_infos_fixed_category.json'
    with open(path_roca_detection_with_gt,'r') as f:
        roca_detection_with_gt = json.load(f)

    for gt_name in tqdm(sorted(roca_detection_with_gt)[:10000]):

        for i in range(len(roca_detection_with_gt[gt_name])):
    
            input_path_img = target_folder +  '/images/' + gt_name.split('.')[0] + '.jpg'
            input_path_lines = target_folder + '/lines_2d_cropped/' + gt_name.split('.')[0] + '.npy'

            if not os.path.exists(input_path_lines):
                continue

            detection = roca_detection_with_gt[gt_name][i]['detection']
            output_path_lines = target_folder + '/lines_2d_filtered/' + detection + '.npy'
            vis_path = target_folder + '/lines_2d_filtered_vis/' + detection + '.png'

            bbox = roca_detection_with_gt[gt_name][i]['bbox']


            lines = np.load(input_path_lines).astype(float)
            assert os.path.exists(input_path_img),input_path_img
            img = cv2.imread(input_path_img)
            h,w,_ = img.shape
            threshold_pixel_bbox = np.max([h,w]) * params["threshold_for_filtering_percent_whole_image"]
            threshold_pixel_remove_duplicates = np.max([h,w]) * params["threshold_for_removing_duplicates"]
            absolute_increase_bbox = np.max([h,w]) * params["increase_bbox_percent_whole_image"]
            remove_duplicates_params["pixel_threshold"] = np.max([h,w]) * remove_duplicates_params["relative_threshold"]



            expanded_bbox = expand_bbox(bbox,absolute_increase_bbox)

            filtered_lines = filter_lines(lines,expanded_bbox,threshold_pixel_bbox)

            no_duplicates = remove_duplicates(filtered_lines,threshold_pixel_remove_duplicates)

            no_duplicates = remove_duplicates_v3(no_duplicates,remove_duplicates_params,device)

            no_duplicates = np.round(no_duplicates).astype(int)
            np.save(output_path_lines,no_duplicates)

            visualise_lines(lines,filtered_lines,no_duplicates,bbox,expanded_bbox,img,vis_path,threshold_pixel_bbox,threshold_pixel_remove_duplicates,absolute_increase_bbox)



    

if __name__ == '__main__':
    print('Filter Lines')
    main()