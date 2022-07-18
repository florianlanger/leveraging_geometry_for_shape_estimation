
from operator import gt
import torch
import numpy as np
import json
import sys
import argparse
import os
from tqdm import tqdm
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from leveraging_geometry_for_shape_estimation.utilities.dicts import open_json_precomputed_or_current,determine_base_dir

def get_angle(m1,m2):

    m = np.matmul(np.array(m1).T,np.array(m2))

    value = (np.trace(m) - 1 )/ 2

    clipped_value = np.clip(value,-0.9999999,0.999999)

    angle = np.arccos(clipped_value)

    return angle * 180 / np.pi 

def plot_nn_selection(cats_best_possible_nn,cats_best_possible_orientation,cats_best_combined,save_path):

    total = np.array([len(cats_best_possible_nn[cat]) for cat in cats_best_possible_nn])
    best_model = np.array([sum(np.array(cats_best_possible_nn[cat])) for cat in cats_best_possible_nn]).astype(float)
    best_orientation = np.array([sum(np.array(cats_best_possible_orientation[cat])) for cat in cats_best_possible_orientation]).astype(float)
    best_combined = np.array([sum(np.array(cats_best_combined[cat])) for cat in cats_best_combined]).astype(float)

    width = 0.2

    x = np.array(range(len(cats_best_possible_nn)))
    

    ax = plt.subplot(111)
    # ax.bar(x-width, y, width=width, color='b', align='center')
    ax.bar(x-width, best_model/total, width=width, color='r', align='center',label='best model')
    ax.bar(x, best_orientation/total, width=width, color='g', align='center',label='best orientation')
    ax.bar(x+width, best_combined/total, width=width, color='b', align='center',label='best combined')

    # ax.set_xticklabels([cat for cat in cats_best_possible],minor=True)
    plt.xticks(x, [cat for cat in cats_best_possible_nn])
    plt.legend()

    # rects = ax.patches
    # for rect, label in zip(rects, list(total)):
    #     height = rect.get_height()
    #     ax.text(
    #         rect.get_x() + rect.get_width() / 2, height + 0.05, label, ha="center", va="bottom"
    #     )

    plt.savefig(save_path)


def get_best_nn(target_folder,top_n_retrieval,indicator,min_or_max,categories,evaluate_all,n_rotations,global_config):

    cats_best_possible_nn = {}
    cats_best_possible_orientation = {}
    cats_best_combined = {}
    for cat in categories:
        cats_best_possible_nn[cat] = []
        cats_best_possible_orientation[cat] = []
        cats_best_combined[cat] = []

    count_no_retrieval = 0

    # print('ONLY TOP 1 NN, always select correct rotation')
   
    for name in tqdm(sorted(os.listdir(determine_base_dir(global_config,'segmentation') + '/poses_R_selected'))):
        # if not "scene0663_01-000000" in name and not "scene0653_00-001300" in name:
        #     continue


        indicator_scores = []
        Fs = []

        if not os.path.exists(target_folder + '/nn_infos/' + name.split('.')[0] + '.json'):
            count_no_retrieval += 1
            continue

        selected_R = open_json_precomputed_or_current('/poses_R_selected/' + name,global_config,"retrieval")["nearest_neighbours"]


        selected_nn = 0
        selected_orientation = selected_R["R_index"]


        with open(target_folder + '/selected_nn/' + name.split('.')[0] + '.json','w') as f:
            json.dump({"selected_nn":selected_nn, 'selected_orientation':selected_orientation},f)


        if evaluate_all:
            best_possible = Fs.index(max(Fs))
            best_nn = best_possible // n_rotations
            best_orientation = best_possible % n_rotations

            gt_cat = gt_infos["objects"][bbox_overlap['index_gt_objects']]['category']

            cats_best_possible_nn[gt_cat].append(selected_nn == best_nn)
            cats_best_possible_orientation[gt_cat].append(selected_orientation == best_orientation)
            cats_best_combined[gt_cat].append(selected_nn == best_nn and selected_orientation == best_orientation)



            with open(target_folder + '/selected_nn/' + name.split('.')[0] + '.json','w') as f:
                json.dump({"selected_nn":selected_nn,'best_nn': best_nn, 'selected_orientation':selected_orientation,'best_orientation':best_orientation},f)

    # print('N cropped_and_masked ',len(os.listdir(target_folder + '/cropped_and_masked')))
    # print('N no retrieval ', count_no_retrieval)
    # print('remaining ',len(os.listdir(target_folder + '/cropped_and_masked')) - count_no_retrieval)
    return cats_best_possible_nn,cats_best_possible_orientation,cats_best_combined


            



def write_aps(global_config):

    
    target_folder = global_config["general"]["target_folder"]

    number_nn = global_config["keypoints"]["matching"]["top_n_retrieval"]
    categories = global_config["dataset"]["categories"]

    setting_to_metric = {'segmentation': 'avg_dist_furthest', 'keypoints': 'avg_dist_reprojected_keypoints', 'combined':'combined','meshrcnn':'meshrcnn','F1':'F1','factor':'factor'}
    setting_to_min_or_max = {'segmentation': 'min', 'keypoints': 'min', 'combined':'min','meshrcnn':'check','F1':'max','factor':'max'}
    indicator = setting_to_metric[global_config["pose_and_shape"]["pose"]["choose_best_based_on"]]
    min_or_max = setting_to_min_or_max[global_config["pose_and_shape"]["pose"]["choose_best_based_on"]]
    evaluate_all = global_config["evaluate_poses"]["evaluate_all"]
    n_rotations = 1
    
    cats_best_possible_nn,cats_best_possible_orientation,cats_best_combined = get_best_nn(target_folder,number_nn,indicator,min_or_max,categories,evaluate_all,n_rotations,global_config)

    plot_nn_selection(cats_best_possible_nn,cats_best_possible_orientation,cats_best_combined,target_folder + '/global_stats/nn_selection.png')





if __name__ == '__main__':
    print('Select BEst ')
    print('USe R closest to gt')
    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    write_aps(global_config)

    
    
    
    
  