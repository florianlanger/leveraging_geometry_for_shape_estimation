
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


def get_best_nn(target_folder,top_n_retrieval,indicator,min_or_max,categories,evaluate_all,n_rotations):

    cats_best_possible_nn = {}
    cats_best_possible_orientation = {}
    cats_best_combined = {}
    for cat in categories:
        cats_best_possible_nn[cat] = []
        cats_best_possible_orientation[cat] = []
        cats_best_combined[cat] = []

    count_no_retrieval = 0
   
    for name in tqdm(sorted(os.listdir(target_folder + '/cropped_and_masked'))):
        # if not "scene0663_01-000000" in name and not "scene0653_00-001300" in name:
        #     continue


        indicator_scores = []
        Fs = []

        if not os.path.exists(target_folder + '/nn_infos/' + name.split('.')[0] + '.json'):
            count_no_retrieval += 1
            continue

        with open(target_folder + '/nn_infos/' + name.split('.')[0] + '.json','r') as open_f:
            retrieval_list = json.load(open_f)["nearest_neighbours"]

        with open(target_folder + '/bbox_overlap/' + name.split('.')[0] + '.json','r') as f:
            bbox_overlap = json.load(f)

        n_neighbours = min([top_n_retrieval,len(retrieval_list)])

        if n_neighbours == 0:
            count_no_retrieval += 1

        for i in range(n_neighbours):

            for k in range(n_rotations):

                with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
                    gt_infos = json.load(f)

                with open(target_folder + '/poses/' + name.split('.')[0] + '_' + str(i).zfill(3) + '_' + str(k).zfill(2) + '.json','r') as f:
                    poses = json.load(f)

                indicator_scores.append(poses[indicator])

                if evaluate_all:
                    with open(target_folder + '/metrics/' + name.split('.')[0] + '_' + str(i).zfill(3) + '_' + str(k).zfill(2) + '.json','r') as f:
                        metrics = json.load(f)
                    
                    Fs.append(metrics['F1'])

        if min_or_max == 'min':
            best_value = min(indicator_scores)
        elif min_or_max == 'max':
            best_value = max(indicator_scores)

        
        selected = indicator_scores.index(best_value)
        selected_nn = selected // n_rotations
        selected_orientation = selected % n_rotations


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

    print('N cropped_and_masked ',len(os.listdir(target_folder + '/cropped_and_masked')))
    print('N no retrieval ', count_no_retrieval)
    print('remaining ',len(os.listdir(target_folder + '/cropped_and_masked')) - count_no_retrieval)
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
    if global_config["pose_and_shape_probabilistic"]["use_probabilistic"] == "True":
        n_rotations = 4
    else:
        n_rotations = 1
    
    cats_best_possible_nn,cats_best_possible_orientation,cats_best_combined = get_best_nn(target_folder,number_nn,indicator,min_or_max,categories,evaluate_all,n_rotations)

    plot_nn_selection(cats_best_possible_nn,cats_best_possible_orientation,cats_best_combined,target_folder + '/global_stats/nn_selection.png')





if __name__ == '__main__':
    print('Select BEst ')
    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    write_aps(global_config)

    
    
    
    
  