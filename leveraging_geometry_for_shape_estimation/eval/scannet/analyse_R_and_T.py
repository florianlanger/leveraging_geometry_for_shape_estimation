import matplotlib
matplotlib.use('Agg')
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
from matplotlib import pyplot as plt
from scipy import stats

from leveraging_geometry_for_shape_estimation.utilities.dicts import open_json_precomputed_or_current,determine_base_dir

def plot_hist(data,cat,dir_path):

    plt.hist(data)
    plt.title(cat + ' mean: ' + str(np.round(np.mean(data),3)) + 'median: ' + str(np.round(np.median(data),3)))
    plt.savefig(dir_path + '/' + cat + '.png')
    plt.close()


def analyse_R_and_T(global_config):

    target_folder = global_config["general"]["target_folder"]

    # t_error = {}
    # for cat in global_config["dataset"]["categories"]:
    #     t_error[cat] = []
    t = []
    r = []
    s = []
    retrieval = []

    categories_gt = {'bin':232,'table':553,'bookshelf':212,"sofa":113,"chair":1093,"display":191,"bathtub":120,"cabinet":260,"bed":70,'all_instances':2844}
    
    # run_sum = 0
    # for cat in total_gt_per_cat:
    #     run_sum += total_gt_per_cat[cat]


    categories_correct = {}
    for cat in global_config["dataset"]["categories"] + ['all_instances']:
        # categories[cat] = {"total": 0, "correct": 0}
        categories_correct[cat] = 0
    

    max_rot_error = global_config["evaluate_poses"]["scannet"]["max_r_error"]


    for name in tqdm(os.listdir(target_folder + '/selected_nn')):

        # with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
        #     gt_infos = json.load(f)

        with open(target_folder + '/selected_nn/' + name.split('.')[0] + '.json','r') as f:
            selected = json.load(f)
        # selected = {}
        # selected["selected_nn"] = 0
        # selected["selected_orientation"] = 0

        
        # if not os.path.exists(target_folder + '/metrics_scannet/' + name.split('.')[0] + '_' + str(selected["selected_nn"]).zfill(3) + '_' + str(selected["selected_orientation"]).zfill(2) + '.json'):
        #     continue

        with open(target_folder + '/metrics_scannet/' + name.split('.')[0] + '_' + str(selected["selected_nn"]).zfill(3) + '_' + str(selected["selected_orientation"]).zfill(2) + '.json','r') as f:
            metrics = json.load(f)

        # with open(target_folder + '/bbox_overlap/' + name.split('.')[0] + '.json','r') as f:
        #     bbox_overlap = json.load(f)

        gt_infos = open_json_precomputed_or_current('/gt_infos/' + name.rsplit('_',1)[0] + '.json',global_config,'segmentation')
        bbox_overlap = open_json_precomputed_or_current('/bbox_overlap/' + name.split('.')[0] + '.json',global_config,'segmentation')


        # between 0 and 180
        rot_error = metrics["rotation_error"]
        # between 0 and 90
        rot_error = rot_error % 90
        rot_error = min(rot_error,90-rot_error)
        rot_correct = rot_error < max_rot_error



        t.append(1*metrics["translation_correct"])
        r.append(1*metrics["rotation_correct"])
        # r.append(1*rot_correct)
        s.append(1*metrics["scaling_correct"])
        retrieval.append(1*metrics["retrieval_correct"])


        if metrics["translation_correct"] and metrics["rotation_correct"] and metrics["scaling_correct"]:
            cat = gt_infos["objects"][bbox_overlap['index_gt_objects']]["category"]
            categories_correct[cat] += 1
            categories_correct['all_instances'] += 1




    text = 'TOTAL DETECTIONS: ' + str(len(t)) + '\n'
    text += 'T correct: {}/{} ({} %)]\n'.format(sum(t),len(t),np.round(100*sum(t)/len(t)),3)
    text += 'R correct: {}/{} ({} %)\n'.format(sum(r),len(r),np.round(100*sum(r)/len(r)),3)
    text += 'S correct: {}/{} ({} %)\n'.format(sum(s),len(s),np.round(100*sum(s)/len(s)),3)
    text += 'Retrieval correct: {}/{} ({} %)\n'.format(sum(retrieval),len(retrieval),np.round(100*sum(retrieval)/len(retrieval)),3)

    print(text)
    # with open(target_folder + '/global_stats/scannet_metrics_all_4_rot_correct.txt','w') as f:
    #     f.write(text)
    with open(target_folder + '/global_stats/scannet_metrics.txt','w') as f:
        f.write(text)

    accuracy_categories = {}
    for cat in categories_gt:
        accuracy_categories[cat] = 100 * float(categories_correct[cat]) / categories_gt[cat]

    print(accuracy_categories)




def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)


    analyse_R_and_T(global_config)

if __name__ == '__main__':
    main()