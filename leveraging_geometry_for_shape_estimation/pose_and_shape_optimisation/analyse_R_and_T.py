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

def plot_hist(data,cat,dir_path):

    plt.hist(data)
    plt.title(cat + ' mean: ' + str(np.round(np.mean(data),3)) + 'median: ' + str(np.round(np.median(data),3)))
    plt.savefig(dir_path + '/' + cat + '.png')
    plt.close()


def analyse_R_and_T(global_config):

    target_folder = global_config["general"]["target_folder"]

    trans_normalised = {}
    for cat in global_config["dataset"]["categories"]:
        trans_normalised[cat] = []

    # for name in os.listdir(target_folder + '/metrics'):
    for name in tqdm(os.listdir(target_folder + '/cropped_and_masked')):

        with open(target_folder + '/segmentation_infos/' + name.split('.')[0] + '.json','r') as f:
            segmentation_infos = json.load(f)

        with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
            gt_infos = json.load(f)

        with open(target_folder + '/bbox_overlap/' + name.split('.')[0] + '.json','r') as f:
            bbox_overlap = json.load(f)
            
        # with open(target_folder + '/segmentation_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
        #     segmentation_infos = json.load(f)

        # with open(target_folder + '/gt_infos/' + name.rsplit('_',2)[0] + '.json','r') as f:
        #     gt_infos = json.load(f)


        with open(target_folder + '/selected_nn/' + name.split('.')[0] + '.json','r') as f:
            selected = json.load(f)
        
        with open(target_folder + '/metrics/' + name.split('.')[0] + '_' + str(selected["selected_nn"]).zfill(3) + '_' + str(selected["selected_orientation"]).zfill(2) + '.json','r') as f:
            metrics = json.load(f)
        # with open(target_folder + '/metrics/' + name,'r') as f:
        #     metrics = json.load(f)

        if segmentation_infos["predictions"]["category"] == gt_infos["objects"][bbox_overlap['index_gt_objects']]["category"]:
            trans_normalised[gt_infos["objects"][bbox_overlap['index_gt_objects']]["category"]].append(metrics["diff_normalised_length"])

    imgs = []
    for cat in global_config["dataset"]["categories"]:
        plot_hist(trans_normalised[cat],cat,target_folder + '/global_stats/T_hists')
        imgs.append(cv2.imread(target_folder + '/global_stats/T_hists/' + cat + '.png'))

    row_1 = cv2.hconcat(imgs[:3])
    row_2 = cv2.hconcat(imgs[3:6])
    row_3 = cv2.hconcat(imgs[6:9])
    combined = cv2.vconcat([row_1,row_2,row_3])
    cv2.imwrite(target_folder + '/global_stats/T_hists.png',combined)

    cats_interest = ['bed','bookcase','desk','sofa','table','wardrobe']
    means = [np.mean(trans_normalised[cat]) for cat in cats_interest]
    medians = [np.median(trans_normalised[cat]) for cat in cats_interest]
    print('cats interest mean ', np.round(np.mean(means),5), '  median ', np.round(np.mean(medians),5))





def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)


    analyse_R_and_T(global_config)

if __name__ == '__main__':
    main()