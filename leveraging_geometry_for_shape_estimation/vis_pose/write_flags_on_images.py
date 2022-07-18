

import numpy as np
from torch import det
np.warnings.filterwarnings('ignore')
import pathlib
import subprocess
import os
import collections
import shutil
import quaternion
import operator
import glob
import csv
import re

import argparse
np.seterr(all='raise')
import argparse
import json
import sys
import cv2
from tqdm import tqdm

from leveraging_geometry_for_shape_estimation.eval.scannet import CSVHelper,SE3,JSONHelper
from leveraging_geometry_for_shape_estimation.utilities.write_on_images import draw_text,draw_text_block
from leveraging_geometry_for_shape_estimation.utilities.dicts import open_json_precomputed_or_current,determine_base_dir,load_json

def load_filtered_and_flags(target_folder):
    filtered = {}
    scan2cad_constraint = {}
    for file0 in os.listdir(target_folder + '/global_stats/eval_scannet/results_per_scene_filtered'):
        scene = file0.split('.')[0]
        filtered[scene] = CSVHelper.read(target_folder + '/global_stats/eval_scannet/results_per_scene_filtered/' + file0)
        scan2cad_constraint[scene] = CSVHelper.read(target_folder + '/global_stats/eval_scannet/results_per_scene_flags_without_retrieval/' + file0)
    return filtered,scan2cad_constraint




def write_flags(target_folder):

    filtered,scan2cad_constraint = load_filtered_and_flags(target_folder)

    # for img_file in tqdm(sorted(os.listdir(target_folder + "/poses_vis_no_flags"))):
    for img_file in tqdm(sorted(os.listdir(target_folder + "/poses_vis"))):


        scene = img_file.split('-')[0]
        detection = img_file.rsplit('_',2)[0]

        # with open(target_folder + '/segmentation_infos/' +detection + '.json','r') as f:
        #     segmentation_infos = json.load(f)

        # if "scene0663_01" not in scene:
        #     continue

        detections_scan2cad_constraints = [alignment[14] for alignment in scan2cad_constraint[scene]]
        detections_after_filter = [alignment[14] for alignment in filtered[scene]]

        # img = cv2.imread(target_folder + "/poses_vis_no_flags/" + img_file)
        img = cv2.imread(target_folder + "/poses_vis/" + img_file)
        top_left_corner = (int(np.round(img.shape[1]/2)),5)
        if detection in detections_scan2cad_constraints:
            index = detections_scan2cad_constraints.index(detection)
            infos = scan2cad_constraint[scene][index]
            text = ['overall: ' + str(infos[15]),'trans: ' + str(infos[16]) + ' ' + str(np.round(float(infos[20]),4)),'rot: ' + str(infos[17]) + ' ' + str(np.round(float(infos[21]),4)),'scale: ' + str(infos[18]) + ' ' + str(np.round(float(infos[22]),4)),'retrieval: ' + str(infos[19]),'score ' + str(np.round(float(infos[12]),5))]
            draw_text_block(img,text,top_left_corner,font_scale=1,font_thickness=1)

        elif detection in detections_after_filter:
            index = detections_after_filter.index(detection)
            infos = filtered[scene][index]
            draw_text_block(img,['survived filter','score ' + str(np.round(float(infos[12]),5))],top_left_corner,font_scale=1,font_thickness=1)

        else:
            draw_text_block(img,['filtered out'],top_left_corner,font_scale=1,font_thickness=1)

        out_path = target_folder + "/poses_vis/" + img_file # .replace('.png','_{}.png'.format(segmentation_infos["predictions"]["category"]))
        cv2.imwrite(out_path,img)
        

        
        

        


if __name__ == "__main__":
    print('Write Flags')

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    target_folder = sys.argv[1]

    write_flags(target_folder)