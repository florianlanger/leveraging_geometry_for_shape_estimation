import csv
import json
from unicodedata import category
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from leveraging_geometry_for_shape_estimation.data_conversion_scannet.get_infos import get_scene_pose,get_scene_pose_2
import quaternion
import SE3
import CSVHelper
import sys
from operator import itemgetter


def main():

    with open('/scratch2/fml35/results/ROCA/per_frame_best.json','r') as f:
        per_frame = json.load(f)


    per_frame_no_null = {}
    for image in tqdm(per_frame):
        for detection in per_frame[image]:
            if type(detection["scene_cad_id"]) == list:
                cad_id = detection["scene_cad_id"][-1]
            else:
                cad_id = detection["scene_cad_id"]

            if cad_id == None:
                continue

            if image in per_frame_no_null:
                per_frame_no_null[image].append(detection)
            else:
                per_frame_no_null[image] = [detection]

    with open('/scratch2/fml35/results/ROCA/per_frame_best_no_null.json','w') as f:
        json.dump(per_frame_no_null,f)
    
if __name__ == '__main__':

    main()