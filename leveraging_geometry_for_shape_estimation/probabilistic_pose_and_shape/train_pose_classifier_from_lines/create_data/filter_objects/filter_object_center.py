from platform import dist
import re
from cv2 import threshold
import numpy as np
import cv2
import os
import sys
import json
from tqdm import tqdm
import shutil



def main():
    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    target_folder = global_config["general"]["target_folder"]

    for name in tqdm(sorted(os.listdir(target_folder + '/lines_2d_filtered'))):

        with open(target_folder + '/T_in_image/' + name.replace('.npy','.json'),'r') as file:
            T_in_image = json.load(file)

        if T_in_image['in_image'] == True:
            shutil.copy(target_folder + '/lines_2d_filtered/' + name,target_folder + '/lines_2d_filtered_only_object_center/' + name)
        



    

if __name__ == '__main__':
    print('Filter Lines')
    main()