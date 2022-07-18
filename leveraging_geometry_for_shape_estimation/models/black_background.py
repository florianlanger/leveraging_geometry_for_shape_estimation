import torch
import matplotlib.pyplot as plt

# add path for demo utils functions 
import sys
import os
import json
from glob import glob
from tqdm import tqdm
import random

def make_empty_folder_structure(inputpath,outputpath):
    for dirpath, dirnames, filenames in os.walk(inputpath):
        structure = os.path.join(outputpath, dirpath[len(inputpath):])
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print("Folder {} already exists!".format(structure))


def make_black_background(old_path,new_path):
    image = plt.imread(old_path)
    alphas = image[:,:,3:] 
    new_image = image[:,:,:3] * alphas
    plt.imsave(new_path,new_image)


def main():
    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)


    target_folder = global_config["general"]["target_folder"]

    # make_empty_folder_structure(target_folder,target_folder.replace('render_no_background','render_black_background'))

    with open(target_folder + '/models/model_list.json','r') as f:
        model_list = json.load(f)['models']
    model_list = [item['model'].split('/')[-2] for item in model_list]

    target_folder += '/models/render_no_background/'

    files = glob(target_folder + '*/*/*')
    random.shuffle(files)


    # for old_path in tqdm(files):
    for old_path in tqdm(['/scratches/octopus/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/3d_future/data_01/models/render_no_background/cabinetshelfdesk/0b2cf6c9-2464-4219-8233-198a1f29776e/elev_045_azim_225.0.png']):

        model = old_path.split('/')[-2]
        if model not in model_list:
            continue


        new_path = old_path.replace('render_no_background','render_black_background')
        if os.path.exists(new_path):
            continue
        make_black_background(old_path,new_path)



if __name__ == '__main__':
    main()
