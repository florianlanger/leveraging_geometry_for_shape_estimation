import torch
import matplotlib.pyplot as plt

# add path for demo utils functions 
import sys
import os
import json
from glob import glob

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
    target_folder += '/models/render_no_background/'

    make_empty_folder_structure(target_folder,target_folder.replace('render_no_background','render_black_background'))

    for old_path in glob(target_folder + '*/*/*'):
        new_path = old_path.replace('render_no_background','render_black_background')
        make_black_background(old_path,new_path)



if __name__ == '__main__':
    main()
