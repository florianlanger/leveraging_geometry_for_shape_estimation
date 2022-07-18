import os
import json
import sys
import random

from leveraging_geometry_for_shape_estimation.utilities.dicts import determine_base_dir

def get_img_list(image_folder,categories,vis_per_cat):
    img_list = []

    img_list = os.listdir(image_folder)
    random.shuffle(img_list)

    return img_list[:len(categories)*vis_per_cat]



def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["dataset"]["which_dataset"] == 'scannet' or global_config["dataset"]["which_dataset"] == 'future3d':

        target_folder = global_config["general"]["target_folder"]
        image_folder = global_config["general"]["image_folder"]
        image_folder = determine_base_dir(global_config,"segmentation") + '/images'
        categories = global_config["dataset"]["categories"]
        vis_per_cat = global_config["general"]["visualisations_per_category"]

        img_list = get_img_list(image_folder,categories,vis_per_cat)

        # with open(target_folder + '/global_stats/visualisation_images_actual.json','w') as f:
        #     json.dump(img_list,f)

        with open(target_folder + '/global_stats/visualisation_images.json','w') as f:
            json.dump(img_list,f)

if __name__ == '__main__':
    print('Create visualisation list')
    main()