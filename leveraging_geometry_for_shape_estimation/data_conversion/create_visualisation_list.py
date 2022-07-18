import os
import json
import sys


def get_img_list(image_folder,categories,vis_per_cat):
    cat_dict = {}
    for cat in categories:
        cat_dict[cat] = 0

    img_list = []
    for name in os.listdir(image_folder):
        cat = name.split('_')[0]
        if cat_dict[cat] < vis_per_cat:
            img_list.append(name)
            cat_dict[cat] += 1
    return img_list



def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["dataset"]["which_dataset"] == 'pix3d':

        target_folder = global_config["general"]["target_folder"]
        image_folder = global_config["general"]["image_folder"]
        categories = global_config["dataset"]["categories"]
        vis_per_cat = global_config["general"]["visualisations_per_category"]

        img_list = get_img_list(image_folder,categories,vis_per_cat)

        with open(target_folder + '/global_stats/visualisation_images.json','w') as f:
            json.dump(img_list,f)

if __name__ == '__main__':
    print('Create visualisation list')
    main()