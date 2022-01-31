
import os
import shutil
import json
import sys
import random

def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    target_folder = global_config["general"]["target_folder"]

    split = global_config["dataset"]["split"]
    pix_path = global_config["dataset"]["pix3d_path"]

    if global_config["general"]["debug_set"] == "True":
        max_count = 1
    elif global_config["general"]["debug_set"] == "False":
        max_count = 2000000

    with open('data/pix3d_{}_test.json'.format(split),'r') as f:
        pix = json.load(f)

    images = []
    indices = list(range(len(pix["images"])))
    for index in indices:
        images.append(pix["images"][index]["file_name"])

    counter_cat = {}
    for cat in global_config["dataset"]["categories"]:
        counter_cat[cat] = 0

    # copy images
    for cat in os.listdir(pix_path + '/img'):

        list_imgs = os.listdir(pix_path + '/img/' + cat)
        random.shuffle(list_imgs)

        for img in list_imgs:
            image_file_name = 'img/' + cat + '/' + img
            if image_file_name in images:

                # examples with imgfiles = {img/table/1749.jpg, img/table/0045.png}
                # have a mismatch between images and masks. Thus, ignore
                if image_file_name in ["img/table/1749.jpg", "img/table/0045.png"]:
                    continue

                if counter_cat[cat] >= max_count:
                    continue

                # copy img
                old_path = pix_path + 'img/' + cat + '/' + img
                new_path = target_folder  + '/images/' + cat + '_' + img
                shutil.copyfile(old_path,new_path)

                # copy mask
                old_path = pix_path + '/mask/' + cat + '/' + img.split('.')[0] + '.png'
                new_path = target_folder +  '/masks/' + cat + '_' + img.split('.')[0] + '.png'
                shutil.copyfile(old_path,new_path)

                counter_cat[cat] += 1

        if global_config["general"]["debug_set"] == "True" and global_config["general"]["debug_single"]:
            break

if __name__ == '__main__':
    random.seed(0)
    main()