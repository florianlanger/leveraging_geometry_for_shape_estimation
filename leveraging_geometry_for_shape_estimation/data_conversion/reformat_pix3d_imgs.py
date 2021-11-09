
import os
import shutil
import json
import sys


def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    target_folder = global_config["general"]["target_folder"] + '/images'

    split = global_config["dataset"]["split"]
    pix_path = global_config["dataset"]["pix3d_path"]


    with open('data/pix3d_{}_test.json'.format(split),'r') as f:
        pix = json.load(f)

    images = []
    for i in range(len(pix["images"])):
        images.append(pix["images"][i]["file_name"])


    # copy images
    for cat in os.listdir(pix_path + '/img'):
        for img in os.listdir(pix_path + '/img/' + cat):
            image_file_name = 'img/' + cat + '/' + img
            if image_file_name in images:

                # examples with imgfiles = {img/table/1749.jpg, img/table/0045.png}
                # have a mismatch between images and masks. Thus, ignore
                if image_file_name in ["img/table/1749.jpg", "img/table/0045.png"]:
                    continue

                # copy img
                old_path = pix_path + '/' + cat + '/' + img
                new_path = target_folder + '/' + cat + '_' + img
                shutil.copyfile(old_path,new_path)

                # copy mask
                old_path = pix_path + '/mask/' + cat + '/' + img.split('.')[0] + '.png'
                new_path = target_folder + '/' + cat + '_' + img.split('.')[0] + '.png'
                shutil.copyfile(old_path,new_path)

if __name__ == '__main__':
    main()