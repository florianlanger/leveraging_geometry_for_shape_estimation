
import os
import shutil
import json
import sys
import random
import imagesize
from tqdm import tqdm
from pycocotools import mask as mutils
from pycocotools import _mask as coco_mask
import cv2
import numpy as np


def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["dataset"]["which_dataset"] == 'scannet' and global_config["segmentation"]["use_gt"] == 'roca':

        target_folder = global_config["general"]["target_folder"]

        with open(global_config["dataset"]["roca_results"],'r') as file:
            roca = json.load(file)

        for img in tqdm(roca):
    
            counter = 0
            for detection in roca[img]:

                if not os.path.exists(target_folder + '/gt_infos/' + img.split('/')[0] + '-' + img.split('/')[2].split('.')[0] + '.json'):
                    continue

                with open(target_folder + '/gt_infos/' + img.split('/')[0] + '-' + img.split('/')[2].split('.')[0] + '.json','r') as file:
                    gt_infos = json.load(file)
            
                # ignore images without gt annotation
                # if gt_infos["objects"] == []:
                #     continue


                new_dict = {}
                new_dict["detection"] = img.split('/')[0] + '-' + img.split('/')[2].split('.')[0] + '_' + str(counter).zfill(2)
                new_dict["img"] = img.split('/')[0] + '-' + img.split('/')[2]
                new_dict["img_size"] = imagesize.get(target_folder + '/images/' + new_dict["img"])
                new_dict["predictions"] = {}
                new_dict["predictions"]["bbox"] = (detection["bbox"] / np.array([480,360,480,360])  * np.array(new_dict["img_size"] + new_dict["img_size"])).tolist()
                new_dict["predictions"]["category"] = detection["category"].replace('bookcase','bookshelf')
                new_dict["predictions"]["score"] = detection["score"]

                with open(target_folder + '/segmentation_infos/' + new_dict["detection"] + '.json','w') as file:
                    json.dump(new_dict,file)

                mask = coco_mask.decode([detection["segmentation"]])
                binaryMask = mask.astype('bool') * 255
                binaryMask = cv2.resize(binaryMask.astype(np.uint8), tuple(new_dict["img_size"]),interpolation = cv2.INTER_CUBIC)
                
                cv2.imwrite(target_folder + '/segmentation_masks/' + new_dict["detection"] + '.png',binaryMask)
                counter += 1

if __name__ == '__main__':
    print('SEGMENTATION ROCA')
    main()