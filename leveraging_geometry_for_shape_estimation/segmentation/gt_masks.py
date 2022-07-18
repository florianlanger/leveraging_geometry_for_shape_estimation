import os
import json
from tqdm import tqdm
import cv2
from imantics import Mask
import sys

def main():


    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["segmentation"]["use_gt"] == "True":


        target_folder = global_config["general"]["target_folder"]
        mask_folder = global_config["general"]["mask_folder"]
        # image_folder = global_config["general"]["image_folder"]


        for name in tqdm(os.listdir(mask_folder)):


            out_path = target_folder + '/segmentation_masks/' + name


            with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
                gt_infos = json.load(f)

            object_id = int(name.rsplit('_',1)[-1].split('.')[0])

            # if os.path.exists(out_path):
            #     continue


            # img_path = image_folder + '/' + name

            mask = cv2.imread(mask_folder + '/' + name)
            h,w = mask.shape[:2]
            mask_imantics = Mask(mask)

            info_folder = {}
            info_folder["detection"] = name.split('.')[0]
            info_folder["img"] = gt_infos["img"]
            info_folder["img_size"] = [w,h]
            info_folder["predictions"] = {}
            info_folder["predictions"]["bbox"] = list(mask_imantics.bbox())
            info_folder["predictions"]["category"] = gt_infos["objects"][object_id]["category"]
            info_folder["predictions"]["score"] = 1.0

            with open(target_folder + '/segmentation_infos/' + name.split('.')[0] + '.json','w') as f:
                json.dump(info_folder, f)

            
            cv2.imwrite(out_path,mask)

            

if __name__ == '__main__':
    print('Get Segmentation Masks')
    main()