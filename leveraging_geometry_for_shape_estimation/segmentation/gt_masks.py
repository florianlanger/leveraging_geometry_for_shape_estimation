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
        image_folder = global_config["general"]["image_folder"]


        for name in tqdm(os.listdir(image_folder)):

            first_name = name.split('.')[0]
    
            detection_name = first_name + '_' + str(0)

            out_path = target_folder + '/segmentation_masks/' + detection_name + '.png'
            if os.path.exists(out_path):
                continue


            img_path = image_folder + '/' + name


            h,w = cv2.imread(img_path).shape[:2]

            mask = cv2.imread(mask_folder + '/' + name.split('.')[0] + '.png')
            print(mask_folder + '/' + name.split('.')[0] + '.png')
            print(mask.shape)
            mask_imantics = Mask(mask)

            info_folder = {}
            info_folder["detection"] = detection_name
            info_folder["img"] = name
            info_folder["img_size"] = [w,h]
            info_folder["predictions"] = {}
            info_folder["predictions"]["bbox"] = list(mask_imantics.bbox())
            info_folder["predictions"]["category"] = name.split('_')[0]
            info_folder["predictions"]["score"] = 1.0

            with open(target_folder + '/segmentation_infos/' + detection_name + '.json','w') as f:
                json.dump(info_folder, f)

            
            cv2.imwrite(out_path,mask)

            

if __name__ == '__main__':
    print('Get Segmentation Masks')
    main()