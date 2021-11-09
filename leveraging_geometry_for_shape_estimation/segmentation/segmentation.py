# import some common libraries
import numpy as np
import os, json, cv2, random
from tqdm import tqdm
from datetime import datetime
import torch
# import some common detectron2 utilities
import matplotlib
matplotlib.use('Agg')

from pycocotools.mask import decode
from PIL import Image
# from mmdet.models import build_detector

import sys
import shutil
import argparse
from datetime import datetime

print('import libraries')
sys.path.insert(0,'/home/mifs/fml35/code/segmentation/mmdet')
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
# import mmcv
print('done')

def visualise_prediction(model,name,class_dict,target_folder,image_folder,testing_threshold,padding,visualise=True):


    img_path = image_folder + '/' + name
    result = inference_detector(model, img_path)
    
    info = result[0]
    mask_preds = result[1]

    h,w = cv2.imread(img_path).shape[:2]

    boxes = []
    pred_categories = []
    scores = []
    masks = []
    for i in range(len(class_dict)):
        for k in range(info[i].shape[0]):

            boxes.append(info[i][k,:4].tolist())
            pred_categories.append(class_dict[i])
            scores.append(float(info[i][k,4]))
            masks.append(mask_preds[i][k] * 255)


    sort_indices = np.argsort(scores)[::-1]

    pred_categories = [pred_categories[index] for index in sort_indices]
    boxes = [boxes[index] for index in sort_indices]
    scores = [scores[index] for index in sort_indices]
    masks = [masks[index] for index in sort_indices]

    first_name = name.split('.')[0]
    

    for counter in range(len(scores)):
            # detection_name = first_name + '_' + str(counter).zfill(2)
            detection_name = first_name + '_' + str(counter)

            # pad_x,pad_y = padding

            box = np.array(boxes[counter])
            # cropped_box = box - np.array([pad_x,pad_y,pad_x,pad_y])
            # cropped_box = np.clip(cropped_box,a_min=[0,0,0,0],a_max=[w-1,h-1,w-1,h-1])


            info_folder = {}
            info_folder["detection"] = detection_name
            info_folder["img"] = name
            info_folder["img_size"] = [w,h]
            info_folder["predictions"] = {}
            info_folder["predictions"]["bbox"] = box.tolist()
            info_folder["predictions"]["category"] = pred_categories[counter] 
            info_folder["predictions"]["score"] = scores[counter]

            with open(target_folder + '/segmentation_infos/' + detection_name + '.json','w') as f:
                json.dump(info_folder, f)

            mask = masks[counter]
            
            # cropped_mask = mask[pad_y:pad_y+h,pad_x:pad_x+w]
            cv2.imwrite(target_folder + '/segmentation_masks/' + detection_name + '.png',mask)
            # cv2.imwrite(target_folder + '/segmentation_masks/' + detection_name + '.' + name.split('.')[1],mask)
            
    if visualise:
        combined_output = model.show_result(img_path, result, score_thr=testing_threshold)
        cv2.imwrite(target_folder + '/segmentation_vis/' + first_name + '.png',combined_output)

def make_directories(target_directory):

    assert os.path.exists(target_directory)

    os.mkdir(target_directory + '/segmentation_masks')
    os.mkdir(target_directory + '/segmentation_infos')
    os.mkdir(target_directory + '/segmentation_vis')


def load_classes(classes_list):

    classes = {}
    for i,cat in enumerate(classes_list):
        classes[i] = cat

    return classes

def main():


    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["segmentation"]["use_gt"] == "False":



        target_folder = global_config["general"]["target_folder"]
        config_file = global_config["segmentation"]["config"]
        checkpoint_file = global_config["segmentation"]["checkpoint"]
        image_folder = global_config["general"]["image_folder"]
        testing_threshold = global_config["segmentation"]["threshold"]

        device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))
        # device = torch.device("cpu")
        torch.cuda.set_device(device)
        model = init_detector(config_file, checkpoint_file, device='cpu')
        print(model.CLASSES)
        # model = init_detector(config_file, checkpoint_file, device='cuda:{}'.format(args.gpu))
        model.to(device)
        print('after model')

        class_dict = load_classes(model.CLASSES)

        for name in tqdm(os.listdir(image_folder)):

            # if name == 'bed_0020.png':

            with torch.no_grad():
                padding = [0,0]
                visualise_prediction(model,name,class_dict,target_folder,image_folder,testing_threshold,padding,visualise=True)



if __name__ == '__main__':
    print('Get Segmentation Masks')
    main()