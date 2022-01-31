import cv2
import numpy as np
import sys
import os
import json
from tqdm import tqdm



def pixel_cropped_to_original(pixel,bb,max_bbox_length,img_size):
    bb_width = bb[2] - bb[0]
    bb_height = bb[3] - bb[1]
    ratio = float(bb_width)/bb_height
    if bb_width <= bb_height:
        resized_h = max_bbox_length
        resized_w = int(np.round(ratio * resized_h))
    else:
        resized_w = max_bbox_length
        resized_h = int(np.round(resized_w / ratio))
    # undo padding
    pixel -= np.array([int((img_size - resized_h)/2),int((img_size - resized_w)/2)])
    # undo resizing
    scaling_factor_real_image = max([bb_width,bb_height]) / float(max_bbox_length)
    pixel = pixel * scaling_factor_real_image
    pixel = np.round(pixel).astype(int)
    # undo mask
    pixel += np.round(np.array([bb[1],bb[0]]),0).astype(int)
    print('bbox',bb)
    print('img_size',img_size)
    return pixel



def get_orig_size_keypoints(target_folder,max_bbox_length,img_size,visualise,visualisation_list):

    for name in tqdm(os.listdir(target_folder + '/cropped_and_masked')):

        if not 'table' in name:
            continue

        kp_real = np.load(target_folder + '/keypoints/' + name.split('.')[0] + '.npz')
        pts_real= np.transpose(np.array(kp_real["pts"])[:2,:])
        pixels_real = pts_real[:,::-1]

        
        out_path = target_folder + '/kp_orig_img_size/' + name.split('.')[0] + '.json'
        vis_path = target_folder + '/kp_orig_img_size_vis/' + name.split('.')[0] + '.png'

        with open(target_folder + '/segmentation_infos/' + name.split('.')[0] + '.json','r') as f:
            seg_info = json.load(f)
        bbox = seg_info["predictions"]["bbox"]


        pixels_real_orig_size = pixel_cropped_to_original(pixels_real,bbox,max_bbox_length,img_size)
        print(pixels_real_orig_size)

        with open(out_path,'w') as f:
            json.dump({"pixels_real_orig_size": pixels_real_orig_size.tolist()},f)

        if visualise == "True":

            if seg_info["img"] in visualisation_list:
                vis_path = vis_path
                img = cv2.imread(target_folder + '/images/' + seg_info["img"])
                size_kp = float(min(seg_info["img_size"])/30)
                kp = [cv2.KeyPoint(float(pixels_real_orig_size[i][1]),float(pixels_real_orig_size[i][0]),size=size_kp) for i in range(pixels_real_orig_size.shape[0])]
                vis_img = cv2.drawKeypoints(img, kp, img, color=(200,0,0))
                cv2.imwrite(vis_path,vis_img)


def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)


    target_folder = global_config["general"]["target_folder"]
    visualise = global_config["general"]["visualise"]
    max_bbox_length = global_config["segmentation"]["max_bbox_length"]
    img_size = global_config["segmentation"]["img_size"]
    models_folder_read = global_config["general"]["models_folder_read"]

    with open(target_folder + '/global_stats/visualisation_images.json','r') as f:
        visualisation_list = json.load(f)
    
    get_orig_size_keypoints(target_folder,max_bbox_length,img_size,visualise,visualisation_list)

    



if __name__ == '__main__':
    print('Get keypoints 2D orig size')
    main()
