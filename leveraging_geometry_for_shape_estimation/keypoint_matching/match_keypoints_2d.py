import cv2
import numpy as np
import sys
import os
import json
from tqdm import tqdm


def find_matches(des1,des2,crossCheck,k_matches_no_crossCheck):


    des1 = np.swapaxes(des1,0,1)
    des2 = np.swapaxes(des2,0,1)

    bf = cv2.BFMatcher(crossCheck=crossCheck)
    matches_with_empty_lists = bf.knnMatch(des1,des2,k=k_matches_no_crossCheck)

    nested_matches = []
    for match in matches_with_empty_lists:
        if len(match) != 0:
            nested_matches.append(match)

    flattened_matches = []
    for match in nested_matches:
        for indiv_match in match:
            flattened_matches.append([indiv_match])

    return flattened_matches,nested_matches


def visualise_matches(img1,kp1,img2,kp2,matches):
    img_match = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None) #,matchColor=(0,255,0),flags=2)
    return img_match



def extract_pixels_from_matches(matches,kp1,kp2):
    # for keypoints first coordinate is col, second is row. (0,0) is top left corner
    pixels_real = [] 
    pixels_rendered = []

    for i in range(len(matches)):
        for k in range(len(matches[i])):
            query_idx = matches[i][k].queryIdx
            train_idx = matches[i][k].trainIdx

            pixels_real.append(kp1[:2,query_idx])
            pixels_rendered.append(kp2[:2,train_idx])

    pixels_real = np.array(pixels_real)
    pixels_rendered = np.array(pixels_rendered)
    # swap order so that first pixel is row and second is col, as had in test.py script
    pixels_real = np.asarray(pixels_real[:,[1,0]],int)
    pixels_rendered = np.asarray(pixels_rendered[:,[1,0]],int)

    return pixels_real,pixels_rendered






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
    return pixel



def get_matches_for_folder(target_folder,top_n_retrieval,crossCheck,k_matches_no_crossCheck,max_bbox_length,img_size,models_folder_read,visualise,visualisation_list):

    for name in tqdm(os.listdir(target_folder + '/cropped_and_masked')):

        kp_real = np.load(target_folder + '/keypoints/' + name.split('.')[0] + '.npz')
        pts_real,desc_real = kp_real["pts"],kp_real["desc"]
        
        with open(target_folder + '/nn_infos/' + name.split('.')[0] + '.json','r') as f:
            retrieval_list = json.load(f)["nearest_neighbours"]
        
        for i in range(top_n_retrieval):
            out_path = target_folder + '/matches_orig_img_size/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json'
            if os.path.exists(out_path):
                continue

            rendered_kp_path = models_folder_read + '/models/keypoints/' + retrieval_list[i]["path"].replace('.png','.npz').replace('.0.','.').replace('.5.','.')
            kp_rendered = np.load(rendered_kp_path)
            pts_rendered,desc_rendered = kp_rendered["pts"],kp_rendered["desc"]


            flattened_matches,nested_matches = find_matches(desc_real,desc_rendered,crossCheck,k_matches_no_crossCheck)

            pixels_real,pixels_rendered = extract_pixels_from_matches(flattened_matches,pts_real,pts_rendered)

            with open(target_folder + '/matches/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json','w') as f:
                json.dump({"pixels_real": pixels_real.tolist(),"pixels_rendered": pixels_rendered.tolist()},f)

            with open(target_folder + '/segmentation_infos/' + name.split('.')[0] + '.json','r') as f:
                seg_info = json.load(f)
            bbox = seg_info["predictions"]["bbox"]

            pixels_real_orig_size = pixel_cropped_to_original(pixels_real,bbox,max_bbox_length,img_size)

            with open(out_path,'w') as f:
                json.dump({"pixels_real_orig_size": pixels_real_orig_size.tolist(),"pixels_rendered": pixels_rendered.tolist()},f)


            if visualise == "True":

                if seg_info["img"] in visualisation_list:
                    path_real = target_folder + '/cropped_and_masked/' + name
                    img_real = cv2.imread(path_real)

                    path_rendered = models_folder_read + '/models/render_black_background/' + retrieval_list[i]["path"]
                    img_rendered = cv2.imread(path_rendered)


                    real_cv = [cv2.KeyPoint(float(pts_real[0][i]),float(pts_real[1][i]),_size=1) for i in range(pts_real.shape[1])]
                    rendered_cv = [cv2.KeyPoint(float(pts_rendered[0][i]),float(pts_rendered[1][i]),_size=1) for i in range(pts_rendered.shape[1])]

                    img_match = visualise_matches(img_real,real_cv,img_rendered,rendered_cv,nested_matches)
                    cv2.imwrite(target_folder + '/matches_vis/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.png',img_match)

        


def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)


    target_folder = global_config["general"]["target_folder"]
    visualise = global_config["general"]["visualise"]
    top_n_retrieval = global_config["keypoints"]["matching"]["top_n_retrieval"]
    crossCheck = global_config["keypoints"]["matching"]["crossCheck"]
    k_matches_no_crossCheck = global_config["keypoints"]["matching"]["k_matches_no_crossCheck"]
    max_bbox_length = global_config["segmentation"]["max_bbox_length"]
    img_size = global_config["segmentation"]["img_size"]
    models_folder_read = global_config["general"]["models_folder_read"]

    with open(target_folder + '/global_stats/visualisation_images.json','r') as f:
        visualisation_list = json.load(f)
    
    get_matches_for_folder(target_folder,top_n_retrieval,crossCheck,k_matches_no_crossCheck,max_bbox_length,img_size,models_folder_read,visualise,visualisation_list)

    



if __name__ == '__main__':
    print('Match keypoints 2D')
    main()
