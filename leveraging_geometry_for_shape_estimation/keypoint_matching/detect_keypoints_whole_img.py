from hashlib import new
import cv2
import numpy as np
import sys
import os
import json
from tqdm import tqdm
import matplotlib.cm as cm


sys.path.append('/data/cornucopia/fml35/other_peoples_work/keypoints/SuperPoint/SuperPointPretrainedNetwork')
from demo_superpoint import SuperPointFrontend


def prepare_image(image_path,target_size):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orig_size = img.shape
    
    img_aspect_ratio = img.shape[0] / float(img.shape[1])
    target_aspect_ratio = target_size[1] / target_size[0]

    if img_aspect_ratio > target_aspect_ratio:
        new_h = target_size[1]
        new_w = int(np.round(new_h / img_aspect_ratio))
        min_pad_x = 0
        max_pad_x = new_h
        min_pad_y = int(np.round((target_size[0] - new_w)/2))
        max_pad_y = min_pad_y + new_w
        scale = float(target_size[1] / img.shape[0])

    else:
        new_w = target_size[0]
        new_h = int(np.round(new_w * img_aspect_ratio))
        min_pad_y = 0
        max_pad_y = new_w
        min_pad_x = int(np.round((target_size[1] - new_h)/2))
        max_pad_x = min_pad_x + new_h
        scale = float(target_size[0] / img.shape[1])

    img_target = np.zeros((target_size[1],target_size[0]),dtype=np.uint8)
    img_target[min_pad_x:max_pad_x,min_pad_y:max_pad_y] = cv2.resize(img, (new_w,new_h))
    return img_target,min_pad_x,max_pad_x,min_pad_y,max_pad_y,scale,orig_size



def detect_keypoints(img,super_point):
    img = img.astype(np.float32)/255.
    pts, desc, heatmap = super_point.run(img)
    return pts,desc

# if points not betwenn minx max x get rid, then rescale 
def filter_points(pts,desc,min_pad_x,max_pad_x,min_pad_y,max_pad_y):
    mins = np.transpose(np.tile(np.array([min_pad_y,min_pad_x]),(pts.shape[1],1)))
    maxs = np.transpose(np.tile(np.array([max_pad_y,max_pad_x]),(pts.shape[1],1)))
    mask = np.all(pts[:2,:] > mins,axis=0) & np.all(pts[:2,:] < (maxs - 1),axis=0)
    
    return pts[:,mask],desc[:,mask]

def upscale_points(pts,min_pad_x,min_pad_y,scale):
    mins = np.transpose(np.tile(np.array([min_pad_y,min_pad_x]),(pts.shape[1],1)))
    pts[:2,:] = (pts[:2,:] - mins) / scale
    return pts


def get_keypoints_for_folder(super_point,visualise,target_folder,visualisation_list,target_size):

    for name in tqdm(os.listdir(target_folder + '/' + 'images')):

        img_path = target_folder + '/images/' + name
        out_path = target_folder + '/keypoints/' + name.split('.')[0] + '.npy'

        image,min_pad_x,max_pad_x,min_pad_y,max_pad_y,scale,orig_size = prepare_image(img_path,target_size)
        pts,desc = detect_keypoints(image.copy(),super_point)

        pts,desc = filter_points(pts,desc,min_pad_x,max_pad_x,min_pad_y,max_pad_y)
        pts = upscale_points(pts,min_pad_x,min_pad_y,scale)
        np.save(out_path,np.transpose(pts))

        if visualise == "True":

            with open(target_folder + '/gt_infos/' + name.split('.')[0] + '.json','r') as f:
                gt_infos = json.load(f)

            if gt_infos["img"] in visualisation_list:
                vis_path = target_folder + '/keypoints_vis/' + name.split('.')[0] + '.png'
                img = cv2.imread(img_path)
                # img = image.copy()
                kp = [cv2.KeyPoint(float(pts[0][i]),float(pts[1][i]),size=int(max(orig_size)/100)) for i in range(pts.shape[1])]
                confidence = [pts[2][i]  for i in range(pts.shape[1])]
                confidence  = np.array(confidence)
                confidence = (confidence - np.min(confidence)) / (np.max(confidence) - np.min(confidence))

                for i in range(pts.shape[1]):
                    color = np.array(cm.hot(confidence[i])[:3])
                    color = np.round(color*255)
                    img = cv2.drawKeypoints(img, [kp[i]], img, color=255)#tuple(color))
                cv2.imwrite(vis_path,img)



def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    detect_info = global_config["keypoints"]["detection"]
    super_point_network = SuperPointFrontend(weights_path=detect_info["weights_path"],nms_dist=detect_info["nms_dist"],conf_thresh=detect_info["conf_thresh"],nn_thresh=detect_info["nn_thresh"],cuda=True)

    target_folder = global_config["general"]["target_folder"]
    visualise = global_config["general"]["visualise"]
    target_size = global_config["keypoints"]["detection"]["size"]


    with open(target_folder + '/global_stats/visualisation_images.json','r') as f:
        visualisation_list = json.load(f)


    get_keypoints_for_folder(super_point_network,visualise,target_folder,visualisation_list,target_size)

    # get keypoints rendered
    # folder_input_rendered = global_config["general"]["models_folder_read"] + '/models/render_black_background/'
    # folder_output_rendered = target_folder + '/models/keypoints/'
    # folder_vis_rendered = target_folder + '/models/keypoints_vis/'
    # get_keypoints_for_folder(folder_input_rendered,folder_output_rendered,folder_vis_rendered,super_point_network,visualise)

    



if __name__ == '__main__':
    main()
