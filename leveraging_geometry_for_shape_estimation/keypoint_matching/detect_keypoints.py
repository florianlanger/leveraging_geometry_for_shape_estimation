import cv2
import numpy as np
import sys
import os
import json
from tqdm import tqdm
import matplotlib.cm as cm


sys.path.append('/data/cornucopia/fml35/other_peoples_work/keypoints/SuperPoint/SuperPointPretrainedNetwork')
from demo_superpoint import SuperPointFrontend


def make_empty_folder_structure(inputpath,outputpath):
    for dirpath, dirnames, filenames in os.walk(inputpath):
        structure = os.path.join(outputpath, dirpath[len(inputpath):])
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print("Folder {} already exists!".format(structure))



def detect_keypoints(path_img,out_path,super_point):

    img_centered_cropped = cv2.imread(path_img)
    img_padded = np.zeros((256,256,3),dtype=np.uint8)
    min_pad = 53
    max_pad = 203

    img_padded[min_pad:max_pad,min_pad:max_pad,:] = img_centered_cropped
    img = cv2.cvtColor(img_padded, cv2.COLOR_BGR2GRAY)

    # print('SYN IMAGES')

    # img = cv2.imread(path_img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)/255.
    pts, desc, heatmap = super_point.run(img)
    np.savez(out_path,pts=pts,desc=desc)
    return pts,desc


def get_keypoints_for_folder(folder_input,folder_output,folder_vis,super_point,visualise,target_folder,visualisation_list,run_on_octopus):

    make_empty_folder_structure(folder_input,folder_output)
    if visualise:
        make_empty_folder_structure(folder_input,folder_vis)

    for dirpath, dirnames, filenames in tqdm(os.walk(folder_input)):
        for name in filenames:
            # if '.png' in name:
            img_path = os.path.join(dirpath,name)

            out_path = os.path.join(dirpath, name.rsplit('.',1)[0] + '.npz').replace(folder_input,folder_output)
            # if os.path.exists(out_path):
            #     continue
            pts,desc = detect_keypoints(img_path,out_path,super_point)

            if visualise == "True":
                with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
                    gt_infos = json.load(f)

                if gt_infos["img"] in visualisation_list:
                    vis_path = os.path.join(dirpath,name).replace(folder_input,folder_vis).split('.')[0] + '.png'
                    img = cv2.imread(img_path.replace('/cropped_and_masked_small/','/cropped_and_masked/'))
                    if pts != []:
                        if run_on_octopus == "False":
                            # because of different opencv version in shape_env compared to shape_env_octo
                            kp = [cv2.KeyPoint(float(pts[0][i]),float(pts[1][i]),_size=1) for i in range(pts.shape[1])]
                        elif run_on_octopus == "True":
                            kp = [cv2.KeyPoint(float(pts[0][i]),float(pts[1][i]),size=1) for i in range(pts.shape[1])]
                        # kp = [cv2.KeyPoint(float(pts[0][i]),float(pts[1][i])) for i in range(pts.shape[1])]
                        confidence = [pts[2][i]  for i in range(pts.shape[1])]
                        # vis_img = cv2.drawKeypoints(img, kp, img, color=(255,0,0))

                        # print(cm.hot(confidence[0]))
                        confidence  = np.array(confidence)
                        confidence = (confidence - np.min(confidence)) / (np.max(confidence) - np.min(confidence))

                        for i in range(pts.shape[1]):
                            color = np.array(cm.hot(confidence[i])[:3])
                            color = np.round(color*255)
                            vis_img = cv2.drawKeypoints(img, [kp[i]], img, color=(0,0,255))# tuple(color))
                    else:
                        vis_img = img
                    cv2.imwrite(vis_path,vis_img)



def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    detect_info = global_config["keypoints"]["detection"]
    super_point_network = SuperPointFrontend(weights_path=detect_info["weights_path"],nms_dist=detect_info["nms_dist"],conf_thresh=detect_info["conf_thresh"],nn_thresh=detect_info["nn_thresh"],cuda=True)

    target_folder = global_config["general"]["target_folder"]
    visualise = global_config["general"]["visualise"]
    # visualise = True

    # get keypoints real
    print('Debug use cropped and masked small')
    folder_input_real = target_folder + '/cropped_and_masked_small/'
    folder_output_real = target_folder + '/keypoints/'
    folder_vis_real = target_folder + '/keypoints_vis/'
    run_on_octopus = global_config["general"]["run_on_octopus"]

    with open(target_folder + '/global_stats/visualisation_images.json','r') as f:
        visualisation_list = json.load(f)

    print('loaded')

    get_keypoints_for_folder(folder_input_real,folder_output_real,folder_vis_real,super_point_network,visualise,target_folder,visualisation_list,run_on_octopus)

    # get keypoints rendered
    # folder_input_rendered = global_config["general"]["models_folder_read"] + '/models/render_black_background/'
    # folder_output_rendered = global_config["general"]["models_folder_read"] + '/models/keypoints/'
    # folder_vis_rendered = global_config["general"]["models_folder_read"] + '/models/keypoints_vis/'
    # visualise = False
    # get_keypoints_for_folder(folder_input_rendered,folder_output_rendered,folder_vis_rendered,super_point_network,visualise,target_folder,visualisation_list,run_on_octopus)

    



if __name__ == '__main__':
    main()
