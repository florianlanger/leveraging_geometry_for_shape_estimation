import os
import json
import shutil
import sys


def preprocess_config(in_path,out_path):

    with open(in_path,'r') as f:
        config = json.load(f)

    # config["general"]["models_folder_read"] = "/data/cornucopia/fml35/experiments/test_output_all_s2"
    config["general"]["models_folder_read"] = config["general"]["target_folder"]
    config["general"]["image_folder"] = config["general"]["target_folder"] + '/images'
    config["general"]["mask_folder"] = config["general"]["target_folder"] + '/masks'

    if config["segmentation"]["use_gt"] == "False":
        config["segmentation"]["config"] = "configs/segmentation_swin_{}.py".format(config["dataset"]["split"])
        config["segmentation"]["checkpoint"] = "models/segmentation_swin_{}.pth".format(config["dataset"]["split"])

        config["retrieval"]["checkpoint_file"] = "models/embedding_predicted_mask_{}.pth".format(config["dataset"]["split"])

    elif config["segmentation"]["use_gt"] == "True":
        config["retrieval"]["checkpoint_file"] = "models/embedding_gt_mask_{}.pth".format(config["dataset"]["split"])

    with open(out_path,'w') as f:
        json.dump(config,f,indent=4)

def make_dirs(path,global_info_path):
    os.mkdir(path)
    os.mkdir(path + '/bbox_overlap')
    os.mkdir(path + '/embedding')
    os.mkdir(path + '/gt_infos')
    os.mkdir(path + '/nn_infos')
    os.mkdir(path + '/segmentation_infos')
    os.mkdir(path + '/segmentation_vis')
    os.mkdir(path + '/cropped_and_masked')
    os.mkdir(path + '/cropped_and_masked_small')
    os.mkdir(path + '/global_stats')
    os.mkdir(path + '/nn_vis')
    os.mkdir(path + '/segmentation_masks')
    os.mkdir(path + '/wc_gt')
    os.mkdir(path + '/keypoints')
    os.mkdir(path + '/keypoints_vis')
    os.mkdir(path + '/matches')
    os.mkdir(path + '/matches_orig_img_size')
    os.mkdir(path + '/matches_vis')
    os.mkdir(path + '/wc_matches')
    os.mkdir(path + '/poses')
    os.mkdir(path + '/poses_vis')
    os.mkdir(path + '/metrics')
    os.mkdir(path + '/matches_quality')
    os.mkdir(path + '/matches_quality_vis')
    os.mkdir(path + '/images')
    os.mkdir(path + '/masks')


    os.mkdir(path + '/models')
    os.mkdir(path + '/models/depth')
    os.mkdir(path + '/models/remeshed')
    os.mkdir(path + '/models/render_black_background')
    os.mkdir(path + '/models/render_no_background')
    os.mkdir(path + '/models/rotations')
    os.mkdir(path + '/models/keypoints')
    os.mkdir(path + '/models/keypoints_vis')

    preprocess_config(global_info_path,path + '/global_information.json')
    



def main():
    global_info_path = os.path.dirname(os.path.abspath(__file__)) + '/../global_information.json'
    with open(global_info_path,'r') as f:
        global_config = json.load(f)

    target_folder = global_config["general"]["target_folder"]

    assert target_folder == sys.argv[1]

    print('Create dirs')
    make_dirs(target_folder,global_info_path)

if __name__ == '__main__':
    main()