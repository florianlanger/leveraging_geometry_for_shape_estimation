import os
import json
import shutil
import sys


def dict_replace_value(d, old, new):
    x = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = dict_replace_value(v, old, new)
        elif isinstance(v, str):
            print(v)
            v = v.replace(old, new)
            print(v)
        x[k] = v
    return x

def preprocess_config(in_path,out_path):

    with open(in_path,'r') as f:
        config = json.load(f)

    # config["general"]["models_folder_read"] = "/data/cornucopia/fml35/experiments/test_output_all_s2"
    config["general"]["models_folder_read"] = "/scratch/fml35/experiments/leveraging_geometry_for_shape/test_output_all_s2"
    config["general"]["image_folder"] = config["general"]["target_folder"] + '/images'
    config["general"]["mask_folder"] = config["general"]["target_folder"] + '/masks'

    if config["segmentation"]["use_gt"] == "False":
        config["segmentation"]["config"] = "configs/segmentation_swin_{}.py".format(config["dataset"]["split"])
        config["segmentation"]["checkpoint"] = "models/segmentation_swin_{}.pth".format(config["dataset"]["split"])

        config["retrieval"]["checkpoint_file"] = "models/embedding_predicted_mask_{}.pth".format(config["dataset"]["split"])

    elif config["segmentation"]["use_gt"] == "True":
        config["retrieval"]["checkpoint_file"] = "models/embedding_gt_mask_{}.pth".format(config["dataset"]["split"])

    if config["general"]["run_on_octopus"] == 'False':
        config = dict_replace_value(config,'/scratch/fml35/','/scratches/octopus/fml35/')
        config = dict_replace_value(config,'/scratch2/fml35/','/scratches/octopus_2/fml35/')
        assert False == True, "accesssing scrathc 2 not working" 


    with open(out_path,'w') as f:
        json.dump(config,f,indent=4)

def make_dirs(path,global_info_path):
    os.mkdir(path)
    os.mkdir(path + '/bbox_overlap')
    os.mkdir(path + '/embedding')
    os.mkdir(path + '/gt_infos')
    os.mkdir(path + '/nn_infos')
    os.mkdir(path + '/segmentation_infos')
    os.mkdir(path + '/segmentation_all_vis')
    os.mkdir(path + '/segmentation_vis')
    os.mkdir(path + '/cropped_and_masked')
    os.mkdir(path + '/cropped_and_masked_small')
    os.mkdir(path + '/global_stats')
    os.mkdir(path + '/global_stats/T_hists')
    os.mkdir(path + '/nn_vis')
    os.mkdir(path + '/segmentation_masks')
    os.mkdir(path + '/wc_gt')
    os.mkdir(path + '/keypoints')
    os.mkdir(path + '/keypoints_vis')
    os.mkdir(path + '/keypoints_filtered')
    os.mkdir(path + '/keypoints_filtered_vis')
    os.mkdir(path + '/kp_orig_img_size')
    os.mkdir(path + '/kp_orig_img_size_vis')
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
    os.mkdir(path + '/factors')
    os.mkdir(path + '/factors_lines_vis')
    os.mkdir(path + '/lines_2d')
    os.mkdir(path + '/lines_2d_vis')
    os.mkdir(path + '/lines_2d_cropped')
    os.mkdir(path + '/lines_2d_cropped_vis')
    os.mkdir(path + '/lines_2d_filtered')
    os.mkdir(path + '/lines_2d_filtered_vis')
    os.mkdir(path + '/combined_vis')
    os.mkdir(path + '/poses_R')
    os.mkdir(path + '/combined_vis_metrics_name')
    os.mkdir(path + '/T_lines_vis')
    os.mkdir(path + '/factors_T')
    os.mkdir(path + '/selected_nn')

    os.mkdir(path + '/models')
    os.mkdir(path + '/models/depth')
    os.mkdir(path + '/models/remeshed')
    os.mkdir(path + '/models/render_black_background')
    os.mkdir(path + '/models/render_no_background')
    os.mkdir(path + '/models/rotations')
    os.mkdir(path + '/models/keypoints')
    os.mkdir(path + '/models/keypoints_vis')

    preprocess_config(global_info_path,path + '/global_information.json')
    shutil.copytree('/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation',path + '/code')
    

def main():
    global_info_path = os.path.dirname(os.path.abspath(__file__)) + '/../global_information.json'
    with open(global_info_path,'r') as f:
        global_config = json.load(f)

    target_folder = global_config["general"]["target_folder"]
    if global_config["general"]["run_on_octopus"] == 'False':
        target_folder = target_folder.replace('/scratch/fml35/','/scratches/octopus/fml35/')

    assert target_folder == sys.argv[1],(target_folder,sys.argv[1])
    print(target_folder)
    print('Create dirs')
    make_dirs(target_folder,global_info_path)

if __name__ == '__main__':
    main()