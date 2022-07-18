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
            # print(v)
            v = v.replace(old, new)
            # print(v)
        x[k] = v
    return x

def preprocess_config(in_path,out_path):

    with open(in_path,'r') as f:
        config = json.load(f)

    assert config["pose_and_shape_probabilistic"]["pose"]["gt_R"] == False,"need to change dir from which read R in translation_from_lines_v5"

    # config["general"]["models_folder_read"] = "/data/cornucopia/fml35/experiments/test_output_all_s2"
    # config["general"]["models_folder_read"] = "/scratch/fml35/experiments/leveraging_geometry_for_shape/test_output_all_s2"
    config["general"]["image_folder"] = config["general"]["target_folder"] + '/images'
    config["general"]["mask_folder"] = config["general"]["target_folder"] + '/masks'

    # assert True == False,("models folder read should be in scannet condidtion")
    # SEGMENTATION and RETRIEVAL

    if config["segmentation"]["use_gt"] == "False" or config["segmentation"]["use_gt"] == "roca":
        config["segmentation"]["config"] = "configs/segmentation_swin_{}.py".format(config["dataset"]["split"])
        config["segmentation"]["checkpoint"] = "models/segmentation_swin_{}.pth".format(config["dataset"]["split"])
        config["retrieval"]["checkpoint_file"] = "models/embedding_predicted_mask_{}.pth".format(config["dataset"]["split"])

    elif config["segmentation"]["use_gt"] == "True":
        config["retrieval"]["checkpoint_file"] = "models/embedding_gt_mask_{}.pth".format(config["dataset"]["split"])


    # DATASET
    if config["dataset"]["which_dataset"] == "scannet":
        config["dataset"]["categories"] = ["bathtub","bed","bin","bookshelf","cabinet","chair","display","sofa","table"]
        config["dataset"]["dir_path"] = "/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/"
        config["dataset"]["dir_path_images"] = "/scratch2/fml35/datasets/scannet/scannet_frames_25k/"
        config["general"]["models_folder_read"] = "/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_117_scannet_models"
        # DOESNT WORK config["dataset"]["dir_path_images_own_data"] = "/scratch2/fml35/datasets/scannet/scannet_frames_25k_own_data_exp_08_min_pixel_per_mask_goban1_test_scenes/"
        config["dataset"]["dir_path_images_own_data"] = "/scratch2/fml35/datasets/scannet/scannet_frames_25k_own_data/"
        config["dataset"]["roca_results"] = "/scratch2/fml35/results/ROCA/per_frame_best_no_null.json"
        config["pose_and_shape"]["pose"]["sensor_width"] = 2
        # config["pose_and_shape_probabilistic"]["reproject_lines"]["line_dir_3d_precise"] = config["general"]["models_folder_read"] + '/models/lines'
        # config["pose_and_shape_probabilistic"]["reproject_lines"]["line_dir_3d_self_scanned"] = config["general"]["models_folder_read"] + '/models/lines'
        config["pose_and_shape_probabilistic"]["reproject_lines"]["line_dir_3d_precise"] = config["general"]["models_folder_read"] + '/models/extract_from_2d/exp_03/lines_3d_reformatted'
        config["pose_and_shape_probabilistic"]["reproject_lines"]["line_dir_3d_self_scanned"] = config["general"]["models_folder_read"] + '/models/extract_from_2d/exp_03/lines_3d_reformatted'
        config["general"]["path_to_existing_data"] = "/scratch2/fml35/datasets/own_data/data_leveraging_geometry_for_shape/data_01"

    elif config["dataset"]["which_dataset"] == "pix3d":
        config["dataset"]["categories"] = ["bed","bookcase","chair","desk","misc","sofa","table","tool","wardrobe"]
        config["dataset"]["dir_path"] = "/scratch/fml35/datasets/pix3d_new/"
        config["pose_and_shape"]["pose"]["sensor_width"] = 32
        config["pose_and_shape_probabilistic"]["reproject_lines"]["line_dir_3d_precise"] = "/scratch/fml35/datasets/pix3d_new/own_data/rendered_models/3d_lines/exp_26_points_on_edges_angle_20_lines_one_and_three_face/lines"
        config["pose_and_shape_probabilistic"]["reproject_lines"]["line_dir_3d_self_scanned"] = "/scratch/fml35/datasets/pix3d_new/own_data/rendered_models/3d_lines/exp_15_min_support_10_continuous/lines"

    elif config["dataset"]["which_dataset"] == "future3d":
        config["dataset"]["categories"] = ["bed","cabinetshelfdesk","chair","lighting","pierstool","sofa","table"]
        config["dataset"]["dir_path"] = "/scratch/fml35/datasets/3d_future/3D-FUTURE-model_reformatted/"
        config["general"]["models_folder_read"] = "/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/3d_future/data_01"
        config["pose_and_shape"]["pose"]["sensor_width"] = 1200
        config["pose_and_shape_probabilistic"]["reproject_lines"]["line_dir_3d_precise"] = "/scratch/fml35/datasets/pix3d_new/own_data/rendered_models/3d_lines/exp_26_points_on_edges_angle_20_lines_one_and_three_face/lines"
        config["pose_and_shape_probabilistic"]["reproject_lines"]["line_dir_3d_self_scanned"] = "/scratch/fml35/datasets/pix3d_new/own_data/rendered_models/3d_lines/exp_15_min_support_10_continuous/lines"
        config["general"]["path_to_existing_data"] = "/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/3d_future/data_01"
    # CHOOSE BEST
    if config["pose_and_shape_probabilistic"]["use_probabilistic"] == "False":
        config["pose_and_shape"]["pose"]["choose_best_based_on"] = "combined"

    elif config["pose_and_shape_probabilistic"]["use_probabilistic"] == "True":
        config["pose_and_shape"]["pose"]["choose_best_based_on"] = "factor"

    if config["general"]["run_on_octopus"] == 'False':
        config = dict_replace_value(config,'/scratch/fml35/','/scratches/octopus/fml35/')
        config = dict_replace_value(config,'/scratch2/fml35/','/scratches/octopus_2/fml35/')

    with open(out_path,'w') as f:
        json.dump(config,f,indent=4)

def make_dirs(path,global_info_path):
    os.mkdir(path)
    # os.mkdir(path + '/bbox_overlap')
    # os.mkdir(path + '/embedding')
    # os.mkdir(path + '/gt_infos')
    # os.mkdir(path + '/nn_infos')
    # os.mkdir(path + '/segmentation_infos')
    # os.mkdir(path + '/segmentation_all_vis')
    # os.mkdir(path + '/segmentation_vis')
    # os.mkdir(path + '/cropped_and_masked')
    # os.mkdir(path + '/cropped_and_masked_small')
    os.mkdir(path + '/global_stats')
    os.mkdir(path + '/global_stats/T_hists')
    os.mkdir(path + '/global_stats/eval_scannet')
    os.mkdir(path + '/global_stats/eval_scannet/results_per_scene')
    os.mkdir(path + '/global_stats/eval_scannet/results_per_scene_filtered')
    os.mkdir(path + '/global_stats/eval_scannet/results_per_scene_visualised')
    os.mkdir(path + '/global_stats/eval_scannet/results_per_scene_filtered_visualised')
    os.mkdir(path + '/global_stats/eval_scannet/results_per_frame_visualised')
    os.mkdir(path + '/global_stats/eval_scannet/results_per_frame_filtered_visualised')

    os.mkdir(path + '/global_stats/eval_scannet/results_per_scene_flags_with_retrieval')
    os.mkdir(path + '/global_stats/eval_scannet/results_per_scene_flags_without_retrieval')
    os.mkdir(path + '/global_stats/eval_scannet/results_per_scene_scan2cad_constraints')
    os.mkdir(path + '/global_stats/eval_scannet/results_per_scene_scan2cad_constraints_visualised')
    
    # os.mkdir(path + '/nn_vis')
    # os.mkdir(path + '/segmentation_masks')
    # os.mkdir(path + '/wc_gt')
    # os.mkdir(path + '/keypoints')
    # os.mkdir(path + '/keypoints_vis')
    # os.mkdir(path + '/keypoints_filtered')
    # os.mkdir(path + '/keypoints_filtered_vis')
    # os.mkdir(path + '/kp_orig_img_size')
    # os.mkdir(path + '/kp_orig_img_size_vis')
    # os.mkdir(path + '/matches')
    # os.mkdir(path + '/matches_orig_img_size')
    # os.mkdir(path + '/matches_vis')
    # os.mkdir(path + '/wc_matches')
    os.mkdir(path + '/poses')
    os.mkdir(path + '/poses_stages')
    os.mkdir(path + '/poses_vis')
    os.mkdir(path + '/metrics')
    os.mkdir(path + '/metrics_scannet')
    # os.mkdir(path + '/matches_quality')
    # os.mkdir(path + '/matches_quality_vis')
    # os.mkdir(path + '/images')
    # os.mkdir(path + '/masks')
    # os.mkdir(path + '/factors')
    # os.mkdir(path + '/factors_lines_vis')
    # os.mkdir(path + '/lines_2d')
    # os.mkdir(path + '/lines_2d_vis')
    # os.mkdir(path + '/lines_2d_cropped')
    # os.mkdir(path + '/lines_2d_cropped_vis')
    # os.mkdir(path + '/lines_2d_filtered')
    # os.mkdir(path + '/lines_2d_filtered_vis')
    os.mkdir(path + '/combined_vis')
    # os.mkdir(path + '/poses_R')
    # os.mkdir(path + '/poses_R_selected')
    os.mkdir(path + '/combined_vis_metrics_name')
    os.mkdir(path + '/T_lines_vis')
    os.mkdir(path + '/T_lines_vis_annotations')
    os.mkdir(path + '/T_lines_factors')
    os.mkdir(path + '/factors_T')
    os.mkdir(path + '/selected_nn')

    # os.mkdir(path + '/models')
    # os.mkdir(path + '/models/depth')
    # os.mkdir(path + '/models/remeshed')
    # os.mkdir(path + '/models/render_black_background')
    # os.mkdir(path + '/models/render_no_background')
    # os.mkdir(path + '/models/rotations')
    # os.mkdir(path + '/models/keypoints')
    # os.mkdir(path + '/models/keypoints_vis')

    os.mkdir(path + '/code')

    preprocess_config(global_info_path,path + '/global_information.json')
    assert os.path.exists(global_info_path)
    shutil.copytree('/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation',path + '/code/leveraging_geometry_for_shape_estimation')
    shutil.copytree('/home/mifs/fml35/code/shape/retrieval_plus_keypoints/probabilistic_formulation',path + '/code/probabilistic_formulation')
    # print('no copy code')
def main():
    global_info_path = os.path.dirname(os.path.abspath(__file__)) + '/../global_information.json'
    with open(global_info_path,'r') as f:
        global_config = json.load(f)

    assert os.path.exists(global_info_path)

    target_folder = global_config["general"]["target_folder"]
    if global_config["general"]["run_on_octopus"] == 'False':
        target_folder = target_folder.replace('/scratch/fml35/','/scratches/octopus/fml35/').replace('/scratch2/fml35/','/scratches/octopus_2/fml35/')

    assert target_folder == sys.argv[1],(target_folder,sys.argv[1])
    print(target_folder)
    print('Create dirs')
    make_dirs(target_folder,global_info_path)

if __name__ == '__main__':
    main()