import os
import json
import shutil
import sys

def make_dirs(path):
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

    shutil.copytree('/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation',path + '/code')
    shutil.copy('/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/global_information.json',path + '/global_information.json')
    



def main():
    global_info = os.path.dirname(os.path.abspath(__file__)) + '/../global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    target_folder = global_config["general"]["target_folder"]

    assert target_folder == sys.argv[1]
    # target_folder = "/data/cornucopia/fml35/experiments/debug_segmentation_5"
    print('Create dirs')
    make_dirs(target_folder)

if __name__ == '__main__':
    main()