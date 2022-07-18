import os
import shutil

def create_directories(exp_path):

    os.mkdir(exp_path)
    shutil.copy('/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/probabilistic_pose_and_shape/train_pose_classifier_from_lines/config.json',exp_path + '/config.json')
    print('Copied config')
    os.mkdir(exp_path + '/code')
    os.mkdir(exp_path + '/log_files')
    os.mkdir(exp_path + '/saved_models')
    os.mkdir(exp_path + '/vis')
    os.mkdir(exp_path + '/vis_roca_eval')
    os.mkdir(exp_path + '/vis_3d')
    os.mkdir(exp_path + '/vis_3d_roca_eval')
    os.mkdir(exp_path + '/predictions')
    print('no copy code')
    # print('copy code')
    # shutil.copytree('/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/probabilistic_pose_and_shape/train_pose_classifier_from_lines',exp_path + '/code/train_pose_classifier_from_lines')
    # shutil.copytree('/home/mifs/fml35/code/shape/retrieval_plus_keypoints/probabilistic_formulation',exp_path + '/code/probabilistic_formulation')
    