import os
from tqdm import tqdm
import cv2

# dir_overlay = '/scratches/octopus_2/fml35/datasets/scannet/scannet_frames_25k_own_data_exp_11_calibration_matrix/'
# dir_depth = '/scratch2/fml35/datasets/own_data/classifier_T_from_lines/data_02/train/reprojected_gt_depth'
# dir_out = '/scratch2/fml35/datasets/own_data/classifier_T_from_lines/data_02/train/depth_previous_reprojected_overlay'
# dir_out = '/scratch2/fml35/datasets/own_data/classifier_T_from_lines/data_02/train/depth_new_reprojected_overlay'

# dir_1 = '/scratch2/fml35/datasets/own_data/classifier_T_from_lines/data_02/train/reprojected_gt_depth'
dir_1 = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val/roca_render/'
dir_2 = '/scratch2/fml35/experiments/ROCA/experiments/exp_07/overlay/'

dir_out = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val/roca_render_overlay/'


for file in tqdm(sorted(os.listdir(dir_1))):

 
    img_1 = cv2.imread(os.path.join(dir_1, file))
    img_2_path = os.path.join(dir_2, file.rsplit('_',1)[0].replace('-','_') + '.jpg')
    assert os.path.exists(img_2_path), '{} does not exist'.format(img_2_path)
    img_2 = cv2.imread(img_2_path)

    # overlay
    out = img_1 / 2 + img_2 / 2
    out_path = os.path.join(dir_out, file)
    cv2.imwrite(out_path, out)