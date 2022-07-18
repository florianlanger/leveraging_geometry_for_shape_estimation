import os
from tqdm import tqdm
import cv2

# dir_overlay = '/scratches/octopus_2/fml35/datasets/scannet/scannet_frames_25k_own_data_exp_11_calibration_matrix/'
# dir_depth = '/scratch2/fml35/datasets/own_data/classifier_T_from_lines/data_02/train/reprojected_gt_depth'
# dir_out = '/scratch2/fml35/datasets/own_data/classifier_T_from_lines/data_02/train/depth_previous_reprojected_overlay'
# dir_out = '/scratch2/fml35/datasets/own_data/classifier_T_from_lines/data_02/train/depth_new_reprojected_overlay'

# dir_1 = '/scratch2/fml35/datasets/own_data/classifier_T_from_lines/data_02/train/reprojected_gt_depth'
dir_1 = '/scratch2/fml35/experiments/ROCA/data/Instance_mask_visualised'
dir_2 = '/scratches/octopus_2/fml35/datasets/scannet/scannet_frames_25k_own_data_exp_12_calibration_matrix/'

dir_out = '/scratch2/fml35/experiments/ROCA/data/Instance_mask_overlay_new_render_01'


for file in tqdm(sorted(os.listdir(dir_1))):
    if file != 'scene0017_00-001000.png':
        continue
    scene = file.split('-')[0].split('/')[-1]
    img = file.split('-')[-1].split('.')[0]

 
    img_1 = cv2.imread(os.path.join(dir_1, file))

    # img_2_path = os.path.join(dir_2, scene,'overlayed_instance_rerendered' ,img + '.png')
    img_2_path = os.path.join(dir_2, scene,'overlayed_instance_rerendered' ,img.split('_')[0] + '.png')
    # overlay_path = os.path.join(dir_instances, file)
    img_2 = cv2.imread(img_2_path)

    # overlay
    out = img_1 / 2 + img_2 / 2
    print('out', out.shape)
    print(os.path.join(dir_out, file))
    assert os.path.exists(dir_out)
    out_path = os.path.join(dir_out, file)
    cv2.imwrite(out_path, out)
    assert os.path.exists(out_path), '{} does not exist'.format(out_path)