
import os
import cv2
from tqdm import tqdm

dir_0 = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/images_480_360'
dir_1 = '/scratch/fml35/experiments/regress_T/vis_own/date_2022_06_16_time_17_46_25_five_refinements_depth_aug_points_in_bbox_3d_coords_epoch_000150_translation_pred_scale_pred_rotation_roca_init_retrieval_roca_all_images_False'
dir_2 = '/scratch/fml35/experiments/regress_T/vis_roca/results_pi_over_16_1_8'
dir_3 = '/scratch/fml35/experiments/regress_T/vis_gt/results_pi_over_16_1_8'

out_dir = '/scratch/fml35/experiments/regress_T/vis_own/combined_just_overlays_date_2022_06_16_time_17_46_25_five_refinements_depth_aug_points_in_bbox_3d_coords_epoch_000150_translation_pred_scale_pred_rotation_roca_init_retrieval_roca_all_images_False'

# os.mkdir(out_dir)

for file in tqdm(sorted(os.listdir(dir_1))):

    if '_geometry' in file:
        continue

    assert os.path.exists(os.path.join(dir_0, file.replace('.png', '.jpg'))), '{} does not exist'.format(os.path.join(dir_0, file.replace('.png', '.jpg')))
    assert os.path.exists(os.path.join(dir_1, file)), '{} does not exist'.format(os.path.join(dir_1, file))
    assert os.path.exists(os.path.join(dir_2, file)), '{} does not exist'.format(os.path.join(dir_2, file))
    assert os.path.exists(os.path.join(dir_3, file)), '{} does not exist'.format(os.path.join(dir_3, file))
    assert os.path.exists(os.path.join(dir_1, file.replace('.png', '_geometry.png'))), '{} does not exist'.format(os.path.join(dir_1, file.replace('.png', '_geometry.png')))
    assert os.path.exists(os.path.join(dir_2, file.replace('.png', '_geometry.png'))), '{} does not exist'.format(os.path.join(dir_2, file.replace('.png', '_geometry.png')))
    assert os.path.exists(os.path.join(dir_3, file.replace('.png', '_geometry.png'))), '{} does not exist'.format(os.path.join(dir_3, file.replace('.png', '_geometry.png')))

    img_0 = cv2.imread(dir_0 + '/' + file.replace('.png', '.jpg'))
    img_1 = cv2.imread(dir_1 + '/' + file)
    # img_2 = cv2.imread(dir_1 + '/' + file.replace('.png', '_geometry.png'))
    img_3 = cv2.imread(dir_2 + '/' + file)
    # img_4 = cv2.imread(dir_2 + '/' + file.replace('.png', '_geometry.png'))
    img_5 = cv2.imread(dir_3 + '/' + file)
    # img_6 = cv2.imread(dir_3 + '/' + file.replace('.png', '_geometry.png'))

    # combined = cv2.hconcat([img_3, img_4, img_1, img_2, img_5, img_6])
    combined = cv2.hconcat([img_0,img_3,img_1,img_5])
    cv2.imwrite(out_dir + '/' + file, combined)