import cv2
import os
from tqdm import tqdm

dir_pred = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_203_new_lines_from_2d_all_vis/T_lines_vis_lines_gt_objects_vis_1'
dir_gt = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_203_new_lines_from_2d_all_vis/images'
dir_output = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_203_new_lines_from_2d_all_vis/T_lines_vis_lines_gt_objects_vis_1_overlayed'

# os.mkdir(dir_output)
for file in tqdm(sorted(os.listdir(dir_pred))):

    pred = cv2.imread(dir_pred + '/' + file)
    # print(pred.shape)
    # print(dir_gt + '/' + file.rsplit('_',3)[0] + '.jpg')
    gt = cv2.imread(dir_gt + '/' + file.rsplit('_',3)[0] + '.jpg')
    # print(gt.shape)
    overlayed = cv2.addWeighted(pred,0.5,gt,0.5,0)
    cv2.imwrite(dir_output + '/' + file,overlayed)

# /scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_203_new_lines_from_2d_all_vis/images/scene0011_00-000000.jpg