import cv2
import os
import numpy as np
import json
from tqdm import tqdm

from leveraging_geometry_for_shape_estimation.utilities.write_on_images import draw_text_block

dir_path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/train/'

name_folder_gt = 'depth_640_480'
name_reprojected_depth = 'reprojected_gt_depth'
name_folder_output = 'accept_model_reprojected_depth'

if not os.path.exists(dir_path + name_folder_output):
    os.mkdir(dir_path + name_folder_output)
    os.mkdir(dir_path + name_folder_output + '_vis')

threshold_depth = 0.3
ratio_accept = 0.5

for img in tqdm(sorted(os.listdir(dir_path + name_reprojected_depth))):

    out_path_vis = dir_path + name_folder_output + '_vis/' + img
    if os.path.exists(out_path_vis):
        continue

    depth = cv2.imread(dir_path + name_reprojected_depth + '/' + img,cv2.IMREAD_UNCHANGED)
    depth = depth.astype(float)

    depth[depth > 60000] = depth[depth > 60000] * 0 - 1
    depth = depth / 1000

    depth_gt = cv2.imread(dir_path + name_folder_gt + '/' + img.rsplit('_',1)[0] + '.png',cv2.IMREAD_UNCHANGED)
    depth_gt = depth_gt.astype(float) / 1000

    depth_diff = np.abs(depth - depth_gt)
    depth_diff[depth < 0] = 10

    n_pixel_accept = int(np.sum(depth_diff < threshold_depth))

    n_total_pixel = int(np.sum(depth>0))

    if n_total_pixel == 0:
        ratio = 0.
    else:
        ratio = n_pixel_accept / n_total_pixel
    accept = ratio > ratio_accept

    infos = {'n_pixel_accept':n_pixel_accept,'n_total_pixel':n_total_pixel,'ratio':ratio,'accept':accept}
    with open(dir_path + name_folder_output + '/' + img.split('.')[0] + '.json','w') as f:
        json.dump(infos,f)


    vis_false = ((depth_diff > threshold_depth) & (depth > 0))
    vis_false = cv2.resize((vis_false * 255).astype(np.uint8),(480,360))
    vis_false = cv2.cvtColor(vis_false,cv2.COLOR_GRAY2BGR) * np.array([0,0,255])

    vis_true = cv2.resize(((depth_diff < threshold_depth) * 255).astype(np.uint8),(480,360))
    vis_true = cv2.cvtColor(vis_true,cv2.COLOR_GRAY2BGR) * np.array([0,255,0])



    # assert os.path.exists(dir_path + name_folder_gt + '/' + img.rsplit('_',1)[0] + '.jpg'),dir_path + name_folder_gt + '/' + img.rsplit('_',1)[0] + '.jpg'
    rgb = cv2.imread(dir_path +  'images_480_360/' + img.rsplit('_',1)[0] + '.jpg')

    out = ((rgb + vis_true + vis_false ) / 3).astype(np.uint8)
    out2 = cv2.hconcat([rgb,out])
    draw_text_block(out2,[str(np.round(ratio,3)) + ' ' + str(accept)],font_scale=1,font_thickness=1)

    cv2.imwrite(dir_path + name_folder_output + '_vis/' + img,out2)