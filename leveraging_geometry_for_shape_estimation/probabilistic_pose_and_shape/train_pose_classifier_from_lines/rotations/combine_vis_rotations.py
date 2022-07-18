
import os
import cv2
from tqdm import tqdm

dir_1 = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/rotation_vis/date_2022_06_08_time_10_41_33_no_threshold_second_stage'
dir_2 = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val/lines_2d_filtered_vis/'

for file in tqdm(sorted(os.listdir(dir_1))):
    if 'full_min_angle' in file:
        file_path_1 = os.path.join(dir_1, file)
        file_path_2 = file_path_1.split('full_min_angle')[0] + 'tilt_and_elev.png'
        file_path_3 = dir_2 + file.split('_full_min_angle')[0] + '.png'

        assert os.path.exists(file_path_1), file_path_1
        assert os.path.exists(file_path_2), file_path_2
        assert os.path.exists(file_path_3), file_path_3
        
        img_1 = cv2.imread(file_path_1)
        img_2 = cv2.imread(file_path_2)
        img_3 = cv2.imread(file_path_3)
        img_3 = cv2.resize(img_3,(img_1.shape[1],img_1.shape[0]))
        combined = cv2.hconcat([img_1,img_2,img_3])
        cv2.imwrite(file_path_1.replace('.png','_combined.png'),combined)