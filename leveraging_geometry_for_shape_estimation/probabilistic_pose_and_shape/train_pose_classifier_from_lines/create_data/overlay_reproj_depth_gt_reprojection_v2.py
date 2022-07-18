import os
from tqdm import tqdm
import cv2

# dir_1 = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/train/valid_objects_render_overlay_correct_offset'
dir_1 = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/train/valid_objects_render_overlay'
dir_2 = '/scratch2/fml35/experiments/ROCA/data/Instance_mask_visualised'

dir_out = dir_1 + 'compare_roca'

os.mkdir(dir_out)


for file in tqdm(sorted(os.listdir(dir_1))):

    scene = file.split('-')[0].split('/')[-1]
    img = file.split('-')[-1].split('.')[0]

 
    img_1 = cv2.imread(os.path.join(dir_1, file))


    img_2_path = os.path.join(dir_2, file.rsplit('_',1)[0] + '.png')
    assert os.path.exists(img_2_path), '{} does not exist'.format(img_2_path)
    img_2 = cv2.imread(img_2_path)
    img_2 = cv2.resize(img_2, (img_1.shape[1], img_1.shape[0]))

    # overlay
    out = img_1 / 2 + img_2 / 2

    out_path = os.path.join(dir_out, file)
    cv2.imwrite(out_path, out)
