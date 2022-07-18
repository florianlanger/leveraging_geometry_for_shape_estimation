import cv2
from tqdm import tqdm
from glob import glob
import os

size = (480,360)
in_dir = '/scratches/gwangban_3/fml35/exp/06_scannet_normal/exp01_every_10/NLL_ours/test/val/'

out_dir = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val/norm_gb_medium_480_360'


dirs_1 = set(os.listdir('/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val/norm_gb_medium_480_360'))
dirs_2 = set(os.listdir('/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/norm_480_360'))

print(dirs_1 - dirs_2)
print(dirs_2 - dirs_1)


# for path in tqdm(sorted(glob(in_dir + '/*/*'))):
#     scene = path.split('/')[-2]
#     image = path.split('/')[-1]

#     name_combined = scene + '-' + image.split('.')[0].zfill(6) + '.png'

#     out_path = out_dir + '/' +  name_combined
#     image = cv2.imread(path)
#     resized = cv2.resize(image,size)
#     cv2.imwrite(out_path,resized)