
import os
from shutil import copyfile

scannet_dir = '/scratch2/fml35/datasets/scannet/scannet_frames_25k/'

target_dir = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/images/'

with open('/scratch2/fml35/datasets/scannet/data_splits/scannetv2_val.txt','r') as f:
    lines = f.readlines()
scenes = [line.split('\n')[0] for line in lines]
for scene in scenes:
    for img in os.listdir(scannet_dir + scene + '/color/'):
        new_name = scene + '-' + img
        copyfile(scannet_dir + scene + '/color/' + img, target_dir + new_name)