import cv2
from glob import glob
import os
from tqdm import tqdm


size = (480,360)
in_dir = '/scratches/gwangban_3/fml35/exp/04_scannet_val'
out_dir = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/'


with open('/scratch2/fml35/datasets/scannet/data_splits/scannetv2_train.txt','r') as f:
    train = f.readlines()
train_scenes = [x.strip() for x in train]

with open('/scratch2/fml35/datasets/scannet/data_splits/scannetv2_val.txt','r') as f:
    val = f.readlines()
val_scenes = [x.strip() for x in val]


for path in tqdm(sorted(glob(in_dir + '/*'))):
    scene = path.split('/')[-1].rsplit('_',3)[0]
    image = path.split('/')[-1].split('_')[2]

    if not 'norm' in path:
        continue
    
    train_or_val = None
    if scene in train_scenes:
        train_or_val = 'train'
    elif scene in val_scenes:
        train_or_val = 'val'

    kind = path.split('/')[-1].split('_')[4].split('.')[0]
    
    out_path = out_dir + train_or_val + '/' + kind + '_{}_{}/'.format(size[0],size[1]) + scene + '-' + image + '.png'
    image = cv2.imread(path)
    resized = cv2.resize(image,size)
    cv2.imwrite(out_path,resized)
    