import cv2
from glob import glob
import os
from tqdm import tqdm


# size = (128,96)
size = (640,480)
# size = (480,360)

out_dir = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/'


with open('/scratch2/fml35/datasets/scannet/data_splits/scannetv2_train.txt','r') as f:
    train = f.readlines()
train_scenes = [x.strip() for x in train]

with open('/scratch2/fml35/datasets/scannet/data_splits/scannetv2_val.txt','r') as f:
    val = f.readlines()
val_scenes = [x.strip() for x in val]


dir_path = '/scratch2/fml35/datasets/scannet/scannet_frames_25k/'


for scene in tqdm(os.listdir(dir_path)):
    for image in os.listdir(dir_path + scene + '/depth'):

    
        train_or_val = None
        if scene in train_scenes:
            train_or_val = 'train'
        elif scene in val_scenes:
            train_or_val = 'val'

        
        out_path = out_dir + train_or_val + '/depth_{}_{}/{}-{}'.format(size[0],size[1],scene,image)
        image = cv2.imread(dir_path + scene + '/depth/' + image,cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image,size)
        # print(out_path)
        cv2.imwrite(out_path,image)