from glob import glob
import cv2
import json
import os
import scipy.ndimage
from tqdm import tqdm

def slice_tuple(slice_):
    return slice_.start, slice_.stop, slice_.step

dataset_dir = '/scratch2/fml35/datasets/scannet/scannet_frames_25k_own_data_exp_12_calibration_matrix'

# for path in tqdm(glob('/scratch2/fml35/datasets/scannet/scannet_frames_25k_own_data/*/masks/*/*')):
for path in tqdm(glob(dataset_dir + '/*/masks/*/*')):
    # print(path)
    # print(os.path.exists(path))
    mask = cv2.imread(path)
    # print(mask.shape)
    mask = mask[:,:,0] / 255

    out = scipy.ndimage.find_objects(mask, max_label=1)
    out = out[0]
    y_start,y_stop,_ = slice_tuple(out[0])
    x_start,x_stop,_ = slice_tuple(out[1])

    # contours,_ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # rect = cv2.minAreaRect(contours[0])
    # (x_start,y_start),(x_stop,y_stop), a = rect

    bbox = [x_start,y_start,x_stop,y_stop]

    out_path = path.replace('/masks/','/bbox/').replace('.png','.json')
    with open(out_path,'w') as file:
        json.dump({'bbox':bbox},file)
        