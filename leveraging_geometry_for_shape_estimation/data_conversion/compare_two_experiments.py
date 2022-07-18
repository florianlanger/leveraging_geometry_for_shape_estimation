import sys
import json
import os
import shutil
from jsondiff import diff
import glob
import cv2
import numpy as np
from tqdm import tqdm


def compare_jsons(path_1,path_2):

    if not os.path.exists(path_2):
        print('Doesnt exist',path_2)

    else:
        with open(path_1,'r') as f:
            cf1 = json.load(f)
        with open(path_2,'r') as f:
            cf2 = json.load(f)

        difference = diff(cf1,cf2,syntax='symmetric')
        if difference != {}:
            print(path_1.rsplit('/',1)[1],difference)

def compare_images(path1,path2):
    img_1 = cv2.imread(path1)
    img_2 = cv2.imread(path2)

    if not os.path.exists(path2):
        print('Doesnt exist',path2)

    elif ((img_1 - img_2) != 0).any():
        print(path1,path2)

def compare_compressed(path1,path2):
    if not os.path.exists(path2):
        print('Doesnt exist',path2)

    else:
        a = np.load(path1)
        b = np.load(path2)

        for key in a:
            if a[key].shape != b[key].shape:
                print('shape mismatch',path1)
            elif (np.abs(a[key] - b[key]) > 0.00001).any():
                print(path1,path2)

def compare_array(path1,path2):
    if not os.path.exists(path2):
        print('Doesnt exist',path2)
    else:
        a = np.load(path1)
        b = np.load(path2)
        if a.shape != b.shape:
            # print(a)
            # print(b)
            # print(fsfs)
            print('shape mismatch',a.shape,b.shape,path1)

        elif (np.abs(a - b) > 0.00001).any():
            print(path1,path2)

def compare_numpys(path1,path2):
    if '.npz' in path1:
        compare_compressed(path1,path2)
    if '.npy' in path1:
        compare_array(path1,path2)



def compare_folder(folder_1,folder_2,max_n=10000000):
    files_1 = sorted(os.listdir(folder_1))
    files_2 = sorted(os.listdir(folder_2))

    print()
    missing = list(sorted(set(files_1) - set(files_2)))
    print(missing)
    print(len(missing))
    print(len(files_1))
    print(len(files_2))
    missing = list(sorted(set(files_2) - set(files_1)))
    print(missing)

    assert len(files_2) == len(set(files_2))
    assert len(files_1) == len(set(files_1))
    # assert files_1 == files_2

    if files_1 != files_2:
        print('Missing files')

    for file in tqdm(sorted(glob.glob(folder_1 + '/*'))[:max_n]):
        if '.png' in file or '.jpg' in file:
            compare_images(file,file.replace(folder_1,folder_2))
        elif '.json' in file:
            compare_jsons(file,file.replace(folder_1,folder_2))
        elif '.np' in file:
            compare_numpys(file,file.replace(folder_1,folder_2))



def main(exp_1,exp_2):

    max_n = 100000
    print('only check ',max_n)
    print('-----------------')

    compare_jsons(exp_1 + '/global_information.json',exp_2 + '/global_information.json')
    json_dirs = ['gt_infos','segmentation_infos','bbox_overlap','nn_infos','poses_R','poses','matches','matches_orig_img_size','selected_nn','metrics','metrics_scannet']
    img_dirs = ['images','masks','segmentation_all_vis','segmentation_vis','cropped_and_masked','cropped_and_masked_small','segmentation_masks','lines_2d_vis','lines_2d_cropped_vis','lines_2d_filtered_vis','keypoints_vis','matches_vis','T_lines_vis','poses_vis']
    numpy_dirs = ['lines_2d','lines_2d_cropped','lines_2d_filtered','wc_matches','keypoints']
    
    json_dirs = ['gt_infos','segmentation_infos','bbox_overlap','nn_infos','poses_R','poses','matches','matches_orig_img_size','selected_nn','metrics_scannet']
    # img_dirs = ['images','masks','segmentation_all_vis','segmentation_vis','cropped_and_masked','cropped_and_masked_small','segmentation_masks','lines_2d_vis','lines_2d_cropped_vis','lines_2d_filtered_vis','keypoints_vis','matches_vis','T_lines_vis','poses_vis']
    # numpy_dirs = ['lines_2d','lines_2d_cropped','lines_2d_filtered','wc_matches','keypoints']

    folders = json_dirs + img_dirs + numpy_dirs

    # folders_no_vis = ['gt_infos','images','masks','segmentation_infos','segmentation_masks','cropped_and_masked','cropped_and_masked_small','bbox_overlap','lines_2d','lines_2d_cropped','lines_2d_filtered','nn_infos','keypoints','matches','wc_matches','matches_orig_img_size','poses_R','poses','selected_nn','metrics','metrics_scannet']
    folders_no_vis = ['images']
    # folders_no_vis = ['lines_2d']
    for folder in folders_no_vis:
    # for folder in ['metrics_scannet']:
        print('Check ',folder)
        compare_folder(exp_1 + '/' + folder,exp_2 + '/' + folder,max_n=max_n)
        





if __name__ == '__main__':
    exp_1 = sys.argv[1]
    exp_2 = sys.argv[2]
    main(exp_1,exp_2)