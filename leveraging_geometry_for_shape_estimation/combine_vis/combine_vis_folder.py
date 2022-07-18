import os
from torch import split_with_sizes
from tqdm import tqdm
import cv2
from combine_vis import resize_img
import sys
import argparse
import numpy as np

def load_image(path,size=None):
    if os.path.isfile(path):
        img = cv2.imread(path)
        if size != None:
            img = resize_img(img,size=size)
    else:
        img = np.zeros((size[0],size[1],3),dtype=np.uint8)
    return img

def load_image_potentially_different_orientation(path,size,file_exists=False):
    img_exist = False
    for orientation in range(4):
        add_on = str(orientation).zfill(2)
        check_path = path.rsplit('_',1)[0] + '_' + add_on + '.png'
        img = load_image(check_path,size)
        if os.path.exists(check_path):
            img_exist = True
            break
    if file_exists == True:
        assert img_exist,path

    return img

def load_image_potentially_different_orientation_category_ending(path,size,file_exists=False):
    img_exist = False
    for orientation in range(4):
        add_on = str(orientation).zfill(2)
        check_path = path.rsplit('_',2)[0] + '_' + add_on + '_' + path.split('_')[-1] #+'.png'
        img = load_image(check_path,size)
        if os.path.exists(check_path):
            img_exist = True
            break
    if file_exists == True:
        assert img_exist,path
    return img

def load_image_original(path,size):
    path_gt = path.rsplit('_',3)[0] +'.jpg'
    img = load_image(path_gt,size)
    return img

def combine_images(images):
    if len(images) == 2:
        combined = cv2.hconcat([images])
    elif len(images) == 3:
        top = cv2.hconcat(images[:2])
        bottom = cv2.hconcat([images[2],images[2]*0])
        combined = cv2.vconcat([top,bottom])
    elif len(images) == 4:
        top = cv2.hconcat(images[:2])
        bottom = cv2.hconcat(images[2:4])
        combined = cv2.vconcat([top,bottom])
    elif len(images) == 5:
        top = cv2.hconcat(images[:3])
        bottom = cv2.hconcat(images[3:5] + [images[4]*0])
        combined = cv2.vconcat([top,bottom])
    elif len(images) == 6:
        top = cv2.hconcat(images[:3])
        bottom = cv2.hconcat(images[3:6])
        combined = cv2.vconcat([top,bottom])
    return combined

def get_img_size(dirs):
    img = cv2.imread(sorted(os.listdir(dirs[0]))[0])
    img_size = img.shape[:2]
    return img_size


def main(dirs,outdir,img_size,dir_original_number):

    check_all_exist = True
    # if img_size == []:
    #     img_size = get_img_size(dirs)

    for name in tqdm(sorted(os.listdir(dirs[0]))):
        # print('name',name)

        images = []
        for i,dir in enumerate(dirs):
            # print('dir\n',dir)
            if i == dir_original_number:
                img = load_image_original(dir + '/' + name,img_size)
            else:
                # print(dir + '/' + name)
                if i in [0]:
                    img = load_image_potentially_different_orientation_category_ending(dir + '/' + name,img_size,file_exists=check_all_exist)
                elif i == 1:
                    # print(dir + '/' + name.rsplit('_',1)[0])
                    img = load_image_potentially_different_orientation(dir + '/' + name.rsplit('_',1)[0] + '.png',img_size,file_exists=check_all_exist)
                elif i == 2:
                    # print(dir + '/' + name.rsplit('_',1)[0])
                    img = load_image(dir + '/' + name.rsplit('_',3)[0] + '.png',size=[480,640])
                    # print(img.shape)
            images.append(img)
        combined = combine_images(images)
        cv2.imwrite(outdir + '/' + name,combined)

if __name__ == '__main__':

    CLI=argparse.ArgumentParser()
    CLI.add_argument("--dirs", nargs="*",type=str,help='list of dirs of images to concatentate. Iterate over first dir.')
    CLI.add_argument("--output",type=str)
    CLI.add_argument("--img_size",nargs="*",type=int,help='height width',default=None)

    args = CLI.parse_args()
    # dirs = args.dirs
    # outdir = args.output
    # img_size = args.img_size

    # dirs = ['/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_203_new_lines_from_2d_all_vis/quality_2D_lines/T_lines_vis_quality_2d_lines_6_absolute_threshold',
    #         '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_203_new_lines_from_2d_all_vis/quality_2D_lines/T_lines_vis_quality_2d_lines_7_absolute_threshold_smaller',
    #         '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_203_new_lines_from_2d_all_vis/quality_2D_lines/T_lines_vis_quality_2d_lines_3_overlap_04_n1',
    #         '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_203_new_lines_from_2d_all_vis/quality_2D_lines/T_lines_vis_quality_2d_lines_9_absolute_threshold_filtered',
    #         '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_203_new_lines_from_2d_all_vis/quality_2D_lines/T_lines_vis_quality_2d_lines_8_absolute_threshold_smaller_filtered',
    #         '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_203_new_lines_from_2d_all_vis/images']

    # outdir = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_203_new_lines_from_2d_all_vis/quality_2D_lines/combined_2_lines_6_absolute_threshold_lines_7_absolute_threshold_smaller_lines_3_overlap_04_n1_2d_lines_9_absolute_threshold_filtered_lines_8_absolute_threshold_smaller_filtered_gt'
    dirs = ['/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_186_roca_retrieval_gt_z_lines_octopus/poses_vis',
    #     '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_156_roca_all_vis/poses_vis',
    # '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_205_filtering_2d_lines_based_on_n_pixel/poses_vis',
    '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_206_filtered_lines/poses_vis',
    '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_206_filtered_lines/lines_2d_filtered_vis']
    
    outdir = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_186_roca_retrieval_gt_z_lines_octopus/combined_poses_1_exp_206_filtered_lines_poses_vis_lines_2d_filtered_vis'
    # img_size = [600,800]
    img_size = [480,640]
    # img_size = [968,1296]
    # img_size = None
    
    dir_original_number = 7

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    
    assert len(dirs) >= 2 and len(dirs) <= 6
    main(dirs,outdir,img_size,dir_original_number)