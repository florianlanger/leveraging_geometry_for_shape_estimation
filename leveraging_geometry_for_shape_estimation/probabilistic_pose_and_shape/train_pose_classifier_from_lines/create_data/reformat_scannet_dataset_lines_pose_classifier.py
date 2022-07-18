
import os
import shutil
import json
import sys
import imagesize
from tqdm import tqdm
import numpy as np

def load_4by4_from_txt(path):
    M = np.zeros((4,4))
    with open(path,'r') as f:
        content = f.readlines()
        for i in range(4):
            line = content[i].split()
            for j in range(4):
                M[i,j] = np.float32(line[j])
        return M

def get_focal_length(path_intriniscs_color,w):
    K = load_4by4_from_txt(path_intriniscs_color)
    focal_length = 2*K[0,0]/w

    sw = w / K[0,0]
    return focal_length,K


def load_scene_list(path):
    with open(path,'r') as f:
        scenes = f.readlines()
    scenes = [scene.replace('\n','') for scene in scenes]
    return scenes

def main():


    # copy images
    dir_path = '/scratch2/fml35/datasets/scannet/scannet_frames_25k/'

    kind = 'val'

    dir_path_own_data = "/scratch2/fml35/datasets/scannet/scannet_frames_25k_own_data_exp_12_calibration_matrix/"
    exp_path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/'
    exp_path += kind + '/'

    # os.mkdir(exp_path + 'gt_infos')
    # os.mkdir(exp_path + 'images')
    # os.mkdir(exp_path + 'masks')

    # shutil.copytree('/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/probabilistic_pose_and_shape/train_pose_classifier_from_lines/create_data',exp_path + '/code')
    
    scenes = load_scene_list('/scratch2/fml35/datasets/scannet/data_splits/scannetv2_{}.txt'.format(kind))

    img_counter = 0
    for scene in tqdm(sorted(os.listdir(dir_path_own_data))):

        if scene not in scenes:
            continue

        for frame in os.listdir(dir_path + scene + '/color/'):

                # skip images for which have no mask i.e. no gt object
                if len(os.listdir(dir_path_own_data + scene + '/masks/' + frame.replace('.jpg',''))) == 0:
                    continue


                out_path = exp_path + 'gt_infos/' + scene + '-' + frame.split('.')[0] + '.json'

                if 'scene0088_02-000300' not in out_path:
                    continue

                new_path_check = exp_path + 'masks/' + scene + '-' + frame.replace('.jpg','') + '_' + str(len(os.listdir(dir_path_own_data + scene + '/masks/' + frame.replace('.jpg','')))-1).zfill(2) + '.png'

                # if os.path.exists(out_path):
                # if os.path.exists(new_path_check):
                #     continue

                # copy mask
                objects = []
                counter = 0
                for object in os.listdir(dir_path_own_data + scene + '/masks/' + frame.replace('.jpg','')):

                    path_object_data = dir_path_own_data + scene + '/info/' + frame.replace('.jpg','') + '/' + object.split('.')[0] + '.json'
                    
                    # RECENT ADDITION 01/03/22 10:32
                    if not os.path.exists(path_object_data):
                        continue
                
                    with open(path_object_data) as file:
                        object_data = json.load(file)


                    with open(dir_path_own_data + scene + '/bbox/' + frame.replace('.jpg','') + '/' + object.split('.')[0] + '.json') as file:
                        bbox = json.load(file)

                    object_data["bbox"] = bbox["bbox"]
                    object_data["mask_path"] = scene + '-' + frame.replace('.jpg','') + '_' + str(counter).zfill(2) + '.png'
                    objects.append(object_data)


                    old_path = dir_path_own_data + scene + '/masks/' + frame.replace('.jpg','') + '/' + object
                    new_path = exp_path + 'masks/' + scene + '-' + frame.replace('.jpg','') + '_' + str(counter).zfill(2) + '.png'
                    shutil.copyfile(old_path,new_path)



                    counter += 1

                
                # copy img
                old_path = dir_path + scene + '/color/' + frame
                new_path = exp_path + 'images/' + scene + '-' + frame
                shutil.copyfile(old_path,new_path)

                gt_infos = {}
                gt_infos["name"] = scene + '-' + frame.replace('.jpg','')
                gt_infos["img"] = scene + '-' + frame
                gt_infos["img_size"] = imagesize.get(old_path)
                # assert gt_infos["img_size"] == (1296, 968), gt_infos["img_size"]
                f, K = get_focal_length(dir_path + scene + '/intrinsics_color.txt',gt_infos["img_size"][0])
                gt_infos["focal_length"] = f
                gt_infos["K"] = K.tolist()
                gt_infos["objects"] = objects
                

                with open(out_path,'w') as file:
                    json.dump(gt_infos,file,indent=4)

                img_counter += 1

        else:
            # Continue if the inner loop wasn't broken.
            continue
        # Inner loop was broken, break the outer.
        break

if __name__ == '__main__':
    main()