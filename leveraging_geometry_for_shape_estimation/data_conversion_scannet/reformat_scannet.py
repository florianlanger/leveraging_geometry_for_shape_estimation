
import os
import shutil
import json
import sys
import random
import imagesize
from tqdm import tqdm

from reproject_scannet import load_4by4_from_txt


def get_focal_length(path_intriniscs_color,w):
    K = load_4by4_from_txt(path_intriniscs_color)
    focal_length = 2*K[0,0]/w

    sw = w / K[0,0]
    return focal_length,K

def main():

    print(sys.argv[1] + '/global_information.json')
    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["dataset"]["which_dataset"] == 'scannet':

        target_folder = global_config["general"]["target_folder"]

        if global_config["general"]["debug_set"] == "True":
            max_count = 2
        elif global_config["general"]["debug_set"] == "False":
            max_count = 200000

        # copy images
        dir_path = global_config["dataset"]["dir_path_images"]

        # dir_path_own_data = dir_path.replace('/scannet_frames_25k/','/scannet_frames_25k_own_data/')
        dir_path_own_data = global_config["dataset"]["dir_path_images_own_data"]
        exp_path = target_folder + '/'
       
        # with open(global_config["dataset"]["roca_results"],'r') as file:
        #     roca = json.load(file)

        print('only load frames for which roca makes predictions')
        if global_config["general"]["only_frames_with_gt_annotation"]:
            print('ONLY frames with gt annotation')

        img_counter = 0
        for scene in tqdm(os.listdir(dir_path_own_data)):
            for frame in os.listdir(dir_path + scene + '/color/'):

                    # skip images for which have no mask i.e. no gt object

                    if global_config["general"]["only_frames_with_gt_annotation"] == "True" and len(os.listdir(dir_path_own_data + scene + '/masks/' + frame.replace('.jpg',''))) == 0:
                        continue

                    # if scene + '/color/' + frame not in roca:
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

                        # if not object_data["category"] == 'bed':
                        #     continue

                        # with open(dir_path_own_data + scene + '/bbox/' + frame.replace('.jpg','') + '/' + object.split('.')[0] + '.json') as file:
                        #     bbox = json.load(file)

                        # object_data["bbox"] = bbox["bbox"]

                        object_data["mask_path"] = scene + '-' + frame.replace('.jpg','') + '_' + str(counter).zfill(2) + '.png'
                        objects.append(object_data)


                        old_path = dir_path_own_data + scene + '/masks/' + frame.replace('.jpg','') + '/' + object
                        new_path = exp_path + 'masks/' + scene + '-' + frame.replace('.jpg','') + '_' + str(counter).zfill(2) + '.png'
                        shutil.copyfile(old_path,new_path)



                        counter += 1

                    # bed_in_objects = False
                    # for object in objects:
                    #     if object["category"] == 'bed':
                    #         bed_in_objects = True
                    # if bed_in_objects == False:
                    #     continue


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
                    

                    with open(exp_path + 'gt_infos/' + scene + '-' + frame.split('.')[0] + '.json','w') as file:
                        json.dump(gt_infos,file,indent=4)

                    img_counter += 1

                    if img_counter == max_count:
                        break
            else:
                # Continue if the inner loop wasn't broken.
                continue
            # Inner loop was broken, break the outer.
            break

if __name__ == '__main__':
    random.seed(0)
    main()