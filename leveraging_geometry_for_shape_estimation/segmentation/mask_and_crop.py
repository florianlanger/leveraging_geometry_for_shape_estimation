import json
from re import T
import numpy as np
from PIL import Image, ImageOps
import os
from tqdm import tqdm
import sys
import cv2
import imageio



def main():
    print('mask and crop')
    print('Continue if bbox 0 in width or height')
    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    target_folder = global_config["general"]["target_folder"]
    image_folder = global_config["general"]["image_folder"]

    target_size = global_config["segmentation"]["max_bbox_length"]
    full_output_size = global_config["segmentation"]["img_size"]

    # debug
    all_overall_names = {}

    for name in tqdm(sorted(os.listdir(target_folder + '/bbox_overlap'))):


            current_overall_name = name.split('_')[0] + '_' + name.split('_')[1]
            if current_overall_name not in all_overall_names:
                all_overall_names[current_overall_name] = 0

            id_dict = {}
            id_dict["masks"] = []

            with open(target_folder + '/bbox_overlap/' + name,'r') as file:
                bbox_infos = json.load(file)

            with open(target_folder + '/segmentation_infos/' + name,'r') as file:
                segmentation_infos = json.load(file)

            if bbox_infos['valid'] == True and bbox_infos['correct_category'] == True:


                mask_path = target_folder + '/segmentation_masks/' + name.replace('.json','.png')

                # debug
                old_ending = segmentation_infos['img'].split('.')[1]

                out_path = target_folder + '/cropped_and_masked_small/' + name.replace('.json','.' + old_ending)

                # if os.path.exists(out_path):
                #     continue

                # mask_path = target_folder + '/segmentation_masks/' + name.replace('.json','.' + old_ending)
                # mask_path = '/data/cornucopia/fml35/experiments/test_output_all_s2/segmentation_masks/' + name.replace('.json','.png')
                # mask_path = '/data/cornucopia/fml35/experiments/test_output_all_s2/segmentation_masks/' + name.split('_')[0] + '_' + name.split('_')[1] + '_' + str(name.split('_')[2].split('.')[0]).zfill(2) + '.png'

                real_image_path = image_folder + '/' + segmentation_infos['img']

                real_image = Image.open(real_image_path)
                im2 = Image.new("RGB", real_image.size)

                mask = cv2.imread(mask_path)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = Image.fromarray(mask)

                real_image = Image.composite(real_image, im2, mask)

                bbox = segmentation_infos["predictions"]["bbox"]
                cropped_image = real_image.crop(bbox)
                (w, h) = cropped_image.size

                if not (w > 0 and h > 0):
                    continue 

                if h >= w: 
                    new_h = target_size
                    new_w = int(np.round(new_h/float(h) * w))
                elif w > h:
                    new_w = target_size
                    new_h = int(np.round(new_w/float(w) * h))
                
                # if not new_w > 0 or not new_h > 0:
                #     continue 

                resized_im = cropped_image.resize((new_w,new_h))

                old_size = resized_im.size
                # new_size = (full_output_size,full_output_size)
                new_size = (target_size,target_size)
                padded_im = Image.new("RGB", new_size)
                padded_im.paste(resized_im , (int((new_size[0]-old_size[0])/2),int((new_size[1]-old_size[1])/2)))

                padded_im.save(out_path)





                # image = imageio.imread(save_path)
                # padded_image_io = np.zeros((256,256,3),dtype=np.uint8)
                # padded_image_io[53:203,53:203,:] = image


                img_centered_cropped = cv2.imread(out_path)
                img_padded_cv2 = np.zeros((256,256,3),dtype=np.uint8)
                min_pad = int((256 - 150)/2)
                max_pad = int((256 + 150)/2)
                img_padded_cv2[min_pad:max_pad,min_pad:max_pad,:] = img_centered_cropped


                # assert (padded_image_io == img_padded_cv2).all()

                cv2.imwrite(target_folder + '/cropped_and_masked/' + name.replace('.json','.' + old_ending),img_padded_cv2)


                # debug
                # save_path = target_folder + '/cropped_and_masked/' + name.replace('.json','.png')
                # save_path = target_folder + '/cropped_and_masked/' + name.replace('.json','.' + old_ending)
                # padded_im.save(save_path)



if __name__ == '__main__':
    main()