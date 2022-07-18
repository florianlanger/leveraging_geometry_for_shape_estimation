from platform import dist
import re
from cv2 import threshold
import numpy as np
import cv2
import os
import sys
import json
from tqdm import tqdm
import torch

from leveraging_geometry_for_shape_estimation.utilities.write_on_images import draw_text_block

from probabilistic_formulation.factors.factors_T.compare_lines import sample_points_from_lines
from probabilistic_formulation.utilities import create_all_possible_combinations_2, create_all_possible_combinations_2_dimension_1



def T_to_pixel(T,f,sw,img_size_reshaped,w):

    cc1 = T
    nc1 = cc1 / (torch.abs(cc1[:,2:3]) / f)

    pix1_3d = pixel_bearing_to_pixel(nc1,w,sw,img_size_reshaped,f)

    return pix1_3d


def pixel_bearing_to_pixel(pb,w,sw,img_size,f):
    # img size needs to have same first dimension as pb
    assert img_size.shape[0] == pb.shape[0]
    assert len(img_size.shape) == 2
    assert img_size.shape[1] == 2
    assert len(pb.shape) == 2
    assert pb.shape[1] == 3
    # assert (torch.abs(pb[:,2] - f) < 0.001).all(),(pb[:,2],f)
    pixel = - pb[:,:2] * w/sw + img_size / 2

    return pixel

def check_reprojected_T_in_image(gt_infos,model_infos):
    T = torch.Tensor(model_infos['trans_mat']).unsqueeze(0)
    f = gt_infos['focal_length']
    sw = 2
    img_size_reshaped = torch.Tensor(gt_infos['img_size']).unsqueeze(0)
    w = gt_infos['img_size'][0]
    pixel = T_to_pixel(T,f,sw,img_size_reshaped,w)

    in_image = False
    if (pixel >= 0).all() and (pixel < img_size_reshaped - 1).all():
        in_image = True

    pixel = np.round(pixel.squeeze().numpy()).astype(int).tolist()

    return in_image,pixel

def main():
    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    size = 3
    color = [255,0,0]

    target_folder = global_config["general"]["target_folder"]
    visualise = True

    for name in tqdm(sorted(os.listdir(target_folder + '/gt_infos'))):

        with open(target_folder + '/gt_infos/' + name,'r') as file:
            gt_infos = json.load(file)

        for i in range(len(gt_infos['objects'])):

            in_image,pixels = check_reprojected_T_in_image(gt_infos,gt_infos['objects'][i])

            detection_name = name.split('.')[0] + '_' + str(i).zfill(2)
            infos = {'in_image': in_image, 'reprojected_T': pixels}
            with open(target_folder + '/T_in_image/' + detection_name + '.json','w') as f:
                json.dump(infos,f)



            if visualise == True:
                input_path = target_folder + '/masks_vis/' + detection_name + '.png'
                output_path = target_folder + '/T_in_image_vis/' + detection_name + '.png'


                img = cv2.imread(input_path)

                img_size = img.shape[:2][::-1]
                pixels = np.array(pixels) * img_size / np.array(gt_infos['img_size'])
                pixels = np.round(pixels).astype(int)

                if in_image == True:
                    try:
                        img[pixels[1]-size:pixels[1]+size,pixels[0]-size:pixels[0]+size,:] = np.tile(np.array([[color]]),(2*size,2*size,1))
                    except ValueError:
                            pass

                draw_text_block(img,['in image: ' + str(in_image),'pixel: ' + str(pixels)],top_left_corner=(20,20),font_scale=1,font_thickness=1)
                cv2.imwrite(output_path,img)
            
        



    

if __name__ == '__main__':
    print('Check object in center')
    main()