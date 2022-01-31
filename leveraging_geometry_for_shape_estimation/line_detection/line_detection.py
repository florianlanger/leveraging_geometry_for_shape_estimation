import torch
import os
import sys
import cv2
from tqdm import tqdm
import json
import numpy as np

sys.path.insert(0,'/scratch/fml35/other_peoples_work/line_detection/mlsd_pytorch')

from models.mbv2_mlsd_tiny import  MobileV2_MLSD_Tiny
from models.mbv2_mlsd_large import  MobileV2_MLSD_Large
from utils import  pred_lines, pred_squares


def predict(input_path,output_path,vis_path,model,visualise=True):
    orig_img = cv2.imread(input_path)
    h,w,_ = orig_img.shape
    img = cv2.resize(orig_img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # lines = pred_lines(img, model, [512, 512], 0.1, 20)

    img_line = orig_img.copy()

    lines, squares, score_array, inter_points = pred_squares(img,
                 model,
                 input_shape=[512, 512],
                #  input_shape=[w,h],
                 params={'score': 0.06,
                         'outside_ratio': 0.28,
                         'inside_ratio': 0.45,
                         'w_overlap': 0.0,
                         'w_degree': 1.95,
                         'w_length': 0.0,
                         'w_area': 1.86,
                         'w_center': 0.14})
    
      # draw lines

    resize_factors = np.array([w/512.,h/512.,w/512.,h/512.])

    if not isinstance(lines,list):
        orig_size_lines = np.round(lines * resize_factors).astype(int)
    if not isinstance(inter_points,list):
        inter_points= inter_points*resize_factors[:2]
    if not isinstance(squares,list):
        squares = np.round(squares * resize_factors[0:2]).astype(int)
 

    
    np.save(output_path,lines)
    if visualise:
        for line in orig_size_lines:
            x_start, y_start, x_end, y_end = [int(val) for val in line]
            cv2.line(img_line, (x_start, y_start), (x_end, y_end), [255,0,0], 2)
        

        # draw intersections
        for pt in inter_points:
            x, y = [int(val) for val in pt]
            cv2.circle(img_line, (x, y), 5, [0,0,255], -1)
        
        # draw squares
        for square in squares:
            cv2.polylines(img_line, [square.reshape([-1, 1, 2])], True, [200,200,0], 2)
        
        for square in squares[0:1]:
            cv2.polylines(img_line, [square.reshape([-1, 1, 2])], True, [255,255,0], 5)
            for pt in square:
                cv2.circle(img_line, (int(pt[0]), int(pt[1])), 5, [0,255,255], -1)

        cv2.imwrite(vis_path, img_line)
   

def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    with open(global_config["general"]["target_folder"] + '/global_stats/visualisation_images.json','r') as f:
        visualisation_list = json.load(f)

    device = torch.device('cuda:{}'.format(global_config["general"]["gpu"]))

    input_dir = global_config["general"]["target_folder"] + '/images'
    output_dir = global_config["general"]["target_folder"] + '/lines_2d/'
    output_dir_vis = global_config["general"]["target_folder"] + '/lines_2d_vis/'


    model_path = global_config["line_detection"]["model_path"]
    model = MobileV2_MLSD_Large().cuda().eval()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    # model = None

    for img_name in tqdm(os.listdir(input_dir)):
        visualise = img_name in visualisation_list
        input_path = input_dir +  '/' + img_name
        output_path = output_dir + img_name.split('.')[0] + '.npy'
        vis_path = output_dir_vis + img_name.split('.')[0] + '.png'
        predict(input_path,output_path,vis_path,model,visualise)




    

if __name__ == '__main__':
    print('Detecting Lines')
    main()

