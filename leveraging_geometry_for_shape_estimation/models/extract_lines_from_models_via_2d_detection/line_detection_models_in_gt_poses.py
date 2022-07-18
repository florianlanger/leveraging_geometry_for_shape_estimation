import torch
import os
import sys
import cv2
from tqdm import tqdm
import json
import numpy as np
from glob import glob

from leveraging_geometry_for_shape_estimation.utilities.folders import make_empty_folder_structure

sys.path.insert(0,'/scratches/octopus/fml35/other_peoples_work/line_detection/mlsd_pytorch')

from models.mbv2_mlsd_tiny import  MobileV2_MLSD_Tiny
from models.mbv2_mlsd_large import  MobileV2_MLSD_Large
from utils import deccode_output_score_and_ptss

def pred_lines(image, model,
               input_shape=[512, 512],
               score_thr=0.10,
               dist_thr=20.0):
    h, w, _ = image.shape
    h_ratio, w_ratio = [h / input_shape[0], w / input_shape[1]]

    resized_image = np.concatenate([cv2.resize(image, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_AREA),
                                    np.ones([input_shape[0], input_shape[1], 1])], axis=-1)

    resized_image = resized_image.transpose((2,0,1))
    batch_image = np.expand_dims(resized_image, axis=0).astype('float32')
    batch_image = (batch_image / 127.5) - 1.0

    batch_image = torch.from_numpy(batch_image).float().cuda()
    outputs = model(batch_image)
    pts, pts_score, vmap = deccode_output_score_and_ptss(outputs, 200, 3)
    start = vmap[:, :, :2]
    end = vmap[:, :, 2:]
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    segments_list = []
    for center, score in zip(pts, pts_score):
        y, x = center
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])

    lines = 2 * np.array(segments_list)  # 256 > 512

    # check if no prediction
    if len((lines.shape)) == 1:
        return []

    else:
        lines[:, 0] = lines[:, 0] * w_ratio
        lines[:, 1] = lines[:, 1] * h_ratio
        lines[:, 2] = lines[:, 2] * w_ratio
        lines[:, 3] = lines[:, 3] * h_ratio

        return lines



def predict(input_path,output_path,vis_path,model,visualise=True):
    orig_img = cv2.imread(input_path)
    h,w,_ = orig_img.shape

    new_w_h = 512

    img = cv2.resize(orig_img,(new_w_h, new_w_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_line = orig_img.copy()

    lines = pred_lines(img, model, [new_w_h, new_w_h], score_thr=0.1, dist_thr=20)

    # draw lines

    resize_factors = np.array([w/float(new_w_h),h/float(new_w_h),w/float(new_w_h),h/float(new_w_h)])

    if not isinstance(lines,list):
        lines = np.round(lines * resize_factors).astype(int)
    # if not isinstance(inter_points,list):
    #     inter_points= inter_points*resize_factors[:2]
 

    
    np.save(output_path,lines)
    if visualise:

        for line in lines:
            x_start, y_start, x_end, y_end = [int(val) for val in line]
            cv2.line(img_line, (x_start, y_start), (x_end, y_end), [255,0,0], 2)
        

        # # draw intersections
        # for pt in inter_points:
        #     x, y = [int(val) for val in pt]
        #     cv2.circle(img_line, (x, y), 5, [0,0,255], -1)

        cv2.imwrite(vis_path, img_line)



def main():


    global_info = sys.argv[1] + '/global_information.json'

    with open(global_info,'r') as f:
        global_config = json.load(f)

    device = torch.device('cuda:{}'.format(global_config["general"]["gpu"]))

    input_dir = global_config["general"]["target_folder"].replace('/scratch2/','/scratches/octopus_2/') + '/T_lines_vis_gt_render_1/'
    output_dir = global_config["general"]["target_folder"].replace('/scratch2/','/scratches/octopus_2/') + '/T_lines_vis_lines_gt_objects_1/'
    output_dir_vis = global_config["general"]["target_folder"].replace('/scratch2/','/scratches/octopus_2/') + '/T_lines_vis_lines_gt_objects_vis_1/'


    model_path = global_config["line_detection"]["model_path"].replace('/scratch/','/scratches/octopus/')
    print(model_path)
    model = MobileV2_MLSD_Large().cuda().eval()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

    for input_path in tqdm(sorted(glob(input_dir + '*'))):
        # print(input_path)
        output_path = input_path.replace(input_dir,output_dir).replace('.png','.npy')
        vis_path = input_path.replace(input_dir,output_dir_vis)
        predict(input_path,output_path,vis_path,model,visualise=True)



if __name__ == '__main__':
    print('Detecting Lines')
    main()