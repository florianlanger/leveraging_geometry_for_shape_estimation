from platform import dist
import numpy as np
import cv2
import os
import sys
import json
from tqdm import tqdm

def visualise_lines(cropped_lines,img,vis_path):

    for line in cropped_lines:
        y_start, x_start, y_end, x_end = [int(val) for val in line]
        cv2.line(img, (x_start, y_start), (x_end, y_end), [255,0,0], 2)

    cv2.imwrite(vis_path, img)


def make_line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def check_point_inside(point,h,w):
    return (point < np.array([h-1+0.3,w-1+0.3])).all() and (point >= 0).all()

def crop_lines(lines,h,w):

    thresh = 1000000

    # move lines one in to avoid rounding errors that push intersect out of image

    # L1 = make_line([0,0], [0,w-1])
    # L2 = make_line([0,0], [h-1,0])
    # L3 = make_line([h-1,0], [h-1,w-1])
    # L4 = make_line([0,w-1], [h-1,w-1])
    L1 = make_line([1,1], [1,w-2])
    L2 = make_line([1,1], [h-2,1])
    L3 = make_line([h-2,1], [h-2,w-2])
    L4 = make_line([1,w-2], [h-2,w-2])
    test_lines = [L1,L2,L3,L4]

    cropped_lines = lines.copy()

    for i in range(lines.shape[0]):
        for j in range(2):
            Rs = []
            dists = []

            point = lines[i,2*j:2*(j+1)]

            if not check_point_inside(point,h,w):
                for test_line in test_lines:
                    line_formatted = make_line(lines[i,:2],lines[i,2:4])

                    R = intersection(test_line, line_formatted)
                    Rs.append(R)
                for k in range(4):
                    if Rs[k]:
                        if check_point_inside(np.array(Rs[k]),h,w):
                            dists.append(np.linalg.norm(np.array(Rs[k]) - point))
                        else:
                            dists.append(thresh)
                    else:
                        dists.append(thresh)
                    
                index = np.argmin(dists)

                    # if R[0] and R[2]:

                    # if R:
                    #     if check_point_inside(np.array(R),h,w):
                if dists[index] < thresh:
                    cropped_lines[i,2*j:2*(j+1)] = np.round(np.array(Rs[index])).astype(int)

            # print(Rs)
            # print(dists)


    h_w = np.tile(np.array([h,w,h,w]),(lines.shape[0],1))
    # print('cropped lines',cropped_lines)
    # print(dfd)
    assert (cropped_lines < h_w).all(), (cropped_lines,cropped_lines < h_w,lines,h,w)
    assert (cropped_lines >= 0).all()
    return cropped_lines


def resize_lines(lines,h,w):
    resize_factors = np.array([w/512.,h/512.,w/512.,h/512.])
    lines = np.round(lines * resize_factors).astype(int)
    lines = lines[:,[1,0,3,2]]
    return lines

def main():
    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    with open(global_config["general"]["target_folder"] + '/global_stats/visualisation_images.json','r') as f:
        visualisation_list = json.load(f)

    target_folder = global_config["general"]["target_folder"]
    
    for img_name in tqdm(os.listdir(target_folder + '/images')):
        visualise = img_name in visualisation_list
        input_path_img = target_folder +  '/images/' + img_name
        input_path_lines = target_folder + '/lines_2d/' + img_name.split('.')[0] + '.npy'
        output_path_lines = target_folder + '/lines_2d_cropped/' + img_name.split('.')[0] + '.npy'
        vis_path = target_folder + '/lines_2d_cropped_vis/' + img_name.split('.')[0] + '.png'

        lines = np.load(input_path_lines)
        img = cv2.imread(input_path_img)
        h,w,_ = img.shape

        lines = resize_lines(lines,h,w)

        cropped_lines = crop_lines(lines,h,w)
        np.save(output_path_lines,cropped_lines)

        if visualise:
            visualise_lines(cropped_lines,img,vis_path)




    

if __name__ == '__main__':
    print('Crop Lines')
    main()