
import os
import numpy as np
import sys
import socket
import pickle
import math
import os
import json
from tqdm import tqdm
import torch

from pytorch3d.io import load_obj,save_ply


def get_lines(verts,faces,threshold):
    line_segs = np.zeros((faces.shape[0]*3,6))
    counter = 0
    checked_pairs = []
    for i in range(faces.shape[0]):
        for pair in [[0,1],[1,2],[2,0]]:
            indices = [faces[i,pair[0]],faces[i,pair[1]]]
            if indices not in checked_pairs and indices[::-1] not in checked_pairs:
                line_segs[counter] = np.concatenate((verts[indices[0]],verts[indices[1]]))
                checked_pairs.append(indices)
                counter += 1

    mask = np.linalg.norm(line_segs[:,:3] - line_segs[:,3:6],axis=1) > threshold
    return line_segs[mask]

def sample_points_from_lines(lines,points_per_line):
    n_lines = lines.shape[0]
    lines = torch.repeat_interleave(lines,points_per_line,dim=0)
    interval = torch.linspace(0,1,points_per_line).repeat(n_lines)
    interval = interval.unsqueeze(1).repeat(1,3)
    points = lines[:,:3] + lines[:,3:6] * interval
    return points

def make_folder_check(path,path2=None):
    if not os.path.exists(path):
        os.mkdir(path)
    
    if path2 != None:
        if not os.path.exists(path2):
            os.mkdir(path2)



def main():

    

    target_folder = '/scratch/fml35/datasets/pix3d_new/own_data/rendered_models/3d_lines/exp_01'
    pix_path = '/scratch/fml35/datasets/pix3d_new/'
    threshold = 0.4
    points_per_line_vis = 30

    make_folder_check(target_folder)
    make_folder_check(target_folder + '/lines')
    make_folder_check(target_folder + '/visualisation')

    # load model list
    with open("/data/cornucopia/fml35/experiments/exp_024_debug/models/model_list.json",'r') as f:
        model_list = json.load(f)["models"]

    for j in tqdm(range(0,len(model_list))):

        make_folder_check(target_folder + '/visualisation/' + model_list[j]["category"])
        make_folder_check(target_folder + '/lines/' + model_list[j]["category"])

        line_path = target_folder + '/lines/' + model_list[j]["category"] + '/' + model_list[j]["model"].split('/')[2]
        vis_path = target_folder + '/visualisation/' + model_list[j]["category"] + '/' + model_list[j]["model"].split('/')[2]

        make_folder_check(line_path,vis_path)

        path = pix_path + model_list[j]["model"]

        verts,faces,_ = load_obj(path,load_textures=False)

        verts = verts.numpy()
        faces = faces[0].numpy()

        line_segs = get_lines(verts,faces,threshold)

        # line segs have shape Nx6 because contain start and endpoint
        point_plus_dir = line_segs.copy()
        point_plus_dir[:,3:6] = point_plus_dir[:,3:6] - point_plus_dir[:,:3]
        np.save(line_path + '/lines.npy',point_plus_dir)

        points_vis = sample_points_from_lines(torch.from_numpy(point_plus_dir),points_per_line_vis)

        save_ply(vis_path + '/points_vis.ply',points_vis)



    
main()