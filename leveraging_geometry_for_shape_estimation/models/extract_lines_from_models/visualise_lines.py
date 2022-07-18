
import numpy as np

from glob import glob
import json
from tqdm import tqdm
import torch
from pytorch3d.io import load_obj,save_ply

def sample_points_from_lines(lines,points_per_line):
    n_lines = lines.shape[0]
    lines = torch.repeat_interleave(lines,points_per_line,dim=0)
    interval = torch.linspace(0,1,points_per_line).repeat(n_lines)
    interval = interval.unsqueeze(1).repeat(1,3)
    points = lines[:,:3] + (lines[:,3:6]-lines[:,:3]) * interval
    return points

def main(input_dir,output_dir,n_points_per_line):

    for path in tqdm(glob(input_dir + '/*')):
        lines_3D = np.load(path)
        points = sample_points_from_lines(torch.from_numpy(lines_3D),n_points_per_line)
        save_ply(path.replace(input_dir,output_dir).replace('.npy','.ply'),points)




if __name__ == '__main__':

    input_dir = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_117_scannet_models/models/lines_filtered'
    output_dir = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_117_scannet_models/models/lines_filtered_visualised'
    n_points_per_line = 100
    main(input_dir,output_dir,n_points_per_line)