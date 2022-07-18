import sys
import os
from pytorch3d.io import load_ply
import numpy as np
from tqdm import tqdm

from leveraging_geometry_for_shape_estimation.utilities_3d.utilities import writePlyFile


def main(folder_1,folder_2,folder_output,color_1,color_2):

    list_dir_1 = os.listdir(folder_1)
    list_dir_2 = os.listdir(folder_2)
    # assert sorted(list_dir_1) == sorted(list_dir_2)

    for file in tqdm(list_dir_1):
        verts1,_ = load_ply(folder_1 + '/' + file)
        verts2,_ = load_ply(folder_2 + '/' + file)
        combined_verts = np.concatenate([verts1.numpy(),verts2.numpy()])
        colors = np.concatenate([verts1.numpy()*0 + color_1,verts2.numpy()*0 + color_2])
        print(combined_verts.shape)
        print(colors.shape)
        print(colors[0])
        print(colors[-1])
        writePlyFile(folder_output + '/' + file,combined_verts,colors)




if __name__ == '__main__':
    folder_1 = sys.argv[1]
    folder_2 = sys.argv[2]
    folder_output = sys.argv[3]
    color_1 = sys.argv[4]
    color_2 = sys.argv[5]

    # os.mkdir(folder_output)
    
    colors = {'red':[255,0,0],'green':[0,255,0],'blue':[0,0,255],'yellow':[255,255,0]}
    assert color_1 in colors and color_2 in colors
    main(folder_1,folder_2,folder_output,colors[color_1],colors[color_2])