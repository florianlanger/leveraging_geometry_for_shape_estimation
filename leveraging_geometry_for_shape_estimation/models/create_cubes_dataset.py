
import os
import numpy as np
import torch
import json
import pytorch3d

from pytorch3d.io import save_obj

target_dir = "/scratch/fml35/datasets/cubes/"

# os.mkdir(target_dir + 'models')

def get_unit_cube():
    verts = np.array([[-1, -1, -1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [ 1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [-1,  1,  1],
        [ 1,  1,  1]])

    faces = np.array([
    [0,2,1],
    [2,3,1],
    [2,6,3],
    [6,7,3],
    [1,3,5],
    [3,7,5],
    [2,0,4],
    [6,2,4],
    [0,1,5],
    [5,4,0],
    [4,7,6],
    [4,5,7]])
    return verts,faces

list_models = []

def main():

    list_models = []

    for i in range(5):

        random_scales = 0.2 + 0.3 * np.random.rand(3)
        verts,faces = get_unit_cube()
        verts = verts * random_scales

        name = 'model_{}'.format(str(i).zfill(3)) + '.obj'

        pytorch3d.io.save_obj(target_dir + 'models/' + name,verts=torch.from_numpy(verts),faces=torch.from_numpy(faces))


        info = {"name": name,"scales":random_scales.tolist()}
        list_models.append(info)

    with open(target_dir + 'model_info.json','w') as file:
        json.dump(list_models,file,indent = 4)

if __name__ == '__main__':
    main()


