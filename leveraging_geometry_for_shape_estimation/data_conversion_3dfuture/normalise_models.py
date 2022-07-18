
import json
from pytorch3d.io import load_obj,save_obj
import os
import torch
from tqdm import tqdm


dir_path = '/scratches/octopus/fml35/datasets/3d_future/3D-FUTURE-model_reformatted/model'
out_path = '/scratches/octopus/fml35/datasets/3d_future/3D-FUTURE-model_reformatted/model_normalised'




for cat in os.listdir(dir_path)[3:4]:
    infos = {}
    if not os.path.exists(os.path.join(out_path, cat)):
        os.mkdir(out_path + '/' + cat)
    for id in tqdm(os.listdir(dir_path + '/' + cat)):

        # if os.path.exists(out_path + '/' + cat + '/' + id):
        #     continue

        # if out_path + '/' + cat + '/' + id + '/model_normalized_own.obj' != '/scratch/fml35/datasets/3d_future/3D-FUTURE-model_reformatted/model_normalised/cabinetshelfdesk/af457c93-97ea-4067-9281-ec2fe19b8b83/model_normalized_own.obj':
        #     continue
        # assert os.path.exists(out_path + '/' + cat + '/' + id + '/model_normalized_own.obj'),(out_path + '/' + cat + '/' + id + '/model_normalized_own.obj')

        if not os.path.exists(out_path + '/' + cat + '/' + id):
            os.mkdir(out_path + '/' + cat + '/' + id)

    

        verts,faces,_ = load_obj(dir_path + '/' + cat + '/' + id + '/raw_model.obj',load_textures=False)
        min,_ = torch.min(verts,dim=0)
        max,_ = torch.max(verts,dim=0)

        largest_scale = torch.max(max - min)
        verts = verts / largest_scale
        center = (max + min) / (2*largest_scale)
        verts = verts - center

        save_obj(out_path + '/' + cat + '/' + id + '/model_normalized_own.obj',verts,faces[0])
        min,_ = torch.min(verts,dim=0)
        max,_ = torch.max(verts,dim=0)

        infos[id] = {'scale': largest_scale.item(), 'center': center.tolist()}

    with open(out_path + '/infos/{}.json'.format(cat), 'w') as f:
        json.dump(infos, f)