
import re
from sys import version
from numpy.lib.npyio import save
import torch
import trimesh
from datetime import datetime
import os
import trimesh
import json
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj
from pytorch3d.renderer import Textures
import sys
from tqdm import tqdm


def remesh(model_path,max_edge_length,device):
    vertices_original,faces_original,properties = load_obj(model_path, device=device,create_texture_atlas=False, load_textures=False)
    vertices_remeshed,faces_remeshed = trimesh.remesh.subdivide_to_size(vertices_original.cpu().numpy(), faces_original[0].cpu().numpy(), max_edge=max_edge_length)
    # vertices_remeshed = vertices_original.cpu().numpy()
    # faces_remeshed = faces_original[0].cpu().numpy()
    vertices = torch.from_numpy(vertices_remeshed).to(torch.float32).to(device)
    faces = torch.from_numpy(faces_remeshed).to(device)
    verts_rgb = torch.ones_like(vertices)[None]
    textures = Textures(verts_rgb=verts_rgb)
    return vertices,faces,textures

def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    with open(global_config["general"]["target_folder"] + "/models/model_list.json",'r') as f:
        model_list = json.load(f)["models"]

    device = torch.device("cpu")

    for j in tqdm(range(0,len(model_list))):

        cat_path = global_config["general"]["target_folder"] + "/models/remeshed/" + model_list[j]["category"]
        if not os.path.exists(cat_path):
            os.mkdir(cat_path)

        
        model_path = global_config["general"]["target_folder"] + "/models/remeshed/" + model_list[j]["category"] + '/' + model_list[j]["model"].split('/')[2]
        out_path = model_path + '/' + model_list[j]["model"].split('/')[3]
        
        if os.path.exists(out_path):
            continue

        if not os.path.exists(model_path):
            os.mkdir(model_path)
        
        # obj_path = global_config["dataset"]["dir_path"] + model_list[j]["model"]
        obj_path = global_config["dataset"]["dir_path"] + model_list[j]["model"]
        verts,faces,_ = remesh(obj_path,global_config["models"]["max_edge_length_remesh"],device)
        
        save_obj(out_path,verts,faces)

if __name__ == '__main__':
    main()