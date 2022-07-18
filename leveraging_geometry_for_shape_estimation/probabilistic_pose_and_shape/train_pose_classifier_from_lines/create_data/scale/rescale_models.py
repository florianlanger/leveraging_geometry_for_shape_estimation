
from glob import glob
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj
import torch
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.ground_plane import get_model_to_infos_scannet
import os
from tqdm import tqdm

model_to_infos_scannet = get_model_to_infos_scannet()


dir_path = '/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/model/'

for path in tqdm(sorted(glob(dir_path + '*/*/*'))):

    vertices_origin,faces,_ = load_obj(path,create_texture_atlas=False, load_textures=False)

    name = path.split('/')[-3] + '_' + path.split('/')[-2]
    factor = torch.Tensor(model_to_infos_scannet[name]['bbox']) * 2
    vertices_rescaled = vertices_origin / factor

    out_path = path.replace('/model/','/model_rescaled/')
    out_path_without_last = out_path.rsplit('/',1)[0]
    os.makedirs(out_path_without_last, exist_ok=True)

    save_obj(out_path,vertices_rescaled,faces[0])
