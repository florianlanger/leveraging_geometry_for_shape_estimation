
import torch
from tqdm import tqdm
import os
import json
import cv2
import numpy as np
from numpy.lib.utils import info
import torch
import os
import sys
import json

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform,FoVPerspectiveCameras,RasterizationSettings, MeshRenderer, MeshRasterizer,SoftPhongShader,Textures)
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj
from scipy.spatial.transform import Rotation as scipy_rot
import quaternion
import collections

from leveraging_geometry_for_shape_estimation.utilities.dicts import open_json_precomputed_or_current,determine_base_dir



def category_to_cat_id(category):                                                                                                                                                                                                                                                                                           
    top = {}
    top["03211117"] = "display"
    top["04379243"] = "table"
    top["02808440"] = "bathtub"
    top["02747177"] = "bin"
    top["04256520"] = "sofa"
    top["03001627"] = "chair"
    top["02933112"] = "cabinet"
    top["02871439"] = "bookshelf"
    top["02818832"] = "bed"
    inv_map = {v: k for k, v in top.items()}
    return inv_map[category]


def get_score_for_folder(global_config):

    target_folder = global_config["general"]["target_folder"]

    all_infos = {}

    for name in tqdm(os.listdir(target_folder + '/selected_nn')):
        with open(target_folder + '/selected_nn/' + name,'r') as f:
            selected = json.load(f)
        number_nn = selected["selected_nn"]
        
        # with open(target_folder + '/nn_infos/' + name,'r') as f:
        #     retrieval_list = json.load(f)["nearest_neighbours"]

        # with open(target_folder + '/segmentation_infos/' + name ,'r') as f:
        #     segmentation_infos = json.load(f)

        name_pose = name.split('.')[0] + '_' + str(number_nn).zfill(3) + '_' + str(selected["selected_orientation"]).zfill(2) + '.json'

        with open(target_folder + '/poses/' + name_pose,'r') as f:
            pose_info = json.load(f)


        retrieval_list = open_json_precomputed_or_current('/nn_infos/' + name,global_config,'retrieval')["nearest_neighbours"]
        segmentation_infos = open_json_precomputed_or_current('/segmentation_infos/' + name,global_config,'segmentation')

        number_nn = selected["selected_nn"]

        infos = {}

        infos["score"] = segmentation_infos["predictions"]["score"]
        infos["bbox"] = segmentation_infos["predictions"]["bbox"]
        infos["t"] = pose_info["predicted_t"]
        
        q = quaternion.from_rotation_matrix(pose_info["predicted_r"])
        q = quaternion.as_float_array(q)
        infos["q"] = q.tolist()

        infos["s"] = pose_info["predicted_s"]
        category = segmentation_infos["predictions"]["category"]
        id_cad = retrieval_list[number_nn]["model"].split('/')[2]
        infos["category"] = category
        infos["scene_cad_id"] = [category_to_cat_id(category),id_cad]
        infos["detection"] = name.split('.')[0]

        key = name.split('-')[0] + '/color/' + name.split('-')[1].split('_')[0] + '.jpg'
        
        if key in all_infos:
            all_infos[key].append(infos)
        else:
            all_infos[key] = [infos]

    with open(target_folder + '/global_stats/eval_scannet/single_frame.json','w') as f:
        json.dump(all_infos,f,indent=4)





def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)


    get_score_for_folder(global_config)

if __name__ == '__main__':
    print('Combine Predictions')
    main()