
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

from leveraging_geometry_for_shape_estimation.data_conversion.create_dirs import dict_replace_value
from leveraging_geometry_for_shape_estimation.utilities.dicts import open_json_precomputed_or_current,determine_base_dir


def convert_metrics_to_json(shape_metrics):

    new_dict = {}
    for metric in shape_metrics:
        new_dict[metric] = shape_metrics[metric][0].item()

    new_dict['F1'] = shape_metrics['F1@0.300000'][0].item()

    return new_dict

def get_top8_classes_scannet():                                                                                                                                                                                                                                                                                           
    top = collections.defaultdict(lambda : "other")
    top["03211117"] = "display"
    top["04379243"] = "table"
    top["02808440"] = "bathtub"
    top["02747177"] = "bin"
    top["04256520"] = "sofa"
    top["03001627"] = "chair"
    top["02933112"] = "cabinet"
    top["02871439"] = "bookshelf"
    top["02818832"] = "bed"
    return top

def load_cad_info(file_path):
    with open(file_path,'r') as file:
        annotation = json.load(file)

    catid2catname = get_top8_classes_scannet()
    cad2info = {}
    for r in annotation:
        for model in r["aligned_models"]:
            id_cad = model["id_cad"]
            catid_cad = model["catid_cad"]
            catname_cad = catid2catname[catid_cad]
            cad2info[catname_cad + '_' + id_cad] = {"sym" : model["sym"], "catname" : catname_cad}
    # print(cad2info)
    return cad2info

def calc_rotation_diff(q, q00):
    rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
    rotation_dot_abs = np.abs(rotation_dot)
    try:                                                                                                                                                                                                                                                                                                                      
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:
        return 0.0
    error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    error_rotation = np.rad2deg(error_rotation_rad)
    return error_rotation


def rotation_diff_considering_symmetry(R_pred,R_gt,sym):

    if (np.abs(np.array(R_pred) - np.array(R_gt)) < 0.0000001).all():
        error_rotation = 0.0

    elif sym == "__SYM_ROTATE_UP_2":
        m = 2
        tmp = [calc_rotation_diff(quaternion.from_rotation_matrix(R_pred), quaternion.from_rotation_matrix(R_gt) *quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_4":
        m = 4
        tmp = [calc_rotation_diff(quaternion.from_rotation_matrix(R_pred), quaternion.from_rotation_matrix(R_gt)*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_INF":
        m = 36
        tmp = [calc_rotation_diff(quaternion.from_rotation_matrix(R_pred), quaternion.from_rotation_matrix(R_gt)*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    else:
        error_rotation = calc_rotation_diff(quaternion.from_rotation_matrix(R_pred), quaternion.from_rotation_matrix(R_gt))

    if np.isnan(error_rotation):
        print(R_pred)
        print(R_gt)
        print(sym)
        print(error_rotation)
        print(srop)
    return error_rotation

def get_score_for_folder(global_config):

    target_folder = global_config["general"]["target_folder"]


    device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))
    torch.cuda.set_device(device)

    cad2info = load_cad_info(global_config["dataset"]["dir_path_images"] + '../scan2cad_annotations/full_annotations.json')

    list_to_do = ['scene0011_00-000700_01_000_03.json','scene0011_00-001600_00_000_01.json','scene0011_00-000800_00_000_03.json' ,'scene0011_00-001600_05_000_01.json',
'scene0011_00-000800_01_000_03.json','scene0011_00-001700_00_000_01.json',
'scene0011_00-000900_00_000_03.json','scene0011_00-001800_00_000_00.json',
'scene0011_00-001100_00_000_03.json','scene0011_00-001800_01_000_01.json',
'scene0011_00-001100_04_000_00.json','scene0011_00-001900_00_000_00.json','scene0011_00-001200_00_000_00.json','scene0011_00-002000_00_000_00.json',
'scene0011_00-001300_00_000_03.json','scene0011_00-002000_01_000_00.json','scene0011_00-001500_00_000_03.json']

    list_to_do_mod = [name.rsplit('_',2)[0] + '.json' for name in list_to_do]

    print(target_folder + '/selected_nn')
    for name in tqdm(os.listdir(target_folder + '/selected_nn')):
        if name not in list_to_do_mod:
            continue
        print('name: ', name)
        with open(target_folder + '/selected_nn/' + name,'r') as f:
            selected = json.load(f)
        number_nn = selected["selected_nn"]
        
        retrieval_list = open_json_precomputed_or_current('/nn_infos/' + name,global_config,"retrieval")["nearest_neighbours"]
        # with open(target_folder + '/nn_infos/' + name,'r') as f:
        #     retrieval_list = json.load(f)["nearest_neighbours"]

        gt_infos = open_json_precomputed_or_current('/gt_infos/' + name.rsplit('_',1)[0] + '.json',global_config,"segmentation")
        # with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
        #     gt_infos = json.load(f)

        if gt_infos["objects"] == []:
            continue

        name_pose = name.split('.')[0] + '_' + str(number_nn).zfill(3) + '_' + str(selected["selected_orientation"]).zfill(2) + '.json'

        out_path = target_folder + '/metrics_scannet/' + name_pose
        # if os.path.exists(out_path):
        #     continue

        # if not os.path.exists(target_folder + '/poses/' + name_pose):
        #     continue

        with open(target_folder + '/poses/' + name_pose,'r') as f:
            pose_info = json.load(f)

        # with open(target_folder + '/poses_R/' + name_pose,'r') as f:
        #     pose_info_R = json.load(f)
        pose_info_R = open_json_precomputed_or_current('/poses_R/' + name_pose,global_config,"R")

        # with open(target_folder + '/bbox_overlap/' + name.split('.')[0] + '.json','r') as f:
        #     bbox_overlap = json.load(f)
        bbox_overlap = open_json_precomputed_or_current('/bbox_overlap/' + name.split('.')[0] + '.json',global_config,"segmentation")

        # predicted obj
        # model_path_pred = models_folder_read + "/models/remeshed/" + retrieval_list[i]["model"].replace('model/','')
        # number_nn = int(name.rsplit('_',2)[1].split('_')[0]) #.split('.')[0])
        number_nn = selected["selected_nn"]

        model_path_pred = global_config["dataset"]["dir_path"] + retrieval_list[number_nn]["model"]
        # R_pred = pose_info["predicted_r"]
        R_pred = pose_info_R["predicted_r"]
        T_pred = pose_info["predicted_t"]
        S_pred = pose_info["predicted_s"]

        model_path_gt = global_config["dataset"]["dir_path"] + gt_infos["objects"][bbox_overlap['index_gt_objects']]["model"]
        R_gt = gt_infos["objects"][bbox_overlap['index_gt_objects']]["rot_mat"]
        T_gt = gt_infos["objects"][bbox_overlap['index_gt_objects']]["trans_mat"]
        S_gt = gt_infos["objects"][bbox_overlap['index_gt_objects']]["scaling"]



        error_translation = np.linalg.norm(np.array(T_pred) - np.array(T_gt), ord=2)
        error_scale = 100.0*np.abs(np.mean(np.array(S_gt)/np.array(S_pred)) - 1)

        # --> resolve symmetry
        sym = cad2info[retrieval_list[number_nn]["name"]]["sym"]
        error_rotation = rotation_diff_considering_symmetry(R_pred,R_gt,sym)

        
        combined_metrics = {}
        combined_metrics["rotation_error"] = error_rotation
        combined_metrics["translation_error"] = error_translation
        combined_metrics["scaling_error"] = error_scale
        combined_metrics["scaling_error_all"] = (100.0*(np.array(S_gt)/np.array(S_pred) - 1)).tolist()
        combined_metrics["translation_error_all"] = (np.array(T_pred) - np.array(T_gt)).tolist()
        combined_metrics["rotation_correct"] = bool(error_rotation < global_config["evaluate_poses"]["scannet"]["max_r_error"])
        combined_metrics["translation_correct"] = bool(error_translation < global_config["evaluate_poses"]["scannet"]["max_t_error"])
        combined_metrics["scaling_correct"] = bool(error_scale < global_config["evaluate_poses"]["scannet"]["max_s_error"])
        combined_metrics["retrieval_correct"] = model_path_gt == model_path_pred


        with open(out_path,'w') as f:
            json.dump(combined_metrics,f,indent=4)





def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    # global_config = dict_replace_value(global_config,'/scratches/octopus_2/fml35/','/scratch2/fml35/')

    get_score_for_folder(global_config)

if __name__ == '__main__':
    print('Compute metrics')
    main()