
import numpy as np
np.warnings.filterwarnings('ignore')
import sys
import os
import collections
import quaternion
import operator
from glob import glob
np.seterr(all='raise')
import argparse
import torch
from tqdm import tqdm
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)

from leveraging_geometry_for_shape_estimation.eval.scannet import CSVHelper,JSONHelper,SE3
from leveraging_geometry_for_shape_estimation.utilities_3d.utilities import writePlyFile
from leveraging_geometry_for_shape_estimation.visualise_3d.visualise_predictions_from_csvs import load_mesh,get_top9_classes_scannet

from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj


from pytorch3d.ops import sample_points_from_meshes

def load_4by4_from_txt(path):
    M = np.zeros((4,4))
    with open(path,'r') as f:
        content = f.readlines()
        for i in range(4):
            line = content[i].split()
            for j in range(4):
                M[i,j] = np.float32(line[j])
        return M

def vertices_and_faces_from_model_anno(model_gt,Mscan,shapenet_path,catid_to_cat_name):
    Mcad = SE3.compose_mat4(model_gt["trs"]["translation"], model_gt["trs"]["rotation"], model_gt["trs"]["scale"])
    t_gt, q_gt, s_gt = SE3.decompose_mat4(np.dot(np.linalg.inv(Mscan), Mcad))
    r_gt = quaternion.as_rotation_matrix(q_gt)
    model_id = model_gt["id_cad"]
    category = catid_to_cat_name[model_gt["catid_cad"]]

    vertices,faces = load_mesh(shapenet_path,category,model_id,r_gt,t_gt,s_gt)
    return vertices,faces

def check_counters_0(scene_info,catid_to_cat_name):
    for model in scene_info:
        catid = model.split('_')[0]
        if catid in catid_to_cat_name:
            assert scene_info[model] == 0

def get_line_points(p1,p2,N_points):
    p1 = np.array(p1)
    p2 = np.array(p2)
    interval = np.tile(np.linspace(0,1,N_points),(3,1)).T
    return p1 + interval * (p2 - p1)

def create_camera_vertices():
    N_points_line = 100
    p0 = [0,0,0]
    p1 = [-0.5,0.5,1.5]
    p2 = [0.5,0.5,1.5]
    p3 = [0.5,-0.5,1.5]
    p4 = [-0.5,-0.5,1.5]
    # p5 = [1,0,1.5]
    p5 = [0,-1,1.5]

    lines = []
    lines.append(get_line_points(p0,p1,N_points_line))
    lines.append(get_line_points(p0,p2,N_points_line))
    lines.append(get_line_points(p0,p3,N_points_line))
    lines.append(get_line_points(p0,p4,N_points_line))
    lines.append(get_line_points(p1,p2,N_points_line))
    lines.append(get_line_points(p2,p3,N_points_line))
    lines.append(get_line_points(p3,p4,N_points_line))
    lines.append(get_line_points(p4,p1,N_points_line))
    lines.append(get_line_points(p4,p5,N_points_line))
    lines.append(get_line_points(p3,p5,N_points_line))

    return np.concatenate(lines)


def calc_Mbbox(model):
    trs_obj = model["trs"]
    bbox_obj = np.asarray(model["bbox"], dtype=np.float64)
    center_obj = np.asarray(model["center"], dtype=np.float64)
    trans_obj = np.asarray(trs_obj["translation"], dtype=np.float64)
    rot_obj = np.asarray(trs_obj["rotation"], dtype=np.float64)
    q_obj = np.quaternion(rot_obj[0], rot_obj[1], rot_obj[2], rot_obj[3])
    scale_obj = np.asarray(trs_obj["scale"], dtype=np.float64)

    tcenter1 = np.eye(4)
    tcenter1[0:3, 3] = center_obj
    trans1 = np.eye(4)
    trans1[0:3, 3] = trans_obj
    rot1 = np.eye(4)
    rot1[0:3, 0:3] = quaternion.as_rotation_matrix(q_obj)
    scale1 = np.eye(4)
    scale1[0:3, 0:3] = np.diag(scale_obj)
    bbox1 = np.eye(4)
    bbox1[0:3, 0:3] = np.diag(bbox_obj)
    M = trans1.dot(rot1).dot(scale1).dot(tcenter1).dot(bbox1)
    return M


def get_points_cube():
    N_points_line = 100

    p1 = [-1,-1,1]
    p2 = [-1,1,1]
    p3 = [1,-1,1]
    p4 = [1,1,1]
    p5 = [-1,-1,-1]
    p6 = [-1,1,-1]
    p7 = [1,-1,-1]
    p8 = [1,1,-1]

    lines = []
    lines.append(get_line_points(p1,p2,N_points_line))
    lines.append(get_line_points(p1,p3,N_points_line))
    lines.append(get_line_points(p2,p4,N_points_line))
    lines.append(get_line_points(p4,p3,N_points_line))
    lines.append(get_line_points(p5,p6,N_points_line))
    lines.append(get_line_points(p5,p7,N_points_line))
    lines.append(get_line_points(p8,p6,N_points_line))
    lines.append(get_line_points(p8,p7,N_points_line))
    lines.append(get_line_points(p1,p5,N_points_line))
    lines.append(get_line_points(p2,p6,N_points_line))
    lines.append(get_line_points(p3,p7,N_points_line))
    lines.append(get_line_points(p4,p8,N_points_line))
    return np.concatenate(lines)


def plot_cameras(scene_id,camera_vertices):

    all_cameras = []
    for frame_4by4_path in glob('/scratches/octopus_2/fml35/datasets/scannet/scannet_frames_25k/' + scene_id + '/pose/*'):
        R_and_T = load_4by4_from_txt(frame_4by4_path)
        R_pose = R_and_T[:3,:3].copy()
        T_pose = R_and_T[:3,3].copy()
        transformed_verts = np.matmul(R_pose,camera_vertices.T).T * 0.2 + T_pose
        all_cameras.append(transformed_verts)
    return np.concatenate(all_cameras)

def get_points_bbox(model_gt,points_cube,Mscan):
    # Mcad = SE3.compose_mat4(model_gt["trs"]["translation"], model_gt["trs"]["rotation"], model_gt["bbox"])
    Mbbox = calc_Mbbox(model_gt)
    t_gt, q_gt, s_gt = SE3.decompose_mat4(np.dot(np.linalg.inv(Mscan), Mbbox))
    r_gt = quaternion.as_rotation_matrix(q_gt)

    vertices_scaled = np.array(s_gt) * points_cube
    vertices =  np.matmul(np.array(r_gt),vertices_scaled.T).T + np.array(t_gt)

    return vertices

def get_points_bbox_false(model,mesh_bbox,Mscan):
    Mbbox = calc_Mbbox(model)
    verts_bbox = []
    for v in mesh_bbox["vertex"]: 
        v1 = np.array([v[0], v[1], v[2], 1])
        v1 = np.dot(Mbbox, v1)[0:3]
        verts_bbox.append(v1)

    verts = np.vstack(verts_bbox)

    t_gt, q_gt, s_gt = SE3.decompose_mat4(np.dot(np.linalg.inv(Mscan), Mbbox))

    vertices_scaled = np.array(s_gt) * vertices

    r_gt = quaternion.as_rotation_matrix(q_gt)
    vertices =  np.matmul(np.array(r_gt),vertices_scaled.T).T + np.array(t_gt)

    return vertices

def visualise(filename_cad_appearance,filename_annotations,shapenet_path,output_dir,color):

        camera_vertices = create_camera_vertices()
        points_cube = get_points_cube()

        scenes_no_objects = []

        with open("/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/visualise_3d/bbox.ply", 'rb') as read_file:
            mesh_bbox = PlyData.read(read_file)


        appearances_cad = JSONHelper.read(filename_cad_appearance)
        annotations = JSONHelper.read(filename_annotations)
        scene_names = [r["id_scan"] for r in annotations]

        assert set(scene_names) == set(appearances_cad.keys())

        device = torch.device("cpu")
        idscan2trs = {}
        catid_to_cat_name = get_top9_classes_scannet()

        scenes_no_objects = ['scene0678_00','scene0016_02','scene0016_01','scene0071_00','scene0531_00','scene0044_02','scene0044_01','scene0635_01','scene0375_02','scene0636_00','scene0692_04','scene0074_00','scene0421_00']

        for r in tqdm(annotations):

            
            all_points = []
            assert r['n_aligned_models'] == len(r["aligned_models"])

            id_scan = r["id_scan"]

            if id_scan in scenes_no_objects:
                continue

            idscan2trs[id_scan] = r["trs"]
            Mscan = SE3.compose_mat4(idscan2trs[id_scan]["translation"], idscan2trs[id_scan]["rotation"], idscan2trs[id_scan]["scale"])

            all_cameras = plot_cameras(id_scan,camera_vertices)
            all_points.append(all_cameras)
            

            for idx, model_gt in enumerate(r["aligned_models"]):
                if model_gt["catid_cad"] in catid_to_cat_name:
                    vertices,faces = vertices_and_faces_from_model_anno(model_gt,Mscan,shapenet_path,catid_to_cat_name)
                    mesh = Meshes(verts=[torch.Tensor(vertices).to(device)],faces=[torch.Tensor(faces).to(device)])
                    sampled_points = sample_points_from_meshes(mesh,num_samples=10000)
                    all_points.append(sampled_points[0].cpu().numpy())
                    points_bbox = get_points_bbox(model_gt,points_cube,Mscan)
            
                    all_points.append(points_bbox)

                    combined_name = model_gt["catid_cad"] + '_' + model_gt["id_cad"]
                    appearances_cad[id_scan][combined_name] = appearances_cad[id_scan][combined_name] - 1

            check_counters_0(appearances_cad[id_scan],catid_to_cat_name)

            file_name = output_dir + '/' + id_scan + '.ply'
            vertices = np.concatenate(all_points)
            vertex_colors = vertices * 0 + color
            writePlyFile(file_name,vertices,vertex_colors)


            

if __name__ == "__main__":

    filename_cad_appearance = '/scratches/octopus_2/fml35/datasets/scannet/scan2cad_annotations/cad_appearances.json'
    filename_annotations = '/scratches/octopus_2/fml35/datasets/scannet/scan2cad_annotations/full_annotations.json'
    shape_net_dir = '/scratches/octopus_2/fml35/datasets/shapenet_v2/ShapeNetRenamed/model'
    output_dir = '/scratches/octopus_2/fml35/datasets/scannet/scenes_visualised_cameras_bbox'
    color = [0,255,0]

    # os.mkdir(output_dir)
    visualise(filename_cad_appearance,filename_annotations,shape_net_dir,output_dir,color)