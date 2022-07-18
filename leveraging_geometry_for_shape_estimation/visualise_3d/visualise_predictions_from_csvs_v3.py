
import numpy as np
np.warnings.filterwarnings('ignore')
import sys
import os
import collections
import quaternion
import operator
import glob
np.seterr(all='raise')
import argparse
import torch
from tqdm import tqdm
import json

from leveraging_geometry_for_shape_estimation.eval.scannet import CSVHelper
from leveraging_geometry_for_shape_estimation.utilities_3d.utilities import writePlyFile
from leveraging_geometry_for_shape_estimation.utilities.folders import make_dir_save

from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes

def load_mesh(shapenet_path,category,model_id,r,t,s):

    vertices,faces,properties = load_obj('{}/{}/{}/model_normalized.obj'.format(shapenet_path,category.replace('bookcase','bookshelf'),model_id),create_texture_atlas=False, load_textures=False)

    vertices = vertices.cpu().numpy()
    vertices_scaled = np.array(s) * vertices
    vertices =  np.matmul(np.array(r),vertices_scaled.T).T + np.array(t)
    return vertices,faces[0].numpy()


# get top8 (most frequent) classes from annotations. 
def get_top9_classes_scannet():                                                                                                                                                                                                                                                                                           
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
    return top


def get_verts_and_faces_from_alignment(alignment,catid_to_cat_name,shape_net_dir):
    catid_cad = str(alignment[0]).zfill(8)
    id_cad = alignment[1]
    category = catid_to_cat_name[catid_cad]

    
    t = np.asarray(alignment[2:5], dtype=np.float64)
    q0 = np.asarray(alignment[5:9], dtype=np.float64)
    q = np.quaternion(q0[0], q0[1], q0[2], q0[3])
    r = quaternion.as_rotation_matrix(q)
    s = np.asarray(alignment[9:12], dtype=np.float64)

    vertices,faces = load_mesh(shape_net_dir,category,id_cad,r,t,s)
    return vertices,faces


def get_color_from_detection(detection,combined_predictions_with_roca):
    gt_name = detection.split('-')[0] + '/color/' + detection.split('-')[1].split('_')[0] + '.jpg'
    detection_id = int(detection.rsplit('_',1)[1])
    own_prediction = combined_predictions_with_roca[gt_name][detection_id]['own_prediction']
    if own_prediction == True:
        color = np.array([0,0,255])
    elif own_prediction == False:
        color = np.array([255,0,0])

    # color = np.array([200,200,0])
    return color


def visualise(projectdir,output_dir,shape_net_dir,scene_results,combined_predictions_with_roca,max_number_scenes=10000):

        catid_to_cat_name = get_top9_classes_scannet()
        device = torch.device("cpu")

        counter = 0
        for file0 in tqdm(sorted(glob.glob(projectdir + "/*.csv"))):

            if counter == max_number_scenes:
                break
            
            alignments = CSVHelper.read(file0)
            id_scan = os.path.basename(file0.rsplit(".", 1)[0])
            all_points = []

            # if not 'scene0300_00' in file0:
            #     continue
            vertex_colors = []

            for alignment in alignments: # <- multiple alignments of same object in scene
                
                vertices,faces = get_verts_and_faces_from_alignment(alignment,catid_to_cat_name,shape_net_dir)
                mesh = Meshes(verts=[torch.Tensor(vertices).to(device)],faces=[torch.Tensor(faces).to(device)])
                sampled_points = sample_points_from_meshes(mesh,num_samples=4000)[0].cpu().numpy()
                all_points.append(sampled_points)
                color = get_color_from_detection(alignment[14],combined_predictions_with_roca)
                vertex_colors.append(sampled_points * 0 + color)

            file_name = output_dir + '/' + id_scan + '_accuracy_' + str(int(np.round(100*scene_results[id_scan]["accuracy"]))).zfill(3) + '_ntotal_' + str(scene_results[id_scan]["n_total"]).zfill(3) + '_ncorrect_' + str(scene_results[id_scan]["n_good"]).zfill(3) + '.ply'
            vertices = np.concatenate(all_points)
            vertex_colors = np.concatenate(vertex_colors)
            writePlyFile(file_name,vertices,vertex_colors)

            counter += 1


def visualise_single_images(projectdir,output_dir,shape_net_dir,combined_predictions_with_roca,max_number_scenes=10000):

        catid_to_cat_name = get_top9_classes_scannet()
        device = torch.device("cpu")

        for file0 in tqdm(glob.glob(projectdir + "/*.csv")[:max_number_scenes]):
            alignments = CSVHelper.read(file0)
            imgs = [alignment[13] for alignment in alignments]
            imgs = set(imgs)
            for img in imgs:
                id_scan = os.path.basename(file0.rsplit(".", 1)[0])
                all_points = []
                vertex_colors = []

                for alignment in alignments: # <- multiple alignments of same object in scene

                    if alignment[13] != img:
                        continue
                    
                    vertices,faces = get_verts_and_faces_from_alignment(alignment,catid_to_cat_name,shape_net_dir)
                    mesh = Meshes(verts=[torch.Tensor(vertices).to(device)],faces=[torch.Tensor(faces).to(device)])
                    sampled_points = sample_points_from_meshes(mesh,num_samples=4000)[0].cpu().numpy()
                    all_points.append(sampled_points)
                    color = get_color_from_detection(alignment[14],combined_predictions_with_roca)
                    vertex_colors.append(sampled_points * 0 + color)

                file_name = output_dir + '/' + id_scan + '-' + img.split('/')[2].split('.')[0] + '.ply'
                vertices = np.concatenate(all_points)
                vertex_colors = np.concatenate(vertex_colors)
                writePlyFile(file_name,vertices,vertex_colors)



       


if __name__ == "__main__":

    # exp = sys.argv[1]
    # shape_net_dir = '/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/model'
    # print('ENABLE ALL SCENES VIS')

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    exp = global_config["general"]["target_folder"]
    shape_net_dir = global_config["dataset"]["dir_path"] + 'model'

    with open(exp + '/global_stats/results_scannet_scenes_without_retrieval.json','r') as f:
    # with open(exp + '/global_stats/results_scannet_scenes_strict_rotation_filtered_no_rotation_without_retrieval.json','r') as f:
        scene_results = json.load(f)

    # RGB
    color = np.array([0,0,255])

    for add_on in ['','_filtered','_scan2cad_constraints']:
    # for add_on in ['_scan2cad_constraints']:
    # for add_on in ['_scan2cad_constraints_filtered_no_rotation']:
        projectdir = exp + '/global_stats/eval_scannet/results_per_scene' + add_on
        output_dir = projectdir + '_visualised'
        make_dir_save(output_dir,assert_not_exist=False)
        visualise(projectdir,output_dir,shape_net_dir,scene_results,combined_predictions_with_roca)

    # for add_on in ['','_filtered']:
    for add_on in ['']:
        projectdir = exp + '/global_stats/eval_scannet/results_per_scene' + add_on
        output_dir = exp + '/global_stats/eval_scannet/results_per_frame' + add_on + '_visualised'
        visualise_single_images(projectdir,output_dir,shape_net_dir,color)