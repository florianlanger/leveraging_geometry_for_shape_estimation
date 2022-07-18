
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




def visualise(projectdir,output_dir,shape_net_dir,color):

        catid_to_cat_name = get_top9_classes_scannet()
        device = torch.device("cpu")

        for file0 in tqdm(glob.glob(projectdir + "/*.csv")):
            alignments = CSVHelper.read(file0)
            id_scan = os.path.basename(file0.rsplit(".", 1)[0])
            all_points = []

            # if not 'scene0300_00' in file0:
            #     continue

            for alignment in alignments: # <- multiple alignments of same object in scene
                
                vertices,faces = get_verts_and_faces_from_alignment(alignment,catid_to_cat_name,shape_net_dir)
                mesh = Meshes(verts=[torch.Tensor(vertices).to(device)],faces=[torch.Tensor(faces).to(device)])
                sampled_points = sample_points_from_meshes(mesh,num_samples=10000)
                all_points.append(sampled_points[0].cpu().numpy())

            file_name = output_dir + '/' + id_scan + '.ply'
            vertices = np.concatenate(all_points)
            vertex_colors = vertices * 0 + color
            writePlyFile(file_name,vertices,vertex_colors)


def visualise_single_images(projectdir,output_dir,shape_net_dir,color):

        catid_to_cat_name = get_top9_classes_scannet()
        device = torch.device("cpu")

        for file0 in tqdm(glob.glob(projectdir + "/*.csv")):
            alignments = CSVHelper.read(file0)
            imgs = [alignment[-1] for alignment in alignments]
            imgs = set(imgs)

            for img in imgs:

                id_scan = os.path.basename(file0.rsplit(".", 1)[0])
                all_points = []

                for alignment in alignments: # <- multiple alignments of same object in scene

                    if alignment[-1] != img:
                        continue
                    
                    vertices,faces = get_verts_and_faces_from_alignment(alignment,catid_to_cat_name,shape_net_dir)
                    mesh = Meshes(verts=[torch.Tensor(vertices).to(device)],faces=[torch.Tensor(faces).to(device)])
                    sampled_points = sample_points_from_meshes(mesh,num_samples=10000)
                    all_points.append(sampled_points[0].cpu().numpy())

                file_name = output_dir + '/' + id_scan + '-' + img.split('/')[2].split('.')[0] + '.ply'
                vertices = np.concatenate(all_points)
                vertex_colors = vertices * 0 + color
                writePlyFile(file_name,vertices,vertex_colors)



       


if __name__ == "__main__":

    # exp = sys.argv[1]
    # shape_net_dir = '/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/model'
    print('ENABLE ALL SCENES VIS')

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    exp = global_config["general"]["target_folder"]
    shape_net_dir = global_config["dataset"]["dir_path"] + 'model'

    # RGB
    color = np.array([0,0,255])

    # for add_on in ['','_filtered']:
    #     projectdir = exp + '/global_stats/eval_scannet/results_per_scene' + add_on
    #     output_dir = projectdir + '_visualised'
    #     visualise(projectdir,output_dir,shape_net_dir,color)

    for add_on in ['','_filtered']:
        projectdir = exp + '/global_stats/eval_scannet/results_per_scene' + add_on
        output_dir = exp + '/global_stats/eval_scannet/results_per_frame' + add_on + '_visualised'
        visualise_single_images(projectdir,output_dir,shape_net_dir,color)