import os
from selectors import EpollSelector
import numpy as np
import sys
import socket
import pickle
import math
import os
import json
from tqdm import tqdm
import torch
import k3d
from shutil import copytree
from time import time

from pytorch3d.io import load_obj,save_ply
from pytorch3d.ops import sample_points_from_meshes,knn_points
from pytorch3d.structures import Pointclouds,Meshes

def make_folder_check(path,path2=None):
    if not os.path.exists(path):
        os.mkdir(path)
    
    if path2 != None:
        if not os.path.exists(path2):
            os.mkdir(path2)


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def find_normals(verts,faces):

    normals = 3*np.ones((faces.shape[0],3))


    for i in range(faces.shape[0]):
        v1,v2,v3 = verts[faces[i,0]],verts[faces[i,1]],verts[faces[i,2]]

        normal = normal_plane(v1,v2,v3)
        # print(normal)
        normalised = normal / np.sum(normal**2)**0.5

        normals[i] = normalised

    return normals


def find_edges(verts,faces,face_normals,vertex_id_to_face_id,mask_vertex_id_to_face_id,angle_threshold):
    # print(face_normals)
    # print(face_normals.shape)
    # print(df)
    line_segs = np.zeros((faces.shape[0]*3,6))

    normals_1 = torch.zeros((faces.shape[0]*3,3))
    normals_2 = torch.zeros((faces.shape[0]*3,3))
    all_indices =  torch.zeros((faces.shape[0]*3,2),dtype=int)

    counter = 0
    checked_pairs = []
    t0 = time()

    # pairs = torch.Tensor([[0,1],[1,2],[2,0]]).long().repeat(faces.shape[0],1)
    # print('pairs',pairs.shape)
    # faces_repeated = faces.repeat_interleave(3,dim=0)
    # print('faces',faces_repeated.shape)
    # print(faces_repeated[:2])
    # all_indices_1 = faces_repeated[torch.arange(faces_repeated.shape[0]),pairs[:,0]]
    # all_indices_2 = faces_repeated[torch.arange(faces_repeated.shape[0]),pairs[:,1]]

    # print(vertex_id_to_face_id.shape)
    # face_ids_pt_1 = vertex_id_to_face_id[all_indices_1]
    # face_ids_pt_2 = vertex_id_to_face_id[all_indices_2]

    # for i in range(vertex_id_to_face_id.shape[0]):
    #     mask = face_ids_pt_1 =


    # print(face_ids_pt_1.shape)
    # print(dfd)
    # print(faces.shape)

    for i in tqdm(range(faces.shape[0])):
        for pair in [[0,1],[1,2],[2,0]]:
            indices = [faces[i,pair[0]],faces[i,pair[1]]]
            if indices not in checked_pairs and indices[::-1] not in checked_pairs:

                start = time()

                face_ids_pt_1 = vertex_id_to_face_id[indices[0]][mask_vertex_id_to_face_id[indices[0]]]
                face_ids_pt_2 = vertex_id_to_face_id[indices[1]][mask_vertex_id_to_face_id[indices[1]]]
                
                faces_id_shared = intersection(face_ids_pt_1,face_ids_pt_2)
                time_1 = time()
                # print('intersection', time_1 - start)
                # print(faces_id_shared)
                if len(faces_id_shared) > 2:
                    n_1 = torch.Tensor([0,0,1])
                    n_2 = torch.Tensor([0,1,0])
                    continue

                elif len(faces_id_shared) == 2:
                    n_1 = face_normals[faces_id_shared[0]]
                    n_2 = face_normals[faces_id_shared[1]]


                elif len(faces_id_shared) == 1:
                    # so that they definitely will be kept
                    n_1 = torch.Tensor([0,0,1])
                    n_2 = torch.Tensor([0,1,0])
                # assert len(faces_id_shared) == 2


                if torch.isnan(n_1).any() or torch.isnan(n_2).any():
                    continue

                normals_1[counter] = n_1
                normals_2[counter] = n_2
                all_indices[counter] = torch.Tensor(indices)
        
                # time_2 = time()
                # print('checks',time_2 - time_1)

                # print(n_1,n_2)
                # normal_dot = torch.sum(n_1*n_2,dim=0)
                # angle = torch.arccos(normal_dot) * 180 / np.pi
                # angle = torch.min(angle,torch.abs(180. - angle))
                # normal_dot = torch.sum(n_1*n_2,dim=0).item()
                # angle = np.arccos(normal_dot) * 180 / np.pi
                # angle = np.min([angle,np.abs(180. - angle)])

                # time_3 = time()
                # print('angle', time_3 - time_2)

                # print(angle)
                # print()
                # print(angle)
                # print(n_1,n_2)

                # if angle > angle_threshold:
                #     # print('True')
                #     line_segs[counter] = np.concatenate((verts[indices[0]],verts[indices[1]]))

                counter += 1

                checked_pairs.append(indices)

                # time_4 = time()
                # print('end',time_4 - time_3)
                # else:
                #     print('False')
                
                # print('--------------')

                # if i == 10:
                #     print(gdgd)
    t4 = time()
    # print()
    # print(counter)
    normals_1 = normals_1[:counter]
    normals_2 = normals_2[:counter]
    indices = all_indices[:counter]

    normal_dot = torch.sum(normals_1*normals_2,dim=1)
    angle = torch.arccos(normal_dot) * 180 / np.pi
    # print('angle',angle.shape)
    angle_and_opposite = torch.cat([angle.unsqueeze(1),torch.abs(180. - angle).unsqueeze(1)],dim=1)
    # print('ange',angle_and_opposite.shape)
    angle,_ = torch.min(angle_and_opposite,dim=1)
    mask = angle > angle_threshold
    # print('mas',mask.shape)
    # print('verts[indices[:,0][mask]',verts[indices[:,0][mask]].shape)
    line_segs = torch.cat([verts[indices[:,0][mask]],verts[indices[:,1][mask]]],dim=1)
    # print(time()-t4)

    return line_segs.numpy()

def sample_points_from_lines(lines,points_per_line):
    n_lines = lines.shape[0]
    lines = torch.repeat_interleave(lines,points_per_line,dim=0)
    interval = torch.linspace(0,1,points_per_line).repeat(n_lines)
    interval = interval.unsqueeze(1).repeat(1,3)
    points = lines[:,:3] + (lines[:,3:6]-lines[:,:3]) * interval
    return points

def normal_plane(point_1,point_2,point_3):
     
    x1, y1, z1 = point_1
    x2, y2, z2 = point_2
    x3, y3, z3 = point_3
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return np.array([a,b,c])

def main():

    angle_threshold_degree = 20
    points_per_line_vis = 30
    # dont use min dist line
    min_dist_line = 0.05
    pix_path = '/scratch/fml35/datasets/pix3d_new/'
    target_folder = '/scratch/fml35/datasets/pix3d_new/own_data/rendered_models/3d_lines/exp_26_points_on_edges_angle_20_lines_one_and_three_face'

    with open("/data/cornucopia/fml35/experiments/exp_024_debug/models/model_list.json",'r') as f:
            model_list = json.load(f)["models"]

    make_folder_check(target_folder)
    make_folder_check(target_folder + '/edge_points')
    make_folder_check(target_folder + '/lines')
    # copytree('/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/models/extract_lines_from_models',target_folder + '/code')

    list_vertices = []
    for j in tqdm(range(0,len(model_list))):
        if 'SS_' in model_list[j]["model"]:
            vert_number = 100000000
        else:
            verts_torch,faces_torch,_ = load_obj(pix_path + model_list[j]["model"],load_textures=False)
            vert_number = verts_torch.shape[0]
        list_vertices.append(vert_number)

    indices = np.argsort(list_vertices)
    for j in tqdm(indices):
    # for j in tqdm(range(0,len(model_list))):
        if 'SS_' in model_list[j]["model"]:
            continue

        # if not 'IKEA_EKTORP' in model_list[j]["model"]:
        #     continue

        verts_torch,faces_torch,_ = load_obj(pix_path + model_list[j]["model"],load_textures=False)
        print(model_list[j]["model"], verts_torch.shape[0])
        # continue
        faces_idx = faces_torch[0]
        # print('finding normals')
        normals = find_normals(verts_torch,faces_idx)
        # print('finding_edges')
        # whic facc
        vertex_id_to_face_id_list = []
        for i in range(verts_torch.shape[0]):
            vertex_id_to_face_id_list.append([])

        for i in range(faces_idx.shape[0]):
            for k in range(3):
                vertex_id_to_face_id_list[faces_idx[i,k]].append(i)

        
        vertex_id_to_face_id = -1 * torch.ones((verts_torch.shape[0],100),dtype=int)
        counter = torch.zeros((verts_torch.shape[0]),dtype=int)

        for i in range(faces_idx.shape[0]):
            for k in range(3):
                vertex_id = faces_idx[i,k]
                vertex_id_to_face_id[vertex_id,counter[vertex_id]] = i
                counter[vertex_id] += 1

        vertex_id_to_face_id = vertex_id_to_face_id[:,:torch.max(counter)]
        mask_vertex_id_to_face_id = vertex_id_to_face_id > -1

  


        lines = find_edges(verts_torch,faces_idx,torch.Tensor(normals),vertex_id_to_face_id,mask_vertex_id_to_face_id,angle_threshold_degree)
        # mask = np.sum((lines[:,:3] - lines[:,3:6])**2,axis=1)**0.5 > min_dist_line
        # lines = lines[mask]
        points = sample_points_from_lines(torch.Tensor(lines),points_per_line_vis)
        save_ply(target_folder + '/edge_points/' + model_list[j]["category"] + '_' + model_list[j]["model"].split('/')[2] + '.ply',points)
        np.save(target_folder + '/lines/' + model_list[j]["category"] + '_' + model_list[j]["model"].split('/')[2] + '.npy',lines)


if __name__ == '__main__':
    main()