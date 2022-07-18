from cgi import test
import csv
import json
from unicodedata import category
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from leveraging_geometry_for_shape_estimation.data_conversion_scannet.get_infos import get_scene_pose,get_scene_pose_2
import quaternion
import SE3
import CSVHelper
from operator import itemgetter
import os
from glob import glob
from socket import INADDR_ALLHOSTS_GROUP
import cv2
import json
import os
import scipy.ndimage
import numpy as np
from scipy.spatial.transform import Rotation as scipy_rot
from tqdm import tqdm

from pytorch3d.ops import sample_points_from_meshes

from leveraging_geometry_for_shape_estimation.data_conversion_scannet.get_infos import make_M_from_tqs
from leveraging_geometry_for_shape_estimation.vis_pose.vis_pose import load_mesh,render_mesh,overlay_rendered_image

def load_4by4_from_txt(path):
    M = np.zeros((4,4))
    with open(path,'r') as f:
        content = f.readlines()
        for i in range(4):
            line = content[i].split()
            for j in range(4):
                M[i,j] = np.float32(line[j])
        return M

def get_focal_length(path_intriniscs_color,w):
    K = load_4by4_from_txt(path_intriniscs_color)
    focal_length = 2*K[0,0]/w

    sw = w / K[0,0]
    return focal_length,K

def writePlyFile(file_name, vertices, colors):
    ply_header = '''ply
                format ascii 1.0
                element vertex %(vert_num)d
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                end_header
               '''

    vertices = vertices.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices, colors])
    with open(file_name, 'w') as f:
      f.write(ply_header % dict(vert_num=len(vertices)))
      np.savetxt(f, vertices, '%f %f %f %d %d %d')

def save_points_from_model(model_path,R,T,S,out_path_ply,color = (0,0,255)):
    mesh = load_mesh(model_path,R,T,S,device=torch.device("cpu"))
    points = sample_points_from_meshes(mesh)
    points = points[0]
    writePlyFile(out_path_ply, points, colors=np.array(points)*0 + np.array(color))

def category_to_id(category):
    top = {}
    top["03211117"] = "display"
    top["04379243"] = "table"
    top["02808440"] = "bathtub"
    top["02747177"] = "bin"
    top["04256520"] = "sofa"
    top["03001627"] = "chair"
    top["02933112"] = "cabinet"
    top["02871439"] = "bookcase"
    top["02818832"] = "bed"

    inv_map = {v: k for k, v in top.items()}
    return inv_map[category]

def convert_to_world_frame(T,R,s,R_scene_pose,T_scene_pose,all_data,id_scan):
    idscan2trs = {}
    for r in all_data:
        id_scan_1 = r["id_scan"]
        idscan2trs[id_scan_1] = r["trs"]
    invert = np.array([[-1,0,0],[0,-1,0],[0,0,1.]])
    T = np.matmul(invert,T)
    t = np.matmul(np.linalg.inv(R_scene_pose),T - T_scene_pose)

    print('t equivalent to tcad',t)

    R = np.matmul(invert,R)
    R = np.matmul(np.linalg.inv(R_scene_pose),R)
    q = quaternion.from_rotation_matrix(R)
    q = quaternion.as_float_array(q)

    Mscan = SE3.compose_mat4(idscan2trs[id_scan]["translation"], idscan2trs[id_scan]["rotation"], idscan2trs[id_scan]["scale"])
    Mcad = SE3.compose_mat4(t, q, s)#, -np.array(model_gt["center"]))    
    t_gt, q_gt, s_gt = SE3.decompose_mat4(np.dot(np.linalg.inv(Mscan), Mcad))
    R_gt = quaternion.as_rotation_matrix(q_gt)
    return t_gt,q_gt,R_gt,s_gt



def render_gt_frame(scene_info,frame,dir_path,catid_to_cat,cats,cats_renamed,device,scene,sw,id_object,all_data):
    model = scene_info['aligned_models'][id_object]
    
    frame_4by4_path = dir_path + scene_info['id_scan'] + '/pose/' + frame + '.txt'
    scene_trs = scene_info["trs"]
    R_scene_pose,T_scene_pose = get_scene_pose(frame_4by4_path,scene_trs)

    t = model["trs"]["translation"]
    q = model["trs"]["rotation"]
    s = model["trs"]["scale"]
    # print('GT t', t, 'Gt q', q, 'GT s',s)
    Mcad,Rcad,Scad = make_M_from_tqs(t, q, s)
    R = np.matmul(R_scene_pose,Rcad)
    T = np.matmul(R_scene_pose,np.array(t)) + T_scene_pose
    invert = np.array([[-1,0,0],[0,-1,0],[0,0,1.]])
    R = np.matmul(invert,R)
    T = np.matmul(invert,T)
    # print(model['catid_cad'])
    # print(model['id_cad'])
    old_category = catid_to_cat[model['catid_cad']]
    index = cats.index(old_category)
    category = cats_renamed[index]
    model_path = '/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/model/{}/{}/model_normalized.obj'.format(category,model['id_cad'])
    mesh = load_mesh(model_path,R,T,s,device)
    # print(model_path)

    original_image = cv2.imread(dir_path + scene + '/color/' + frame + '.jpg')
    h,w,_ = original_image.shape
    f,_ = get_focal_length(dir_path + scene_info['id_scan'] + '/intrinsics_color.txt',w)
    rendered_image = render_mesh(w,h,f,mesh,device,sw,flip=False)
    out_img = overlay_rendered_image(original_image,rendered_image)
    cv2.imwrite('/scratch2/fml35/results/debug_single_frame_to_raw/' + scene + '_' + frame + '.png',out_img)

    t_world,q_world,R_world,s_world = convert_to_world_frame(T,R,s,R_scene_pose,T_scene_pose,all_data,scene)

    model_path = '/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/model/{}/{}/model_normalized.obj'.format('table',model["id_cad"])
    out_path_ply = '/scratch2/fml35/results/debug_single_frame_to_raw/' + scene + '.ply'
    save_points_from_model(model_path,R_world,t_world,s_world,out_path_ply)


# def visualise_raw():

def render_pred_frame(scene_info,frame,dir_path,device,scene,sw,single_frame,R_scene_pose,T_scene_pose,all_data,df):

    infos = single_frame[scene + '/color/' + frame + '.jpg']
    for i in range(len(infos)):
        category = infos[i]['category']
        id_cad = infos[i]["scene_cad_id"][1]
        if category != 'table':
            continue
        model_path = '/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/model/{}/{}/model_normalized.obj'.format(category,id_cad)
        invert = np.array([[-1,0,0],[0,-1,0],[0,0,1.]])
        q = infos[i]["q"]
        q = [q[1],q[2],q[3],q[0]]
        R = scipy_rot.from_quat(q).as_matrix()
        R = np.matmul(invert,R)
        T = np.matmul(invert,np.array(infos[i]["t"]))

        mesh = load_mesh(model_path,R,T,infos[i]["s"],device)

        original_image = cv2.imread(dir_path + scene + '/color/' + frame + '.jpg')
        h,w,_ = original_image.shape
        f,_ = get_focal_length(dir_path + scene_info['id_scan'] + '/intrinsics_color.txt',w)
        rendered_image = render_mesh(w,h,f,mesh,device,sw,flip=False)
        out_img = overlay_rendered_image(original_image,rendered_image)
        cv2.imwrite('/scratch2/fml35/results/debug_single_frame_to_raw/' + scene + '_' + frame + 'pred_{}.png'.format(i),out_img)

        # CONVERT single frame prediction to worldframe and save
        t_world,q_world,R_world,s_world = convert_to_world_frame(T,R,infos[i]["s"],R_scene_pose,T_scene_pose,all_data,scene)
        print('score',infos[i]["score"])
        print('t transformed',t_world)
        out_path_ply = '/scratch2/fml35/results/debug_single_frame_to_raw/' + scene + '_' + frame + '.ply'
        save_points_from_model(model_path,R_world,t_world,s_world,out_path_ply)

        # VISUALISE raw frame prediction
        scores_raw = list(df["object_score"])
        index_raw = scores_raw.index(str(infos[i]["score"]))
        t = [df.iloc[index_raw]['tx'],df.iloc[index_raw]['ty'],df.iloc[index_raw]['tz']]
        t = [float(x) for x in t]
        q = [df.iloc[index_raw]['qw'],df.iloc[index_raw]['qx'],df.iloc[index_raw]['qy'],df.iloc[index_raw]['qz']]
        s = [df.iloc[index_raw]['sx'],df.iloc[index_raw]['sy'],df.iloc[index_raw]['sz']]
        s = [float(x) for x in s]
        q = np.quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        R = quaternion.as_rotation_matrix(q)
        print('t raw',t)
        print('___')

        out_path_ply = '/scratch2/fml35/results/debug_single_frame_to_raw/' + scene + '_' + frame + 'raw.ply'
        save_points_from_model(model_path,R,t,s,out_path_ply)


def convert_single_frame(scene,frame,dir_path,scene_info):

    frame_4by4_path = dir_path + scene + '/pose/' + frame + '.txt'
    scene_trs = scene_info["trs"]
    R_scene_pose,T_scene_pose = get_scene_pose(frame_4by4_path,scene_trs)

    invert = np.array([[-1,0,0],[0,-1,0],[0,0,1.]])

    # Get T 
    # T = np.matmul(invert,t_start)
    t_start = detection['t']
    T = np.array(t_start)
    t_cad = np.matmul(np.linalg.inv(R_scene_pose),T - T_scene_pose)


    # GET R
    q = np.quaternion(detection['q'][0], detection['q'][1], detection['q'][2], detection['q'][3])
    R = quaternion.as_rotation_matrix(q)
    # R = np.matmul(invert,R)
    R = np.matmul(np.linalg.inv(R_scene_pose),R)
    q = quaternion.from_rotation_matrix(R)
    r_cad = quaternion.as_float_array(q)
    # IDEA go from predictions back to kind of like T,R,S of CAD model, thans apply same transformation as in EvalBenchmark script
    # scan_tran = -np.array(idscan2trs[id_scan]["translation"])
    Mscan = SE3.compose_mat4(idscan2trs[id_scan]["translation"], idscan2trs[id_scan]["rotation"], idscan2trs[id_scan]["scale"])
    Mcad = SE3.compose_mat4(t_cad, r_cad, detection['s'])#,-np.array(object_center))

    t_pred, q_pred, s_pred = SE3.decompose_mat4(np.dot(np.linalg.inv(Mscan), Mcad))
    q_pred = quaternion.as_float_array(q_pred)


def get_gt_3d(all_data,id_scan,id_object,device):
    idscan2trs = {}
    
    for r in all_data:
        id_scan_1 = r["id_scan"]
        idscan2trs[id_scan_1] = r["trs"]

    for r in all_data:
        test_id_scan = r["id_scan"]
        if test_id_scan == id_scan:
            models = r["aligned_models"]
    # print(models)
    # for model in models:
    
    model = models[id_object]
    assert model["id_cad"] == '24942a3b98d1bcb6a570c6c691c987a8'
    Mscan = SE3.compose_mat4(idscan2trs[id_scan]["translation"], idscan2trs[id_scan]["rotation"], idscan2trs[id_scan]["scale"])
    Mcad = SE3.compose_mat4(model["trs"]["translation"], model["trs"]["rotation"], model["trs"]["scale"])#, -np.array(model_gt["center"]))    
    t_gt, q_gt, s_gt = SE3.decompose_mat4(np.dot(np.linalg.inv(Mscan), Mcad))
    
    # q = np.quaternion(detection['q'][0], detection['q'][1], detection['q'][2], detection['q'][3])
    # R_gt = quaternion.as_rotation_matrix(q_gt)
    # model_path = '/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/model/{}/{}/model_normalized.obj'.format('table',model["id_cad"])
    # mesh = load_mesh(model_path,R_gt,t_gt,s_gt,device=torch.device("cpu"))
    # points = sample_points_from_meshes(mesh)
    # points = points[0]
    # writePlyFile('/scratch2/fml35/results/debug_single_frame_to_raw/' + id_scan + '.ply', points, colors=np.array(points)*0 + np.array([0,0,255]))
      

def main():
    with open('/scratch2/fml35/datasets/scannet/scan2cad_annotations/full_annotations.json') as json_file:
            all_data = json.load(json_file)

    df = pd.read_csv('/scratch2/fml35/results/ROCA/raw_results.csv',dtype=str)

    with open('/scratch2/fml35/results/ROCA/per_frame_best.json') as file:
        single_frame = json.load(file)

    with open('/scratch2/fml35/datasets/shapenet_v2/ShapeNetCore.v2/taxonomy.json') as file:
        shapenet_taxonomy = json.load(file)

    sw = 2
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    catid_to_cat = {}
    for item in shapenet_taxonomy:
        catid_to_cat[item["synsetId"]] = item["name"]

    dir_path = '/scratch2/fml35/datasets/scannet/scannet_frames_25k/'
    dir_path_own_data = '/scratch2/fml35/datasets/scannet/scannet_frames_25k_own_data/'

    cats = ["bathtub,bathing tub,bath,tub","bed","ashcan,trash can,garbage can,wastebin,ash bin,ash-bin,ashbin,dustbin,trash barrel,trash bin","bookshelf","cabinet","chair","display,video display","sofa,couch,lounge","table"]
    cats_renamed = ["bathtub","bed","bin","bookshelf","cabinet","chair","display","sofa","table"]

    for scene_info in tqdm(all_data):
        scene = scene_info["id_scan"]

        for frame in os.listdir(dir_path_own_data + scene + '/masks/'):
            # print(frame)
            # if '0653_00' not in scene or ('001300' not in frame and '001200' not in frame and '001400' not in frame):
            if '0653_00' not in scene or ('001300' not in frame):
                continue


            for object in os.listdir(dir_path_own_data + scene + '/masks/' + frame):
                id_object = int(object.split('.')[0])
                if id_object != 17:
                    continue
                get_gt_3d(all_data,scene,id_object,device)
                render_gt_frame(scene_info,frame,dir_path,catid_to_cat,cats,cats_renamed,device,scene,sw,id_object,all_data)

            frame_4by4_path = dir_path + scene_info['id_scan'] + '/pose/' + frame + '.txt'
            scene_trs = scene_info["trs"]
            R_scene_pose,T_scene_pose = get_scene_pose(frame_4by4_path,scene_trs)
            render_pred_frame(scene_info,frame,dir_path,device,scene,sw,single_frame,R_scene_pose,T_scene_pose,all_data,df)
        
              
if __name__ == '__main__':
    main() 