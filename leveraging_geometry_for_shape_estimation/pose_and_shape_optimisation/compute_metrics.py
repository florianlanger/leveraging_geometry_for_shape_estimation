
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

from metrics import compare_meshes
from pose_selection import stretch_3d_coordinates


def convert_metrics_to_json(shape_metrics):

    new_dict = {}
    for metric in shape_metrics:
        new_dict[metric] = shape_metrics[metric][0].item()

    new_dict['F1'] = shape_metrics['F1@0.300000'][0].item()

    return new_dict

def F1_from_prediction(R_gt,T_gt,gt_obj,predicted_obj,R_total,T_total,device,pose_path,visualisations,wc_predicted_depth,scaling_factor):

    # load and position ground truth object
    bs = R_total.shape[0]

    gt_vertices_origin,gt_faces,gt_properties = gt_obj
    R_gt = torch.Tensor(R_gt).to(device) #.inverse().to(device)
    T_gt = torch.Tensor(T_gt).to(device)
    gt_vertices = torch.transpose(torch.matmul(R_gt,torch.transpose(gt_vertices_origin,0,1)),0,1) + T_gt
    textures_gt = Textures(verts_rgb=torch.ones((1,gt_vertices.shape[0],3),device=device))
    gt_mesh = Meshes(verts=[gt_vertices], faces=[gt_faces[0]],textures=textures_gt)
    
    # load and position predicted object
    pred_vertices_origin,pred_faces,pred_properties = predicted_obj
    pred_vertices = pred_vertices_origin.repeat(bs,1,1)
    pred_faces = pred_faces[0].repeat(bs,1,1)

    # pred_vertices_no_z = torch.transpose(torch.matmul(R_total,torch.transpose(pred_vertices_origin,0,1)),0,1) + T_total
    # textures_predicted = Textures(verts_rgb=torch.ones((1,pred_vertices_no_z.shape[0],3),device=device))
    # pred_mesh_no_z = Meshes(verts=[pred_vertices_no_z], faces=[pred_faces[0]], textures=textures_predicted)

    # T_predicted_gt_z = torch.cat((T_total[:2],T_gt[2:]))
    pred_vertices_no_z = torch.transpose(torch.matmul(R_total,torch.transpose(pred_vertices,-1,-2)),-1,-2) + T_total
    textures_predicted = Textures(verts_rgb=torch.ones((bs,pred_vertices_no_z.shape[0],3),device=device))
    pred_mesh_no_z = Meshes(verts=pred_vertices_no_z, faces=pred_faces, textures=textures_predicted)

    meshes_path = None
    # visualisations = False
    if visualisations:
        meshes_path = pose_path + '/meshes'
        os.mkdir(meshes_path)
        save_obj(meshes_path + '/predicted_before_transform.obj',pred_vertices_origin,pred_faces[0])
        save_obj(meshes_path + '/predicted.obj',pred_vertices_no_z[0],pred_faces[0])
        save_obj(meshes_path + '/gt.obj',gt_vertices,gt_faces[0])
        save_obj(meshes_path + '/gt_scaled.obj',gt_vertices*scaling_factor,gt_faces[0])
        # print('saving ply')
        # save_ply(meshes_path + '/predicted_depth.ply',wc_predicted_depth.reshape(-1,3).cpu())
        # print('saved ply')

    shape_metrics_no_z = compare_meshes(pred_mesh_no_z, gt_mesh, meshes_path,visualisations, reduce=False, thresholds=[0.3,0.5,0.7])



    pred_vertices_gt_pose = torch.transpose(torch.matmul(R_gt,torch.transpose(pred_vertices_origin,0,1)),0,1) + T_gt
    # pred_vertices_gt_z = torch.transpose(torch.matmul(R_total,torch.transpose(pred_vertices_origin,0,1)),0,1) + T_predicted_gt_z
    textures_predicted = Textures(verts_rgb=torch.ones((1,pred_vertices_no_z.shape[1],3),device=device))


    pred_mesh_no_z = Meshes(verts=[pred_vertices_no_z[0]], faces=[pred_faces[0]], textures=textures_predicted)

    pred_mesh_gt_pose = Meshes(verts=[pred_vertices_gt_pose], faces=[pred_faces[0]], textures=textures_predicted)



    # shape_metrics_gt_z = compare_meshes(pred_mesh_gt_z, gt_mesh, reduce=False, thresholds=[0.3])

    # return pred_mesh_no_z, pred_mesh_gt_z, gt_mesh, shape_metrics_no_z, shape_metrics_gt_z

    return pred_mesh_no_z, pred_mesh_gt_pose, shape_metrics_no_z


def F1_from_prediction_shape(R_gt,T_gt,gt_obj,predicted_obj,R_total,T_total,predicted_stretching,planes,device,pose_path,visualisations,wc_predicted_depth,scaling_factor):

    # load and position ground truth object
    bs = R_total.shape[0]

    gt_vertices_origin,gt_faces,gt_properties = gt_obj
    R_gt = torch.Tensor(R_gt).to(device) #.inverse().to(device)
    T_gt = torch.Tensor(T_gt).to(device)
    gt_vertices = torch.transpose(torch.matmul(R_gt,torch.transpose(gt_vertices_origin,0,1)),0,1) + T_gt
    textures_gt = Textures(verts_rgb=torch.ones((1,gt_vertices.shape[0],3),device=device))
    gt_mesh = Meshes(verts=[gt_vertices], faces=[gt_faces[0]],textures=textures_gt)
    
    # load and position predicted object
    pred_vertices_origin,pred_faces,pred_properties = predicted_obj
    pred_vertices = pred_vertices_origin.repeat(bs,1,1)
    pred_faces = pred_faces[0].repeat(bs,1,1)



    pred_vertices = stretch_3d_coordinates(pred_vertices,planes,predicted_stretching)


    pred_vertices_no_z = torch.transpose(torch.matmul(R_total,torch.transpose(pred_vertices,-1,-2)),-1,-2) + T_total
    # pred_vertices_no_z = pred_vertices
    textures_predicted = Textures(verts_rgb=torch.ones((bs,pred_vertices_no_z.shape[0],3),device=device))
    pred_mesh_no_z = Meshes(verts=pred_vertices_no_z, faces=pred_faces, textures=textures_predicted)

    meshes_path = None
    if visualisations:
        meshes_path = pose_path + '/meshes'
        os.mkdir(meshes_path)
        save_obj(meshes_path + '/predicted_before_transform.obj',pred_vertices_origin,pred_faces[0])
        save_obj(meshes_path + '/predicted.obj',pred_vertices_no_z[0],pred_faces[0])
        save_obj(meshes_path + '/gt.obj',gt_vertices,gt_faces[0])
        save_obj(meshes_path + '/gt_scaled.obj',gt_vertices*scaling_factor,gt_faces[0])
        # print('saving ply')
        # save_ply(meshes_path + '/predicted_depth.ply',wc_predicted_depth.reshape(-1,3).cpu())
        # print('saved ply')
        # visualisations = False
        

    shape_metrics_no_z = compare_meshes(pred_mesh_no_z, gt_mesh, meshes_path,visualisations, reduce=False, thresholds=[0.3,0.4,0.5,0.7])

    pred_vertices_gt_pose = torch.transpose(torch.matmul(R_gt,torch.transpose(pred_vertices_origin,0,1)),0,1) + T_gt
    # pred_vertices_gt_z = torch.transpose(torch.matmul(R_total,torch.transpose(pred_vertices_origin,0,1)),0,1) + T_predicted_gt_z
    textures_predicted = Textures(verts_rgb=torch.ones((1,pred_vertices_no_z.shape[1],3),device=device))


    pred_mesh_no_z = Meshes(verts=[pred_vertices_no_z[0]], faces=[pred_faces[0]], textures=textures_predicted)

    pred_mesh_gt_pose = Meshes(verts=[pred_vertices_gt_pose], faces=[pred_faces[0]], textures=textures_predicted)


    return pred_mesh_no_z, pred_mesh_gt_pose, shape_metrics_no_z



def get_score_for_folder(global_config):

    target_folder = global_config["general"]["target_folder"]
    image_folder = global_config["general"]["image_folder"]
    models_folder_read = global_config["general"]["models_folder_read"]
    top_n_retrieval = global_config["keypoints"]["matching"]["top_n_retrieval"]


    pose_config = global_config["pose_and_shape"]["pose"]

    device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))
    torch.cuda.set_device(device)

    for name in tqdm(os.listdir(target_folder + '/cropped_and_masked')):

        # if not 'desk_0526_3' in name:
        #     print('no ')
        #     continue
        
        with open(target_folder + '/nn_infos/' + name.split('.')[0] + '.json','r') as f:
            retrieval_list = json.load(f)["nearest_neighbours"]

        with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
            gt_infos = json.load(f)

        for i in range(top_n_retrieval):
            
            out_path = target_folder + '/metrics/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json'
            # if os.path.exists(out_path):
            #     continue

            with open(target_folder + '/poses/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json','r') as f:
                pose_info = json.load(f)

            # predicted obj
            model_path_pred = models_folder_read + "/models/remeshed/" + retrieval_list[i]["model"].replace('model/','')
            R_pred = torch.Tensor(pose_info["predicted_r"]).to(device).unsqueeze(0)
            T_pred = torch.Tensor(pose_info["predicted_t"]).to(device).unsqueeze(0)
            predicted_obj = load_obj(model_path_pred,device=device,create_texture_atlas=False, load_textures=False)
            

            # gt obj
            # model_path_gt = models_folder_read + "/models/remeshed/" + gt_infos["model"].replace('model/','')
            model_path_gt = global_config["dataset"]["pix3d_path"] + gt_infos["model"]
            R_gt = gt_infos["rot_mat"]
            T_gt = gt_infos["trans_mat"]
            gt_obj = load_obj(model_path_gt,device=device,create_texture_atlas=False, load_textures=False)


            if global_config["pose_and_shape"]["shape"]["optimise_shape"] == "False":
                _,_,shape_metrics = F1_from_prediction(R_gt,T_gt,gt_obj,predicted_obj,R_pred,T_pred,device,None,False,None,None)
            elif global_config["pose_and_shape"]["shape"]["optimise_shape"] == "True":
                print('Unsqueeze ?')
                predicted_stretching = torch.Tensor(pose_info["predicted_stretching"]).to(device).unsqueeze(0).to(device)
                planes = torch.Tensor(global_config["pose_and_shape"]["shape"]["planes"]).to(device)
                _,_,shape_metrics = F1_from_prediction_shape(R_gt,T_gt,gt_obj,predicted_obj,R_pred,T_pred,predicted_stretching,planes,device,None,False,None,None)
        

            shape_metrics = convert_metrics_to_json(shape_metrics)

            with open(out_path,'w') as f:
                json.dump(shape_metrics,f,indent=4)





def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)


    get_score_for_folder(global_config)

if __name__ == '__main__':
    main()