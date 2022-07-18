from yaml import scan
import imageio
import numpy as np
import cv2
import json
import torch
from scipy.spatial.transform import Rotation as scipy_rot
import sys
from torchvision.models import vgg16
from PIL import Image
from matplotlib import pyplot as plt
import itertools
import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
from PIL import Image
import trimesh
import imagesize

import numpy as np

from tqdm import tqdm

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj
import pytorch3d

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PerspectiveCameras,
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    TexturesVertex,
    Textures
)
from pytorch3d.renderer.blending import BlendParams,hard_rgb_blend
# add path for demo utils functions 
import sys
import os

from PIL import Image

from leveraging_geometry_for_shape_estimation.models.remesh_models import remesh


class myShader(torch.nn.Module):

    def __init__(
        self, device="cpu", cameras=None, blend_params=None
    ):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of TexturedSoftPhongShader"
            raise ValueError(msg)
        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)

        return images

def make_M_from_tqs(t, q, s,center=None):
    # q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    # R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    q = [q[1],q[2],q[3],q[0]]
    R[0:3, 0:3] = scipy_rot.from_quat(q).as_matrix()
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    C = np.eye(4)
    if center is not None:
        print('CENTER is not working !!! NO ONLY APPLIES TO BBOXES' )
        assert center == None
        C[0:3, 3] = center

    M = T.dot(R).dot(S).dot(C)
    return M,T,R,S


def load_4by4_from_txt(path):
    M = np.zeros((4,4))
    with open(path,'r') as f:
        content = f.readlines()
        for i in range(4):
            line = content[i].split()
            for j in range(4):
                M[i,j] = np.float32(line[j])
        return M


def convert_K(K,width,height):
    K[0,2] = K[0,2] - width/2
    K[1,2] = K[1,2] - height/2
    K = K/(width/2)
    K[2:4,2:4] = torch.Tensor([[0,1],[1,0]])
    return K

def create_setup_calibration_matrix(scene,frame,dir_path,device,n_mesh):
    K = load_4by4_from_txt(dir_path + '/intrinsics_color.txt')

    # get size of image want to rerender, assume all horizontal
    width, height = imagesize.get(dir_path + '/color/{}.jpg'.format(frame))

    assert width >= height

    raster_settings = RasterizationSettings(image_size=width,blur_radius=0.0, faces_per_pixel=1)
    materials = Materials(device=device,specular_color=[[0.0, 0.0, 0.0]],shininess=1.0)

    K = convert_K(torch.Tensor(K),width,height)

    # load camera pose
    R_and_T = load_4by4_from_txt(dir_path + '/pose/' + frame + '.txt')
    R_pose = R_and_T[:3,:3].copy()
    T_pose = R_and_T[:3,3].copy()

    # transform camera pose according to scene transformation
    T_pose = np.concatenate((T_pose,np.ones((1))),axis=0)
    Mscene,_,_,_ = make_M_from_tqs(scene["trs"]['translation'],scene["trs"]['rotation'],scene["trs"]['scale'])
    T_scene_pose = np.matmul(Mscene,T_pose)[:3]
    R_scene_pose = np.matmul(Mscene[:3,:3],R_pose).copy()
    R_scene_pose = np.linalg.inv(R_scene_pose)

    # do this because of pytorch3d convention for specifying camera
    T_final_pose = - np.matmul(R_scene_pose,T_scene_pose)
    R_final_pose = np.linalg.inv(R_scene_pose)
    R_final_pose = R_scene_pose

    T = torch.Tensor(T_final_pose).unsqueeze(0).repeat(n_mesh,1)
    R = torch.Tensor(R_final_pose).unsqueeze(0).repeat(n_mesh,1,1)

    r_cam = torch.eye(3).unsqueeze(0).repeat(n_mesh,1,1)
    t_cam = torch.zeros((n_mesh,3))

    print('n_mesh',n_mesh)

    cameras = PerspectiveCameras(device=device,K = K.unsqueeze(0),T=t_cam,R=r_cam)
    
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer_textured = MeshRenderer(rasterizer=rasterizer,shader=myShader(device=device, cameras=cameras,blend_params = BlendParams(background_color = (0.0, 0.0, 0.0))))

    # return cameras,rasterizer,renderer_textured,width,height,R_scene_pose,T_scene_pose,focal_length
    return cameras,rasterizer,renderer_textured,width,height,R,T,None


def load_all_meshes_as_one(shapenet_path,scene_info,device,catid_to_detectron_id):
    shapenet_path = '/data/cvfs/fml35/derivative_datasets/shapenet_core_2/ShapeNetCore.v2/'

    all_vertices = []
    all_faces = []
    all_verts_rgb = []

    counter = 0

    list_objects = ['bed', 'sofa', 'chair', 'trash bin', 'cabinet', 'display', 'table', 'bookshelf','bathtub']
    for i in range(scene_info["n_aligned_models"]):
        model = scene_info['aligned_models'][i]
        if model["catid_cad"] in catid_to_detectron_id:
            # catid = scene_info['aligned_models'][i]['catid_cad']
            # category_id = catid_to_detectron_id[catid]
            # print(mode)
            model_path = '{}{}/{}/models/model_normalized.obj'.format(shapenet_path,model['catid_cad'],model['id_cad'])
            if not '03211117' in model_path:
                continue
            if i != 3:
                continue
            print(model_path)
            # max_edge_length = 0.15
            max_edge_length = 0.05
            vertices,faces,_ = remesh(model_path,max_edge_length,device)
            # vertices,faces,properties = load_obj('{}/{}/{}/models/model_normalized.obj'.format(shapenet_path,model['catid_cad'],model['id_cad']),device=device,create_texture_atlas=False, load_textures=False)

            # load CAD model transformation
            t = model["trs"]["translation"]
            q = model["trs"]["rotation"]
            s = model["trs"]["scale"]
            center = model["center"]
            # Mcad = make_M_from_tqs(t, q, s, - np.array(center))
            Mcad,T,R,S = make_M_from_tqs(t, q, s)

            T = T[:3,3]
            S = S[:3,:3]
            R = R[:3,:3]

            print('T',T)
            print('R',R)
            print('S',S)   

            # vertices = np.concatenate((np.array(vertices.cpu()),np.ones((vertices.shape[0],1))),axis=1)
            # vertices =  np.dot(Mcad,vertices.T).T
            # vertices_transformed = vertices[:,0:3]

            vertices_transformed = np.matmul(R,np.matmul(S,np.array(vertices.cpu()).T)).T + T
            

            color = 3*[(i+1)/ 255.]
            
            verts_rgb = np.tile(color,(vertices.shape[0],1))
        
            
            all_vertices.append(vertices_transformed[:,:])
            all_faces.append(faces + counter)
            all_verts_rgb.append(verts_rgb)

            counter += vertices.shape[0]
    # max_vertices = max([vertices.shape[0] for vertices in all_vertices])

    if all_vertices == []:
        return None
    
    else:
        vertices = torch.Tensor(np.concatenate(all_vertices,axis=0)).to(device)
        verts_rgb = torch.Tensor(np.concatenate(all_verts_rgb,axis=0)).unsqueeze(0).to(device)
        all_faces = torch.cat(all_faces,dim=0)
        # verts_rgb = torch.ones((scene_info["n_aligned_models"],max_vertices,3))
        textures = Textures(verts_rgb=verts_rgb)
        meshes = Meshes(verts=[vertices],faces=[all_faces],textures=textures)
        return meshes,vertices,all_faces,textures,T,R,S

def flip_and_crop(input,diff):
    output = input[:,::-1,::-1]
    output = output[:,int(diff/2):-int(diff/2),:]
    return output

def make_dir_check(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    
def get_color_image(image,n_models):
    colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[128,0,0],[0,128,0],[0,0,128],[128,128,0],[128,0,128],[0,128,128]]
    for i in range(1,n_models+1):
        mask = np.all(image == [i,i,i],axis=-1)
        image[mask] = image[mask]*0 + np.array(colors[i%len(colors)])
    return image


def save_separate_masks(image,n_models,output_dir,min_pixel_per_mask):
    for i in range(1,n_models+1):
        mask = np.all(image == [i,i,i],axis=-1)
        # if np.any(mask):
        if np.sum(mask) > min_pixel_per_mask:
        # image[mask] = image[mask]*0 + np.array(colors[i%len(colors)])
            cv2.imwrite(output_dir + '/' + str(i-1).zfill(3) + '.png',mask*255)

def rerender_single_scene(scene_info,input_dir,output_dir,device,shapenet_path,catid_to_detectron_id,min_pixel_per_mask):

    visualisation = True


    frames_dir = os.listdir(input_dir + '/color')
    frames = [frame.replace('.jpg','') for frame in frames_dir]
    path_rendered = output_dir + '/instance_rerendered'
    path_overlayed_rerendered = output_dir + '/overlayed_instance_rerendered'
    path_masks = output_dir + '/masks'
    
    make_dir_check(output_dir)
    make_dir_check(path_rendered)
    make_dir_check(path_masks)
    make_dir_check(output_dir + '/bbox')
    make_dir_check(output_dir + '/info')
    
    if visualisation:
        make_dir_check(path_overlayed_rerendered)

    meshes,vertices,all_faces,textures,T,R,S = load_all_meshes_as_one(shapenet_path,scene_info,device,catid_to_detectron_id)

    for frame in frames:
        if frame != '001000':
            continue
        camera,rasterizer,renderer_textured,width,height,R_scene_pose,T_scene_pose,focal_length = create_setup_calibration_matrix(scene_info,frame,input_dir,device,1)
        


        S = torch.Tensor(S).to(device)
        R = torch.Tensor(R).to(device)
        T = torch.Tensor(T).to(device)

        R_scene_pose = torch.Tensor(R_scene_pose[0]).to(device)
        T_scene_pose = torch.Tensor(T_scene_pose[0]).to(device)


        R_total = torch.matmul(R_scene_pose,R)
        T_total = T_scene_pose + torch.matmul(R_scene_pose,T)
        S_total = S
        print('R_scene_pose',R_scene_pose)
        print('T_scene_pose',T_scene_pose)
        print(vertices.shape)

        transformed_verts = torch.transpose(torch.matmul(R_total,torch.matmul(S_total,torch.transpose(vertices,0,1)),0,1)) + T_total

        # transformed_verts = torch.transpose(torch.matmul(R_scene_pose,torch.transpose(vertices,0,1)),0,1) + T_scene_pose

        print('mi',torch.min(transformed_verts,dim=0))
        print('max',torch.max(transformed_verts,dim=0))

        meshes = Meshes(verts=[transformed_verts],faces=[all_faces],textures=textures)
        
        if meshes != None:
            images = renderer_textured(meshes)
            image = images[0,:,:,:3].cpu().numpy()
        else:
            image = np.zeros((width,width,3))
        # flip
        image = image[::-1,::-1,:]
        # crop
        diff = width-height
        image_cropped = image[int(diff/2):-int(diff/2),:,:]
        image_clipped = np.clip(0,1,image_cropped)
        image_final = (np.round(image_clipped *255)).astype(np.uint8)

        cv2.imwrite('{}/{}.png'.format(path_rendered,frame),image_final)

        make_dir_check(path_masks + '/' + frame)
        make_dir_check(output_dir + '/bbox/' + frame)
        make_dir_check(output_dir + '/info/' + frame)
        save_separate_masks(image_final,len(scene_info['aligned_models']),path_masks + '/' + frame,min_pixel_per_mask)

        # save overlays
        if visualisation:
            seg_rerendered = cv2.imread('{}/{}.png'.format(path_rendered,frame))
            original = cv2.imread(input_dir + '/color/{}.jpg'.format(frame))
            color_image = get_color_image(seg_rerendered,len(scene_info['aligned_models']))
            overlayed = color_image /2 + original/2
            cv2.imwrite(output_dir + '/overlayed_instance_rerendered/{}.png'.format(frame),overlayed)


def create_category_to_index():
    CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture','display']
    VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39,40])
    category_to_index = {}
    # NOTE: + 1 as for some reason objects all have id +1 
    for i in range(len(VALID_CLASS_IDS)):
        category_to_index[CLASS_LABELS[i]] = VALID_CLASS_IDS[i] + 1
    return category_to_index,CLASS_LABELS

     
def dict_catid_to_detectron_id():
        with open('/scratches/octopus_2/fml35/datasets/scannet/shapenet_synset_list.txt') as text_file:
            lines = text_file.readlines()
        catid_to_detectron_id = {}
        list_objects = ['bed', 'sofa', 'chair', 'trash bin', 'cabinet', 'display', 'table','bookshelf','bathtub']
    
        for line in lines:
            split_line = line.split()
            id = split_line[0]
            name = ' '.join(split_line[1:])
            if name in list_objects:
                catid_to_detectron_id[id] = list_objects.index(name)
        
        return  catid_to_detectron_id,list_objects 


def get_test_scenes(path):
    with open(path,'r') as f:
        test_scenes = f.readlines()
    test_scenes = [scene.replace('\n','') for scene in test_scenes]
    return test_scenes


def main():

    min_pixel_per_mask = 100
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    with open('/scratches/octopus_2/fml35/datasets/scannet/scan2cad_annotations/full_annotations.json') as json_file:
        all_data = json.load(json_file)
    
    input_main_dir = '/scratches/octopus_2/fml35/datasets/scannet/scannet_frames_25k/'
    output_main_dir = '/scratches/octopus_2/fml35/datasets/scannet/scannet_frames_25k_own_data_exp_13_calibration_matrix/'
    shapenet_path = '/scratches/octopus_2/fml35/datasets/shapenet_v2/ShapeNetCore.v2/'
    test_scenes_path = '/scratches/octopus_2/fml35/datasets/scannet/data_splits/scannetv2_val.txt'
    
    test_scenes = get_test_scenes(test_scenes_path)
    catid_to_detectron_id,_ = dict_catid_to_detectron_id()
    # os.mkdir(output_main_dir)
    

    list_dir = os.listdir(input_main_dir)

    for i in tqdm(range(len(list_dir))):

        scan_id = list_dir[i]

        # if scan_id not in test_scenes:
        #     continue
        
        if scan_id != 'scene0017_00':
            continue
        print('only one scene')
        # get data for this scan_id
        for j in range(len(all_data)):
            if all_data[j]["id_scan"] == scan_id:
                break

        scene_info = all_data[j]
        # for key in scene_info:
        #     print(key)
        # print(scene_info["scene"])
        input_dir = input_main_dir  + scan_id
        output_dir = output_main_dir + scan_id
        rerender_single_scene(scene_info,input_dir,output_dir,device,shapenet_path,catid_to_detectron_id,min_pixel_per_mask)
        
        # masks_single_scene(scene_info,dir_path)


if __name__ == '__main__':
    main()



