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


sys.path.insert(0,'/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/')
from pose_and_shape_optimisation.pose_selection import compute_rendered_pixel,compute_rendered_pixel_shape,stretch_3d_coordinates


def load_mesh(full_path,R,T,device):
    vertices_origin,faces,_ = load_obj(full_path, device=device,create_texture_atlas=False, load_textures=False)
    R = torch.Tensor(R).to(device) 
    T = torch.Tensor(T).to(device)
    vertices = torch.transpose(torch.matmul(R,torch.transpose(vertices_origin,0,1)),0,1) + T
    textures = Textures(verts_rgb=torch.ones((1,vertices.shape[0],3),device=device))
    mesh = Meshes(verts=[vertices], faces=[faces[0]],textures=textures)
    return mesh

def load_mesh_shape(full_path,R,T,planes,predicted_stretching,device):
    vertices_origin,faces,_ = load_obj(full_path, device=device,create_texture_atlas=False, load_textures=False)
    R = torch.Tensor(R).to(device) 
    T = torch.Tensor(T).to(device)
    stretched_vertices = stretch_3d_coordinates(vertices_origin.unsqueeze(0),planes,predicted_stretching).squeeze()
    vertices = torch.transpose(torch.matmul(R,torch.transpose(stretched_vertices,0,1)),0,1) + T
    textures = Textures(verts_rgb=torch.ones((1,vertices.shape[0],3),device=device))
    mesh = Meshes(verts=[vertices], faces=[faces[0]],textures=textures)

    return mesh


def render_mesh(w,h,f,mesh,device):
    if w >= h:
        fov = 2 * np.arctan((16.)/f)
    elif w < h:
        fov = 2 * np.arctan((16. * h/w)/f)

    r_cam = torch.eye(3).unsqueeze(0).repeat(1,1,1)
    t_cam = torch.zeros((1,3))
    cameras_pix = FoVPerspectiveCameras(device=device,fov = fov,degrees=False,R = r_cam, T = t_cam)
    raster_settings_soft = RasterizationSettings(image_size = max(w,h),blur_radius=0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras_pix,raster_settings=raster_settings_soft)
    renderer_textured = MeshRenderer(rasterizer=rasterizer,shader=SoftPhongShader(device=device))

    image = renderer_textured(mesh,cameras=cameras_pix).cpu().numpy()[0,:,:,:]
    # crop
    if w >= h:
        image = image[int((w-h)/2):int((w+h)/2),:,:]
    elif w < h:
        image = image[:,int((h-w)/2):int((h+w)/2),:]

    return image


def overlay_rendered_image(original_image,rendered_image):
    h,w,_ = original_image.shape
    alpha = 255*np.ones((h,w,4),dtype=np.uint8)
    alpha[:,:,:3] = original_image
    alpha = np.clip(alpha,a_min=0,a_max=255)

    image = np.round((255*rendered_image)).astype(np.uint8)
    image[np.where((image == [255,255,255,0]).all(axis = 2))] = [0,0,0,0]
    overlayed = cv2.addWeighted(alpha[:,:,:3],0.5,image[:,:,:3],0.5,0)
    return overlayed




def plot_matches(img,pixels_real,pixels_rendered):

    assert pixels_real.shape == pixels_rendered.shape

    size = int(max(img.shape) / 200)
    size = max(1,size)


    kinds = [pixels_real,pixels_rendered]
    colors = [[255,0,0],[0,0,255],[0,255,0]]

    for color,kind in zip(colors,kinds):
        if len(kind.shape) != 1:
            for pixels in kind:
                if (pixels >= 0).all():
                    try:
                        img[pixels[0]-size:pixels[0]+size,pixels[1]-size:pixels[1]+size,:] = np.tile(np.array([[color]]),(2*size,2*size,1))
                    except ValueError:
                        pass

    
    scale = max(img.shape) / 500.
    thickness = max(int(np.round(scale)),1)
    line_color = (0,255,0)
    for i in range(pixels_rendered.shape[0]):
        cv2.line(img, tuple(pixels_rendered[i,::-1]), tuple(pixels_real[i,::-1]), line_color, thickness)

    return img
    



def get_pose_for_folder(global_config):

    target_folder = global_config["general"]["target_folder"]
    image_folder = global_config["general"]["image_folder"]
    models_folder_read = global_config["general"]["models_folder_read"]
    top_n_retrieval = global_config["keypoints"]["matching"]["top_n_retrieval"]

    with open(target_folder + '/global_stats/visualisation_images.json','r') as f:
        visualisation_list = json.load(f)


    pose_config = global_config["pose_and_shape"]["pose"]

    device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))
    torch.cuda.set_device(device)

    for name in tqdm(os.listdir(target_folder + '/cropped_and_masked')):
        
        with open(target_folder + '/nn_infos/' + name.split('.')[0] + '.json','r') as f:
            retrieval_list = json.load(f)["nearest_neighbours"]

        with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
            gt_infos = json.load(f)

        if gt_infos["img"] in visualisation_list:

            for i in range(top_n_retrieval):

                    out_path = target_folder + '/poses_vis/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.png'
                    # if os.path.exists(out_path):
                    #     continue
                    print(out_path)
                    print('debug dont skip')

                    with open(target_folder + '/poses/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json','r') as f:
                        pose_info = json.load(f)

                    with open(target_folder + '/matches_orig_img_size/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json','r') as f:
                        matches_orig_img_size = json.load(f)

                    wc_matches = np.load(target_folder + '/wc_matches/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.npy')


                    model_path = models_folder_read + "/models/remeshed/" + retrieval_list[i]["model"].replace('model/','')
                    R = pose_info["predicted_r"]
                    T = pose_info["predicted_t"]
                    f = gt_infos["focal_length"]
                    w = gt_infos["img_size"][0]
                    h = gt_infos["img_size"][1]

                    if global_config["pose_and_shape"]["shape"]["optimise_shape"] == "False":
                        mesh = load_mesh(model_path,R,T,device)
                        rendered_pixel = compute_rendered_pixel(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0),torch.from_numpy(wc_matches),f,w,h,global_config["pose_and_shape"]["pose"]["sensor_width"],already_batched=False)
                    
                    elif global_config["pose_and_shape"]["shape"]["optimise_shape"] == "True":
                        print('Unsqueeze ?')
                        predicted_stretching = torch.Tensor(pose_info["predicted_stretching"]).to(device).unsqueeze(0)
                        planes = torch.Tensor(global_config["pose_and_shape"]["shape"]["planes"]).to(device)
                        mesh = load_mesh_shape(model_path,R,T,planes,predicted_stretching,device)
                        rendered_pixel = compute_rendered_pixel_shape(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0),predicted_stretching.cpu(),planes.cpu(),torch.from_numpy(wc_matches),f,w,h,global_config["pose_and_shape"]["pose"]["sensor_width"],already_batched = False)

                    

                    rendered_image = render_mesh(w,h,f,mesh,device)
                    original_image = cv2.imread(image_folder + '/' + gt_infos["img"])
                    out_img = overlay_rendered_image(original_image,rendered_image)

                    
                    rendered_pixel = np.round(rendered_pixel.squeeze().numpy()).astype(int)[pose_info["indices"]]
                    real_pixel = np.array(matches_orig_img_size["pixels_real_orig_size"])[pose_info["indices"]]
                    out_img = plot_matches(out_img,real_pixel,rendered_pixel)

                    cv2.imwrite(out_path,out_img)







def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["general"]["visualise"] == "True":
        print('Visualising Poses')
        get_pose_for_folder(global_config)

if __name__ == '__main__':
    main()