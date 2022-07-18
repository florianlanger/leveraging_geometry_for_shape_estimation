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
import matplotlib.cm as cm

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform,FoVPerspectiveCameras,PerspectiveCameras,RasterizationSettings, MeshRenderer, MeshRasterizer,SoftPhongShader,Textures)
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj
from scipy.spatial.transform import Rotation as scipy_rot

from leveraging_geometry_for_shape_estimation.pose_and_shape_optimisation.pose_selection import compute_rendered_pixel
from leveraging_geometry_for_shape_estimation.vis_pose.vis_pose import load_mesh,render_mesh,overlay_rendered_image,plot_matches

    


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
        dist = np.linalg.norm(pixels_rendered[i] - pixels_real[i])
        rel_dist = np.clip(dist / max(img.shape),0,1)
        color = tuple(np.array(cm.hot(rel_dist)[:3])*255)
        
        cv2.line(img, tuple(pixels_rendered[i,::-1]), tuple(pixels_real[i,::-1]), color, thickness)

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
    # device = torch.device("cpu")
    torch.cuda.set_device(device)

    for name in tqdm(os.listdir(target_folder + '/cropped_and_masked')):
        
        with open(target_folder + '/nn_infos/' + name.split('.')[0] + '.json','r') as f:
            retrieval_list = json.load(f)["nearest_neighbours"]

        with open(target_folder + '/bbox_overlap/' + name.split('.')[0] + '.json','r') as f:
            bbox_overlap = json.load(f)

        with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
            gt_infos = json.load(f)

        # R = gt_infos["objects"][bbox_overlap['index_gt_objects']]["rot_mat"]
        R = gt_infos["objects"][bbox_overlap['index_gt_objects']]["rot_mat"]
        scaling = gt_infos["objects"][bbox_overlap['index_gt_objects']]["scaling"]
        T = gt_infos["objects"][bbox_overlap['index_gt_objects']]["trans_mat"]
        f = gt_infos["focal_length"]
        w = gt_infos["img_size"][0]
        h = gt_infos["img_size"][1]
        # K = np.array(gt_infos["K"])
        sw = global_config["pose_and_shape"]["pose"]["sensor_width"]
        # scaling = np.array(gt_infos["objects"][bbox_overlap['index_gt_objects']]["scaling"])
        # ppoint = torch.Tensor([[w/2-K[0,2],h/2 - K[1,2]]])/ (w/2)

        if global_config["general"]["visualise"] == "True":
            if gt_infos["img"] in visualisation_list:


        # print('R_with_scaling',R)
        # print('R no scaling',R_no_scaling)
        # print('scaling',scaling)
        # print('prod',np.matmul(R_no_scaling,np.diag(scaling)))

                for i in range(top_n_retrieval):
                    out_path = target_folder + '/matches_quality/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json'
                    # if os.path.exists(out_path):
                    #     continue
                
                    # if not 'scene0000_00-000700_0_000' in out_path:
                    #     continue 
                    # if not 'scene0000_00-000800_2_000' in out_path:
                    #     continue
                    

                    with open(target_folder + '/matches_orig_img_size/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json','r') as fil:
                        matches_orig_img_size = json.load(fil)

                    wc_matches = np.load(target_folder + '/wc_matches/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.npy')

                    # print(target_folder + '/wc_matches/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.npy')
                    # print(models_folder_read + "/models/remeshed/" + retrieval_list[i]["model"].replace('model/',''))
                    
                    # wc_matches = wc_matches * np.array([scaling[1],scaling[0],scaling[2]])

                    # R_torch = torch.Tensor(R).to(device) 
                    # T_torch  = torch.Tensor(T).to(device)
                    # # wc_matches = wc_matches * np.array([scaling[0],scaling[1],scaling[2]])
                    # # wc_matches = wc_matches[1:2]
        
                    # faces = []
                    # verts = []
                    # # print(ico_sphere().verts_list()[0])
                    # # print(df)
                    # for l,point in enumerate(wc_matches):
                    #     verts_orig = (0.02 * ico_sphere().verts_list()[0] + torch.Tensor(point).unsqueeze(0).repeat(12,1)).to(device)
                    #     vertices = torch.transpose(torch.matmul(R_torch,torch.transpose(verts_orig,0,1)),0,1) + T_torch
                    #     verts.append(vertices)
                    #     faces.append((ico_sphere().faces_list()[0]).to(device))

                
                    # textures = Textures(verts_rgb=torch.ones((wc_matches.shape[0],12,3),device=device))
                    # mesh = Meshes(verts=verts,faces=faces,textures=textures)

                    # rendered_image = render_mesh(w,h,f,mesh,device,ppoint)
                    # original_image = cv2.imread(image_folder + '/' + gt_infos["img"])
                    # out_img = overlay_rendered_image(original_image,rendered_image)

                    # cv2.imwrite(target_folder + '/matches_quality_vis/' + name.split('.')[0] + '_' + str(i).zfill(3) + '_sphere.png',out_img)

                    
                    # r_cam = torch.eye(3).unsqueeze(0).repeat(1,1,1)
                    # t_cam = torch.zeros((1,3))
                    # cameras_pix = PerspectiveCameras(device=device,focal_length = f,R = r_cam, T = t_cam,principal_point=ppoint,image_size=torch.Tensor([[h,w]]))
                    # # print('cameras_pix.get_full_projection_transform()',cameras_pix.get_full_projection_transform().get_matrix())
                    # pixel = cameras_pix.transform_points_screen(vertices)
                    # print('pixel',pixel)
                    # print(df)
                    # r_cam = torch.eye(3).unsqueeze(0).repeat(1,1,1)
                    # t_cam = torch.zeros((1,3))
                    # print('CHANGE BACK render_mesh in vis_pose')
                    # # cameras_pix = FoVPerspectiveCameras(device=device,fov = fov,degrees=False,R = r_cam, T = t_cam)
                    # # print('f',f,'r_cam',r_cam,'t_cam',t_cam,'ppoint',ppoint)
                    # # print('w',w,'h',h)
                    # cameras_pix = PerspectiveCameras(device=device,focal_length = f,R = r_cam, T = t_cam,principal_point=ppoint)
                    # # print('cameras_pix.get_full_projection_transform()',cameras_pix.get_full_projection_transform().get_matrix())
                    # pixel = cameras_pix.transform_points(torch.Tensor(wc_matches).to(device))
                    # print(pixel)
                    # p = pixel[:,:2] / pixel[:,2:].repeat(1,2)
                    
                    # rendered_image = render_mesh(w,h,f,mesh,device,ppoint)
                
                    # TODO figure out what is up with matches ?
            
                    sw = global_config["pose_and_shape"]["pose"]["sensor_width"]
                    rendered_pixel = compute_rendered_pixel(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0),torch.from_numpy(wc_matches),f,w,h,sw,already_batched=False)
                    rendered_pixel = np.round(rendered_pixel.squeeze(dim=0).numpy()).astype(int)


                    quality_dict = {}
                    quality_dict["pixels_real_orig_size"] = matches_orig_img_size["pixels_real_orig_size"]
                    quality_dict["reprojected_rendered_gt_pose"] = rendered_pixel.tolist()
                    quality_dict["distances"] = np.linalg.norm(np.array(matches_orig_img_size["pixels_real_orig_size"]) - rendered_pixel,axis=1).tolist()
                    quality_dict["img_size"] = gt_infos["img_size"]

                    with open(out_path,'w') as fil:
                        json.dump(quality_dict,fil)

                    model_path = models_folder_read + "/models/remeshed/" + retrieval_list[i]["model"].replace('model/','')
                    # print('LOAD SPECIAL BED')
                    # model_path = global_config["dataset"]["dir_path"] + "model/bed/edf13191dacf07af42d7295fb0533ac0/model_normalized.obj" #retrieval_list[i]["model"]
                    mesh = load_mesh(model_path,R,T,scaling,device)

                    rendered_image = render_mesh(w,h,f,mesh,device,sw)
                    original_image = cv2.imread(image_folder + '/' + gt_infos["img"])
                    out_img = overlay_rendered_image(original_image,rendered_image)

                    # # reproject rendered keypoints
                    real_pixel = np.array(matches_orig_img_size["pixels_real_orig_size"])
                    out_img = plot_matches(out_img,real_pixel,rendered_pixel)
                    # p = pixel.cpu().numpy()[:,0:2].astype(int)
                    # p = p[:,::-1]
                    # out_img = plot_matches(out_img,p,p)

                    cv2.imwrite(target_folder + '/matches_quality_vis/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.png',out_img)





def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)


    get_pose_for_folder(global_config)

if __name__ == '__main__':
    print('Get quality of matches')
    main()