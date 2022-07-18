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
# import quaternion
import shutil

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform,PerspectiveCameras,FoVPerspectiveCameras,RasterizationSettings, MeshRenderer, MeshRasterizer,SoftPhongShader,Textures)
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj
from scipy.spatial.transform import Rotation as scipy_rot

from leveraging_geometry_for_shape_estimation.pose_and_shape_optimisation.pose_selection import compute_rendered_pixel,compute_rendered_pixel_shape,stretch_3d_coordinates
from leveraging_geometry_for_shape_estimation.utilities.write_on_images import draw_text_block
from leveraging_geometry_for_shape_estimation.utilities.dicts import open_json_precomputed_or_current,determine_base_dir,load_json
from leveraging_geometry_for_shape_estimation.models.remesh_models import remesh

def load_mesh(full_path,R,T,scaling,device,color=(1,1,1),remesh=False):
    assert os.path.exists(full_path),full_path

    if remesh == False:
        vertices_origin,faces,_ = load_obj(full_path, device=device,create_texture_atlas=False, load_textures=False)
        faces = faces[0]
    elif remesh == True:
        vertices_origin,faces = remesh(full_path,max_edge_length=0.1,device=device)
    R = torch.Tensor(R).to(device) 
    T = torch.Tensor(T).to(device)
    scaling = torch.Tensor(scaling).to(device)
    scaled_vertices = vertices_origin * scaling.unsqueeze(0).repeat(vertices_origin.shape[0],1)
    vertices = torch.transpose(torch.matmul(R,torch.transpose(scaled_vertices,0,1)),0,1) + T
    # if color == None:
    #     textures = Textures(verts_rgb=torch.ones((1,vertices.shape[0],3),device=device))
    # else:
    textures = Textures(verts_rgb=torch.ones((1,vertices.shape[0],3),device=device)*torch.Tensor(color).to(device))
    mesh = Meshes(verts=[vertices], faces=[faces],textures=textures)
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


def render_mesh(w,h,f,mesh,device,sw,flip=False):
    if w >= h:
        fov = 2 * np.arctan((sw/2)/f)
    elif w < h:
        fov = 2 * np.arctan(((sw/2) * h/w)/f)

    # fov = 2 * np.arctan(w/(2*1169))


    r_cam = torch.eye(3).unsqueeze(0).repeat(1,1,1)
    t_cam = torch.zeros((1,3))
    # print('CHANGE BACK render_mesh in vis_pose')
    cameras_pix = FoVPerspectiveCameras(device=device,fov = fov,degrees=False,R = r_cam, T = t_cam)
    # print('f',f,'r_cam',r_cam,'t_cam',t_cam,'ppoint',ppoint)
    # print('w',w,'h',h)
    # cameras_pix = PerspectiveCameras(device=device,focal_length = f,R = r_cam, T = t_cam,principal_point=ppoint,image_size=torch.Tensor([[h,w]]))
    # print('cameras_pix.get_full_projection_transform()',cameras_pix.get_full_projection_transform().get_matrix())
    #principal_point=ppoint

    raster_settings_soft = RasterizationSettings(image_size = max(w,h),blur_radius=0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras_pix,raster_settings=raster_settings_soft)
    renderer_textured = MeshRenderer(rasterizer=rasterizer,shader=SoftPhongShader(device=device))

    image = renderer_textured(mesh,cameras=cameras_pix).cpu().numpy()[0,:,:,:]

    # # flip
    # image = image[::-1,::-1,:]
    # # crop
    # diff = w-h
    # image_cropped = image[int(diff/2):-int(diff/2),:,:]
    # # print(np.max(image_cropped))
    # # print(np.min(image_cropped))
    # image_clipped = np.clip(0,255,image_cropped*255).astype(int)
    # image = image_clipped
    # print(image[:3,:3])

    # crop
    if w >= h:
        image = image[int((w-h)/2):int((w+h)/2),:,:]
    elif w < h:
        image = image[:,int((h-w)/2):int((h+w)/2),:]

    if flip:
        image = image[::-1,::-1,:]

    return image

def render_mesh_from_calibration(w,h,K,mesh,device,r_cam=torch.eye(3),t_cam=torch.zeros(3)):

    r_cam = r_cam.unsqueeze(0)
    t_cam = t_cam.unsqueeze(0)
    cameras_pix = PerspectiveCameras(device=device,K = K.unsqueeze(0),R = r_cam, T = t_cam)

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

def overlay_rendered_image_v2(original_image,rendered_image):

    image = np.round((255*rendered_image)).astype(np.uint8)
    mask = np.where((image != [255,255,255,0]).all(axis = 2))
    original_image[mask] = 0.1 * original_image[mask[0],mask[1],:] +  0.9 * image[mask[0],mask[1],:3]
    original_image = np.clip(original_image,a_min=0,a_max=255).astype(np.uint8)
    return original_image

def just_rendered_image(rendered_image):

    image = np.round((255*rendered_image)).astype(np.uint8)
    # image[np.where((image == [255,255,255,0]).all(axis = 2))] = [255,255,255,0]
    return image[:,:,:3]

def plot_points(img,pixels_real,color=(0,255,0)):
    assert type(pixels_real).__module__ == 'numpy'

    size = int(max(img.shape) / 200)
    size = max(1,size)

    for pixels in pixels_real:
        if (pixels >= 0).all():
            try:
                img[pixels[0]-size:pixels[0]+size,pixels[1]-size:pixels[1]+size,:] = np.tile(np.array([[color]]),(2*size,2*size,1))
            except ValueError:
                pass

    return img

def plot_matches(img,pixels_real,pixels_rendered,line_color=(0,255,0)):
    assert pixels_real.shape == pixels_rendered.shape
    assert type(pixels_real).__module__ == 'numpy'
    assert type(pixels_rendered).__module__ == 'numpy'


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
    for i in range(pixels_rendered.shape[0]):
        # if (pixels_rendered[i] >= 0).all() and (pixels_real[i] >= 0).all():
        cv2.line(img, tuple(pixels_rendered[i,::-1]), tuple(pixels_real[i,::-1]), line_color, thickness)

    return img


def plot_matches_individual_color(img,pixels_real,pixels_rendered,line_colors):
    assert pixels_real.shape == pixels_rendered.shape
    assert type(pixels_real).__module__ == 'numpy'
    assert type(pixels_rendered).__module__ == 'numpy'


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


    for i in range(pixels_rendered.shape[0]):
        # if (pixels_rendered[i] >= 0).all() and (pixels_real[i] >= 0).all():

        cv2.line(img, tuple(pixels_rendered[i,::-1]), tuple(pixels_real[i,::-1]), line_colors[i], thickness)

    return img

def plot_matches_individual_color_no_endpoints(img,pixels1,pixels2,line_colors=[255,0,0],thickness=None):

    assert pixels1.shape == pixels2.shape
    assert pixels1.shape[1] == 2
    if type(line_colors[0]) != int:
        assert len(line_colors) == pixels1.shape[0]

    pixels1 = np.round(np.array(pixels1)).astype(int)
    pixels2 = np.round(np.array(pixels2)).astype(int)

    scale = max(img.shape) / 500.
    if thickness == None:
        thickness = max(int(np.round(scale)),1)

    if type(line_colors[0]) == int:
        line_colors = [line_colors] * pixels1.shape[0]

    # print('pixels1',pixels1)
    # print('pixels2',pixels2)
    for i in range(pixels1.shape[0]):
        # print(tuple(pixels1[i]),tuple(pixels2[i]))
        # had crash here if numbers overflow or wrong type which think because was say int32 instead of int16
        if (pixels1[i] > -10000).all() and (pixels1 < 10000).all() and (pixels2[i] > -10000).all() and (pixels2 < 10000).all():
            cv2.line(img, tuple(pixels1[i]), tuple(pixels2[i]), line_colors[i], thickness)

    return img
    

def plot_polygons(img,pix1_3d,pix2_3d,pix1_2d,pix2_2d,indices_lines_2d_to_3d,indices_which_way_round,mask_plot_polygons):


    assert indices_which_way_round.shape[0] == pix1_2d.shape[0]
    for i in range(pix1_2d.shape[0]):
        if mask_plot_polygons[i] == False:
            continue

        p1 = pix1_3d[indices_lines_2d_to_3d[i]]
        p2 = pix2_3d[indices_lines_2d_to_3d[i]]

        if indices_which_way_round[i] == 0:
            p3 = pix1_2d[i]
            p4 = pix2_2d[i]
        elif indices_which_way_round[i] == 1:
            p3 = pix2_2d[i]
            p4 = pix1_2d[i]

        pts = torch.stack([p1,p2,p3,p4])
        pts = np.round(pts.numpy()).astype(np.int32)
        # if (pts > 0).all():
        cv2.fillPoly(img,[pts],(0,255,255,0.5))

    return img

def plot_polygons_v2(img,pix1_3d,pix2_3d,pix1_2d,pix2_2d,indices_lines_2d_to_3d,indices_which_way_round,mask_plot_polygons):

    all_indices = [[0,1,2,3],[0,1,3,2],[0,2,1,3],[0,2,3,1],[0,3,1,2],[0,3,2,1]]

    assert indices_which_way_round.shape[0] == pix1_2d.shape[0]
    for i in range(pix1_2d.shape[0]):
        if mask_plot_polygons[i] == False:
            continue

        
        p1 = pix1_3d[indices_lines_2d_to_3d[i]]
        p2 = pix2_3d[indices_lines_2d_to_3d[i]]
        p3 = pix1_2d[i]
        p4 = pix2_2d[i]
        points = [p1,p2,p3,p4]

        selected_indices = all_indices[indices_which_way_round[i,indices_lines_2d_to_3d[i]]]


        pts = np.stack([points[selected_indices[0]],points[selected_indices[1]],points[selected_indices[2]],points[selected_indices[3]]])
        # print(np.stack([points[selected_indices[0]],points[selected_indices[1]],points[selected_indices[2]],points[selected_indices[3]]]))
        pts = np.round(pts).astype(np.int32)
        # if (pts > 0).all():
        cv2.fillPoly(img,[pts],(0,255,255,0.5))

    return img


def put_text(img,text,relative_position):
    h,w,_ = img.shape
    font_size = min(h,w)/25 * 0.05
    thickness = int(font_size * 4)

    y = int(relative_position[0] * h)
    x = int(relative_position[1] * w)
    # x = int(min(h,w) /8)
    # y = 2*x
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, font_size,(255, 0, 0), thickness, cv2.LINE_AA)
    return img


def get_pose_for_folder(global_config):

    target_folder = global_config["general"]["target_folder"]
    image_folder = determine_base_dir(global_config,'segmentation') + '/images'
    models_folder_read = global_config["general"]["models_folder_read"]
    top_n_retrieval = global_config["keypoints"]["matching"]["top_n_retrieval"]

    with open(target_folder + '/global_stats/visualisation_images.json','r') as f:
        visualisation_list = json.load(f)


    pose_config = global_config["pose_and_shape"]["pose"]

    device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))
    torch.cuda.set_device(device)

    evaluate_all = global_config["evaluate_poses"]["evaluate_all"]

    if global_config["pose_and_shape_probabilistic"]["use_probabilistic"] == "True":
        n_rotations = 4
    else:
        n_rotations = 1

    print('DEBUG vis pose')
    print('NEED SCALING FOR ROCA')
    print('NO METRICS')

    print('DEBUG NOT SELECTED')
    for name in tqdm(sorted(os.listdir(determine_base_dir(global_config,'segmentation') + '/cropped_and_masked'))):
    
        # if not "scene0025_01-001000" in name and not "scene0025_01-001000" in name:
        #     continue

        # if not 'sofa_0149' in name:
        #     continue

        gt_infos = open_json_precomputed_or_current('/gt_infos/' + name.rsplit('_',1)[0] + '.json',global_config,'segmentation')
        bbox_overlap = open_json_precomputed_or_current('/bbox_overlap/' + name.split('.')[0] + '.json',global_config,'segmentation')
        retrieval_list = open_json_precomputed_or_current('/nn_infos/' + name.split('.')[0] + '.json',global_config,'retrieval')["nearest_neighbours"]
        
        # with open(target_folder + '/nn_infos/' + name.split('.')[0] + '.json','r') as f:
        #     retrieval_list = json.load(f)["nearest_neighbours"]

        # with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
        #     gt_infos = json.load(f)

        # with open(target_folder + '/bbox_overlap/' + name.split('.')[0] + '.json','r') as f:
        #     bbox_overlap = json.load(f)

        if gt_infos["img"] in visualisation_list:

            with open(target_folder + '/selected_nn/' + name.split('.')[0] + '.json','r') as f:
                selected_info = json.load(f)
            # selected_info = {}
            # selected_info["selected_orientation"] = 0
            # selected_info["selected_nn"] = 0

            for i in range(top_n_retrieval):
                for k in range(n_rotations):

                    if evaluate_all == False and (i != selected_info["selected_nn"] or k !=  selected_info["selected_orientation"]):
                        continue


                    out_path = target_folder + '/poses_vis/' + name.split('.')[0] + '_' + str(i).zfill(3) + '_' + str(k).zfill(2) + '.png'
                    # print(out_path)
                    # shutil.copy(out_path.replace('/poses_vis/','/poses_vis_old/'),out_path)
                    # continue
                    # if os.path.exists(out_path):
                    #     continue

                    with open(target_folder + '/poses/' + name.split('.')[0] + '_' + str(i).zfill(3) + '_' + str(k).zfill(2) + '.json','r') as f:
                        pose_info = json.load(f)

                    vis_matches = False
                    if os.path.exists(determine_base_dir(global_config,'segmentation') + '/matches_orig_img_size/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json'):
                        vis_matches = True
                        with open(determine_base_dir(global_config,'segmentation') + '/matches_orig_img_size/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json','r') as f:
                            matches_orig_img_size = json.load(f)
                        wc_matches = np.load(determine_base_dir(global_config,'segmentation') + '/wc_matches/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.npy')


                    # predictions
                    model_path = global_config["dataset"]["dir_path"] + retrieval_list[i]["model"]
                    R = pose_info["predicted_r"]
                    T = pose_info["predicted_t"]
                    scaling = pose_info["predicted_s"]

                    # GT
                    # model_path = global_config["dataset"]["dir_path"] + gt_infos["objects"][bbox_overlap['index_gt_objects']]["model"]
                    # R = gt_infos["objects"][bbox_overlap['index_gt_objects']]["rot_mat"]
                    # T = gt_infos["objects"][bbox_overlap['index_gt_objects']]["trans_mat"]
                    # scaling = gt_infos["objects"][bbox_overlap['index_gt_objects']]["scaling"]

                    # invert = np.array([[-1,0,0],[0,-1,0],[0,0,1.]])
                    # print(' invert')
                    # R = np.matmul(invert,np.array(R)).tolist()
                    # T = np.matmul(invert,np.array(T)).tolist()


                    # m=4
                    # for i in range(m):
                    #     R_rotated = (quaternion.from_rotation_matrix(R)*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0]))
                    #     R_rotated = quaternion.as_rotation_matrix(R_rotated)
                    #     out_path = target_folder + '/poses_vis_symm_object_rotations/' + name.split('.')[0] + '_' + str(i).zfill(3) + '_' + str(k).zfill(2) + '__rot_' + str(m) + '.png'
                   

                    f = gt_infos["focal_length"]
                    # w = gt_infos["img_size"][0]
                    # h = gt_infos["img_size"][1]
                    sw = global_config["pose_and_shape"]["pose"]["sensor_width"]

                    # was bug where assigned wrong w and h to images
                    original_image = cv2.imread(image_folder + '/' + gt_infos["img"])
                    h,w,_ = original_image.shape

                    

                    if global_config["pose_and_shape"]["shape"]["optimise_shape"] == "False":
                        mesh = load_mesh(model_path,R,T,scaling,device)
                        if vis_matches:
                            rendered_pixel = compute_rendered_pixel(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0),torch.from_numpy(wc_matches),f,w,h,sw,already_batched=False)
                    
                    elif global_config["pose_and_shape"]["shape"]["optimise_shape"] == "True":
                        print('Unsqueeze ?')
                        predicted_stretching = torch.Tensor(pose_info["predicted_stretching"]).to(device).unsqueeze(0)
                        planes = torch.Tensor(global_config["pose_and_shape"]["shape"]["planes"]).to(device)
                        mesh = load_mesh_shape(model_path,R,T,planes,predicted_stretching,device)
                        if vis_matches:
                            rendered_pixel = compute_rendered_pixel_shape(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0),predicted_stretching.cpu(),planes.cpu(),torch.from_numpy(wc_matches),f,w,h,global_config["pose_and_shape"]["pose"]["sensor_width"],already_batched = False)

                    rendered_image = render_mesh(w,h,f,mesh,device,sw,flip=False)
                    
                    out_img = overlay_rendered_image(original_image,rendered_image)

                    if vis_matches:
                        
                        rendered_pixel = np.round(rendered_pixel.squeeze(dim=0).numpy()).astype(int)[pose_info["indices"]]
                        real_pixel = np.array(matches_orig_img_size["pixels_real_orig_size"])[pose_info["indices"]]
            
                        out_img = plot_matches(out_img,real_pixel,rendered_pixel)


                    selected = selected_info['selected_nn'] == i and selected_info['selected_orientation'] == k
                    if selected:
                        selected_text = 'SELECTED'
                    else:
                        selected_text = 'NOT SELECTED'

                    # F1_score = str(np.round(metrics["F1@0.300000"],1))
                    # text = 'F1: ' + F1_score + 'total: ' + str(np.round(metrics["total_angle_diff"],1)) + 'e: ' + str(np.round(metrics["diff_elev"],1)) + ' a: ' + str(np.round(metrics["diff_azim"],1)) + ' t: ' + str(np.round(metrics["diff_tilt"],1))


                    # with open(target_folder + '/metrics/' + name.split('.')[0] + '_' + str(i).zfill(3) + '_' + str(k).zfill(2) + '.json','r') as f:
                    #     metrics = json.load(f)

                    # out_img = put_text(out_img,selected_text,[0.1,0.1])
                    # out_img = put_text(out_img,'F1: ' + str(np.round(metrics["F1@0.300000"],1)),[0.2,0.1])
                    # out_img = put_text(out_img,'total: ' + str(np.round(metrics["total_angle_diff"],1)),[0.4,0.1])
                    # out_img = put_text(out_img,'t: ' + str(np.round(metrics["diff_tilt"],1)) + ' a: ' + str(np.round(metrics["diff_azim"],1)) + ' e: ' + str(np.round(metrics["diff_elev"],1)),[0.6,0.1])
                    

                    # text = 'F1: ' + F1_score + 'total: ' + str(np.round(metrics["total_angle_diff"],1)) + 'e: ' + str(np.round(metrics["diff_elev"],1)) + ' a: ' + str(np.round(metrics["diff_azim"],1)) + ' t: ' + str(np.round(metrics["diff_tilt"],1))

                    if os.path.exists(target_folder + '/metrics_scannet/' + name.split('.')[0] + '_' + str(i).zfill(3) + '_' + str(k).zfill(2) + '.json'):
                        with open(target_folder + '/metrics_scannet/' + name.split('.')[0] + '_' + str(i).zfill(3) + '_' + str(k).zfill(2) + '.json','r') as f:
                            metrics = json.load(f)

                        out_img = put_text(out_img,'t: ' + str(np.round(metrics["translation_error"],3)),[0.2,0.1])
                        out_img = put_text(out_img,'t_all: ' + str(np.round(metrics["translation_error_all"],3).tolist()),[0.3,0.1])
                        out_img = put_text(out_img,'r: ' + str(np.round(metrics["rotation_error"],1)),[0.4,0.1])
                        out_img = put_text(out_img,'s: ' + str(np.round(metrics["scaling_error"],1)),[0.5,0.1])
                        out_img = put_text(out_img,'s_all: ' + str(np.round(metrics["scaling_error_all"],3).tolist()),[0.6,0.1])
                        out_img = put_text(out_img,'cad: ' + str(metrics["retrieval_correct"]),[0.7,0.1])
                    
                    
                    out_img = cv2.resize(out_img,(480,360))
                    cv2.imwrite(out_path,out_img)







def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["general"]["visualise"] == "True":
        print('Visualising Poses')
        get_pose_for_folder(global_config)
    print(global_config["general"]["target_folder"])

if __name__ == '__main__':
    main()