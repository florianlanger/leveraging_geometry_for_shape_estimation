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
from pytorch3d.renderer import (look_at_view_transform,FoVPerspectiveCameras,RasterizationSettings, MeshRenderer, MeshRasterizer,SoftPhongShader,Textures)
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj
from scipy.spatial.transform import Rotation as scipy_rot


from leveraging_geometry_for_shape_estimation.pose_and_shape_optimisation.pose_selection import compute_rendered_pixel
from leveraging_geometry_for_shape_estimation.pose_and_shape_optimisation.vis_pose import load_mesh,render_mesh,overlay_rendered_image,plot_matches

    


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

        with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
            gt_infos = json.load(f)

        R = gt_infos["rot_mat"]
        T = gt_infos["trans_mat"]
        f = gt_infos["focal_length"]
        w = gt_infos["img_size"][0]
        h = gt_infos["img_size"][1]

        for i in range(top_n_retrieval):
            out_path = target_folder + '/matches_quality/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json'
            if os.path.exists(out_path):
                continue

            with open(target_folder + '/matches_orig_img_size/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json','r') as fil:
                matches_orig_img_size = json.load(fil)

            wc_matches = np.load(target_folder + '/wc_matches/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.npy')

            rendered_pixel = compute_rendered_pixel(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0),torch.from_numpy(wc_matches),f,w,h,global_config["pose_and_shape"]["pose"]["sensor_width"],already_batched=False)
            rendered_pixel = np.round(rendered_pixel.squeeze().numpy()).astype(int)


            quality_dict = {}
            quality_dict["pixels_real_orig_size"] = matches_orig_img_size["pixels_real_orig_size"]
            quality_dict["reprojected_rendered_gt_pose"] = rendered_pixel.tolist()
            quality_dict["distances"] = np.linalg.norm(np.array(matches_orig_img_size["pixels_real_orig_size"]) - rendered_pixel,axis=1).tolist()
            quality_dict["img_size"] = gt_infos["img_size"]

            with open(out_path,'w') as fil:
                json.dump(quality_dict,fil)


            if global_config["general"]["visualise"] == "True":
                if gt_infos["img"] in visualisation_list:

                    model_path = models_folder_read + "/models/remeshed/" + retrieval_list[i]["model"].replace('model/','')
                    mesh = load_mesh(model_path,R,T,device)

                    rendered_image = render_mesh(w,h,f,mesh,device)
                    original_image = cv2.imread(image_folder + '/' + gt_infos["img"])
                    out_img = overlay_rendered_image(original_image,rendered_image)

                    # reproject rendered keypoints
                    real_pixel = np.array(matches_orig_img_size["pixels_real_orig_size"])
                    out_img = plot_matches(out_img,real_pixel,rendered_pixel)

                    cv2.imwrite(target_folder + '/matches_quality_vis/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.png',out_img)







def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)


    get_pose_for_folder(global_config)

if __name__ == '__main__':
    print('Get quality of matches')
    main()