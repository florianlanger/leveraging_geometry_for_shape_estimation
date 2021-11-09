import cv2
import numpy as np
import sys
import os
import json
from torchvision import models
from tqdm import tqdm
from pytorch3d.renderer import look_at_view_transform

import torch



def load_information_depth_camera(fov):

    assert fov == 60.0, "check this"

    val = 1/ np.tan(fov * np.pi/180 * 0.5)

    assert np.abs(val - 3.**0.5) < 0.00000001

    P_proj = torch.Tensor([[[val, 0.0000, 0.0000, 0.0000],
                            [0.0000, val, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 1.0000],
                            [0.0000, 0.0000, 1.0000, 0.0000]]])

    return P_proj
    focal_length = (W /2) / np.tan(np.pi/6)



def create_pixel_bearing(W,H,P_proj):

    assert W == H , "This function does not work if W and H diff, think need to change W and H in line above .view(H,W,4) but not sure if this is all"
    x = -(2 * (torch.linspace(0,W-1,W)+0.5)/W - 1)
    y = - (2 * (torch.linspace(0,H-1,H)+0.5)/H - 1)

    ys,xs = torch.meshgrid(x,y)

    xyz_hom = torch.stack([xs,ys,xs*0+1,xs*0+1],axis=-1)

    P_proj_inv = torch.inverse(P_proj[0,:,:])

    # TODO: whcih order W,H
    xyz_proj = torch.matmul(P_proj_inv,xyz_hom.view(W*H,4).T).T.view(W,H,4)


    pb_x = xyz_proj[:,:,0]
    pb_y = xyz_proj[:,:,1]
    pb_z = xyz_proj[:,:,2]
    return pb_x,pb_y,pb_z



def pb_and_depth_to_wc(pb_x,pb_y,pb_z,depth,elev,azim,mask):

    cc_x = pb_x * depth
    cc_y = pb_y * depth
    cc_z = pb_z * depth

    cc = torch.stack([cc_x,cc_y,cc_z],dim=-1)

    # transform coordinates
    elev = float(elev)
    azim = float(azim)


    R_mesh, T_mesh = look_at_view_transform(dist=1.2, elev=elev, azim=azim)
    R_mesh = R_mesh[0]
    T_mesh = T_mesh[0]

    cc_shape = cc.shape
    cc_translated = torch.transpose(torch.reshape(cc - T_mesh,(cc_shape[0]*cc_shape[1],cc_shape[2])),-1,-2)

    wc = torch.matmul(R_mesh, cc_translated)
    wc = torch.reshape(torch.transpose(wc,-1,-2),cc_shape)

    wc[~mask] = wc[~mask] * 0 - 1
    return wc

def get_world_coordinates(wc,pixels_rendered):

    wc = wc.numpy()
    
    search_increments = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1],
    # fist check those that 2 away right, up, left, down
    [2,0],[0,2],[-2,0],[0,-2],
    # then check also those offsetted by 1
    [2,-1],[2,1],[1,2],[-1,2],[-2,1],[-2,-1],[-1,-2],[1,-2],
    # finally corners
    [2,2],[-2,2],[-2,-2],[2,-2],
    [3,0],[3,1],[3,2],[3,3],[2,3],[1,3],[0,3],[-1,3],[-2,3],[-3,3],[-3,2],[-3,1],[-3,0],[-3,-1],[-3,-2],[-3,-3],[-2,-3],[-1,-3],[0,-3],[1,-3],[2,-3],[3,-3],[3,-2],[3,-1],
    [4,0],[4,1],[4,2],[4,3],[4,4],[3,4],[2,4],[1,4],[0,4],[-1,4],[-2,4],[-3,4],[-4,4],[-4,3],[-4,2],[-4,1],[-4,0],[-4,-1],[-4,-2],[-4,-3],[-4,-4],[-3,-4],[-2,-4],[-1,-4],[0,-4],[1,-4],[2,-4],[3,-4],[4,-4],[4,-3],[4,-2],[4,-1]
    ],dtype=int)

    world_coordinates_matches = []
    for i in range(pixels_rendered.shape[0]):
        pixel = pixels_rendered[i]
        for j in range(search_increments.shape[0]):
            if wc[pixel[0],pixel[1],2] != -1:

                world_coordinates_matches.append(list(wc[pixel[0],pixel[1]]))
                break
            

            pixel = pixels_rendered[i] + search_increments[j]

        if j == search_increments.shape[0] - 1:
            world_coordinates_matches.append([-100000,-100000,-100000])

    return np.array(world_coordinates_matches)




def get_3d_wc_for_folder(target_folder,models_folder_read,top_n_retrieval,fov,W):

    P_proj = load_information_depth_camera(fov)
    pb_x,pb_y,pb_z = create_pixel_bearing(W,W,P_proj)

    for name in tqdm(os.listdir(target_folder + '/cropped_and_masked')):
        
        with open(target_folder + '/nn_infos/' + name.split('.')[0] + '.json','r') as f:
            retrieval_list = json.load(f)["nearest_neighbours"]
        
        for i in range(top_n_retrieval):
            out_path = target_folder + '/wc_matches/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.npy'
            if os.path.exists(out_path):
                continue


            depth_path = models_folder_read + '/models/depth/' + retrieval_list[i]["path"].replace('.png','.npy')
            depth = np.load(depth_path)
            with open(target_folder + '/matches_orig_img_size/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json','r') as f:
                matches_orig_img_size = json.load(f)


            elev = retrieval_list[i]["elev"]
            azim = retrieval_list[i]["azim"]

            mask = (depth > -1)

            wc_grid = pb_and_depth_to_wc(pb_x,pb_y,pb_z,depth,elev,azim,mask)

            pixels_rendered = np.array(matches_orig_img_size["pixels_rendered"])

            wc = get_world_coordinates(wc_grid,pixels_rendered)

            np.save(out_path,wc)




def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    target_folder = global_config["general"]["target_folder"]
    models_folder_read = global_config["general"]["models_folder_read"]
    top_n_retrieval = global_config["keypoints"]["matching"]["top_n_retrieval"]
    fov = global_config["models"]["fov"]
    W = global_config["models"]["img_size"]
    print('pb_and_depth_to_wc is new function')
    print("Get matches 3D")

    get_3d_wc_for_folder(target_folder,models_folder_read,top_n_retrieval,fov,W)
    

    



if __name__ == '__main__':
    main()
