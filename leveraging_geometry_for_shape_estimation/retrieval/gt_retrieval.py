import json
import numpy as np
from tqdm import tqdm
import os

import torch

from scipy.spatial.transform import Rotation as scipy_rot


import sys

def get_nearest_elev_azim_from_R(R):
    
    
    rot = list(scipy_rot.from_matrix(R).as_euler('yzx', degrees=True))
    true_elev = -rot[2]
    true_azim = 180 - rot[0]


    azim_angles = torch.linspace(0.,360.,16+1)
    elev_angles = torch.linspace(0,45,4)
    _,nearest_neighbours_azim = torch.abs(azim_angles-true_azim).topk(1, largest=False)
    _,nearest_neighbours_elev = torch.abs(elev_angles-true_elev).topk(1, largest=False)


    azim = str((azim_angles[nearest_neighbours_azim[0]]%360).item()).zfill(3)
    elev = str(int(elev_angles[nearest_neighbours_elev[0]].item())).zfill(3)

    return elev,azim



def get_nn_from_gt_infos(gt_infos):

    infos = {}

    for key in ["model","category"]:
        infos[key] = gt_infos[key]

    elev,azim = get_nearest_elev_azim_from_R(gt_infos["rot_mat"])
    infos["elev"] = elev
    infos["azim"] = azim
    infos["name"] = infos["model"].split('/')[1] + '_' + infos["model"].split('/')[2]
    infos["path"] = "{}/{}/elev_{}_azim_{}.png".format(infos["category"],infos["model"].split('/')[2],elev,azim)

    nn = {"nearest_neighbours":[infos]}
    return nn




if __name__ == "__main__":   

    print('Get nn retrieval')

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["retrieval"]["gt"] == "True":

        target_folder = global_config["general"]["target_folder"]
        number_nn = global_config["retrieval"]["number_nearest_neighbours"]

        for name in tqdm(os.listdir(target_folder + '/cropped_and_masked_small')):

            img_path = target_folder + '/cropped_and_masked_small/' + name  

            new_name =  name.split('.')[0] + '.json'
        
            with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as file:
                gt_infos = json.load(file)
            

            nn = get_nn_from_gt_infos(gt_infos)

            with open(target_folder + '/nn_infos/' + new_name,'w') as f:
                json.dump(nn, f,indent=4)



    

    



   