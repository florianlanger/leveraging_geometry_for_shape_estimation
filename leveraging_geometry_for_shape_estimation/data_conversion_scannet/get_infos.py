from glob import glob
from socket import INADDR_ALLHOSTS_GROUP
import cv2
import json
import os
import scipy.ndimage
import numpy as np
from scipy.spatial.transform import Rotation as scipy_rot
from tqdm import tqdm

def make_dir_check(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def make_M_from_tqs(t, q, s):
    # q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    # R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    q = [q[1],q[2],q[3],q[0]]
    R[0:3, 0:3] = scipy_rot.from_quat(q).as_matrix()
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M,R[0:3, 0:3],S[0:3, 0:3]

def load_4by4_from_txt(path):
    M = np.zeros((4,4))
    with open(path,'r') as f:
        content = f.readlines()
        for i in range(4):
            line = content[i].split()
            for j in range(4):
                M[i,j] = np.float32(line[j])
        return M

def get_scene_pose(frame_4by4_path,scene_trs):
    # R_and_T = load_4by4_from_txt(dir_path + scene['id_scan'] + '/pose/' + frame + '.txt')
    R_and_T = load_4by4_from_txt(frame_4by4_path)
    R_pose = R_and_T[:3,:3].copy()
    T_pose = R_and_T[:3,3].copy()

    T_pose = np.concatenate((T_pose,np.ones((1))),axis=0)
    

    # Mscene,_,_ = make_M_from_tqs(scene["trs"]['translation'],scene["trs"]['rotation'],scene["trs"]['scale'])
    Mscene,_,_ = make_M_from_tqs(scene_trs['translation'],scene_trs['rotation'],scene_trs['scale'])

    T_scene_pose = np.matmul(Mscene,T_pose)[:3]

    R_scene_pose = np.matmul(Mscene[:3,:3],R_pose).copy()
    R_scene_pose = np.linalg.inv(R_scene_pose)
    
    T_final_pose = - np.matmul(R_scene_pose,T_scene_pose)
    R_final_pose = np.linalg.inv(R_scene_pose)

    return R_scene_pose,T_final_pose

def get_scene_pose_2(frame_4by4_path,scene_trs):
    # R_and_T = load_4by4_from_txt(dir_path + scene['id_scan'] + '/pose/' + frame + '.txt')
    R_and_T = load_4by4_from_txt(frame_4by4_path)
    R_pose = R_and_T[:3,:3].copy()
    T_pose = R_and_T[:3,3].copy()

    T_pose = np.concatenate((T_pose,np.ones((1))),axis=0)
    

    # Mscene,_,_ = make_M_from_tqs(scene["trs"]['translation'],scene["trs"]['rotation'],scene["trs"]['scale'])
    Mscene,_,_ = make_M_from_tqs(scene_trs['translation'],scene_trs['rotation'],scene_trs['scale'])

    T_scene_pose = np.matmul(Mscene,T_pose)[:3]

    R_scene_pose = np.matmul(Mscene[:3,:3],R_pose).copy()
    R_scene_pose = np.linalg.inv(R_scene_pose)
    
    T_final_pose = np.matmul(np.linalg.inv(R_scene_pose),T_scene_pose)
    R_final_pose = np.linalg.inv(R_scene_pose)

    return R_scene_pose,T_final_pose



def main():
    with open('/scratch2/fml35/datasets/scannet/scan2cad_annotations/full_annotations.json') as json_file:
            all_data = json.load(json_file)

    with open('/scratch2/fml35/datasets/shapenet_v2/ShapeNetCore.v2/taxonomy.json') as file:
        shapenet_taxonomy = json.load(file)

    catid_to_cat = {}
    for item in shapenet_taxonomy:
        catid_to_cat[item["synsetId"]] = item["name"]

    dir_path = '/scratch2/fml35/datasets/scannet/scannet_frames_25k/'
    # dir_path_own_data = '/scratch2/fml35/datasets/scannet/scannet_frames_25k_own_data_exp_02/'
    dir_path_own_data = '/scratch2/fml35/datasets/scannet/scannet_frames_25k_own_data_exp_12_calibration_matrix/'

    cats = ["bathtub,bathing tub,bath,tub","bed","ashcan,trash can,garbage can,wastebin,ash bin,ash-bin,ashbin,dustbin,trash barrel,trash bin","bookshelf","cabinet","chair","display,video display","sofa,couch,lounge","table"]
    cats_renamed = ["bathtub","bed","bin","bookshelf","cabinet","chair","display","sofa","table"]

    for scene_info in tqdm(all_data):
        scene = scene_info["id_scan"]

        if not os.path.exists(dir_path_own_data + scene + '/masks/'):
            # print('cotn')
            # print(dir_path_own_data + scene + '/masks/')
            continue

        for frame in os.listdir(dir_path_own_data + scene + '/masks/'):

            make_dir_check(dir_path_own_data + scene + '/info/' + frame)

            # if '0653' not in scene or '001300' not in frame:
            #     continue

            for object in os.listdir(dir_path_own_data + scene + '/masks/' + frame):
                # print(dir_path_own_data + scene + '/masks/' + frame)

                # if not dir_path_own_data + scene + '/masks/' + frame  == '/scratch2/fml35/datasets/scannet/scannet_frames_25k_own_data_exp_08_min_pixel_per_mask_goban1_test_scenes/scene0378_01/masks/000000':
                #     continue

                # mask = cv2.imread(dir_path_own_data + scene + '/masks/' + frame + '/' + object)
                # if np.sum(mask > 0) < 10000:
                #     continue

                # if dir_path_own_data + scene + '/masks/' + frame + '/' + object != "/scratch2/fml35/datasets/scannet/scannet_frames_25k_own_data/scene0000_00/masks/000400/007.png":
                #     continue
                # print('in')
                id_object = int(object.split('.')[0])
                model = scene_info['aligned_models'][id_object]
                
                frame_4by4_path = dir_path + scene_info['id_scan'] + '/pose/' + frame + '.txt'
                # print(frame_4by4_path)
                scene_trs = scene_info["trs"]
                R_scene_pose,T_scene_pose = get_scene_pose(frame_4by4_path,scene_trs)
            
                t = model["trs"]["translation"]
                q = model["trs"]["rotation"]
                s = model["trs"]["scale"]

                # print('R_scene_pose,T_scene_pose',R_scene_pose,T_scene_pose)
                

                Mcad,Rcad,Scad = make_M_from_tqs(t, q, s)

                # print('Rcad,Scad,t',Rcad,Scad,t)

                R_no_scaling = np.matmul(R_scene_pose,Rcad)
                # R_with_scaling = np.matmul(R_no_scaling,Scad)

                # print('R_with_scaling',R_with_scaling)

                # print(R_scene_pose,np.array(t),T_scene_pose)
                T = np.matmul(R_scene_pose,np.array(t)) + T_scene_pose

                # invert axes to match our convention
                invert = np.array([[-1,0,0],[0,-1,0],[0,0,1.]])

                R_no_scaling = np.matmul(invert,R_no_scaling)
                # R_with_scaling = np.matmul(invert,R_with_scaling)
                T = np.matmul(invert,T)
                # print(T)

                infos = {}
                # infos["rot_mat"] = R_with_scaling.tolist()

                # print(scene)
                # print(frame)
                # print(model['catid_cad'])
                old_category = catid_to_cat[model['catid_cad']]

                if old_category not in cats:
                    continue

                assert old_category in cats,(dir_path_own_data + scene + '/masks/' + frame + '/' + object,old_category,np.sum(mask > 0))

                index = cats.index(old_category)
                category = cats_renamed[index]

                infos["rot_mat"] = R_no_scaling.tolist()
                infos["trans_mat"] = T.tolist()
                infos["scaling"] = s

                infos["orig_q"] = q
                infos["orig_t"] = t
                infos["orig_s"] = s

                infos["model"] = 'model/{}/{}/model_normalized.obj'.format(category,model['id_cad'])
                infos["catid"] =  model['catid_cad']
                infos["category"] =  category
                

                with open(dir_path_own_data + scene + '/info/' + frame + '/' + object.split('.')[0] + '.json','w') as file:
                    json.dump(infos,file)
            
        # print(df)
if __name__ == '__main__':
    main()