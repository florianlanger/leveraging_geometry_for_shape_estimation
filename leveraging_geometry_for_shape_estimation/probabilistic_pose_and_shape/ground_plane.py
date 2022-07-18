import json
import numpy as np
import os

# def catid_to_cat_name(catid):
#     top = {}                                                                                                                                                                                                                                                                                      
#     top["03211117"] = "display"
#     top["04379243"] = "table"
#     top["02808440"] = "bathtub"
#     top["02747177"] = "bin"
#     top["04256520"] = "sofa"
#     top["03001627"] = "chair"
#     top["02933112"] = "cabinet"
#     top["02871439"] = "bookshelf"
#     top["02818832"] = "bed"
#     return top[catid]

def init_Ts(xs,ys,zs):

    xs = np.linspace(xs[0],xs[1],xs[2])
    ys = np.linspace(ys[0],ys[1],ys[2])
    zs = np.linspace(zs[0],zs[1],zs[2])
    x, y, z = np.meshgrid(xs,ys,zs, indexing='ij')
    Ts = np.stack([x.flatten(),y.flatten(),z.flatten()], axis=1)

    return Ts

def get_model_to_infos_scannet():
    with open('/scratches/octopus_2/fml35/datasets/scannet/scan2cad_annotations/full_annotations.json','r') as f:
        annos = json.load(f)

    top = {}
    top["03211117"] = "display"
    top["04379243"] = "table"
    top["02808440"] = "bathtub"
    top["02747177"] = "bin"
    top["04256520"] = "sofa"
    top["03001627"] = "chair"
    top["02933112"] = "cabinet"
    top["02871439"] = "bookshelf"
    top["02818832"] = "bed"

    infos = {}

    for scene in annos:
        for model in scene['aligned_models']:
            if model['catid_cad'] in top:
                name = top[model['catid_cad']] + '_' + model['id_cad']
                infos_model = {}
                infos_model["bbox"] = model["bbox"]
                infos_model["center"] = model["center"]
                if name in model:
                    assert infos_model == infos[name], (infos_model,infos[name])
                else:
                    infos[name] = infos_model
    return infos

def get_model_to_infos_scannet_just_id():
    with open('/scratches/octopus_2/fml35/datasets/scannet/scan2cad_annotations/full_annotations.json','r') as f:
        annos = json.load(f)

    top = {}
    top["03211117"] = "display"
    top["04379243"] = "table"
    top["02808440"] = "bathtub"
    top["02747177"] = "bin"
    top["04256520"] = "sofa"
    top["03001627"] = "chair"
    top["02933112"] = "cabinet"
    top["02871439"] = "bookshelf"
    top["02818832"] = "bed"

    infos = {}

    for scene in annos:
        for model in scene['aligned_models']:
            if model['catid_cad'] in top:
                name = model['id_cad']
                infos_model = {}
                infos_model["bbox"] = model["bbox"]
                infos_model["center"] = model["center"]
                infos_model["category"] = top[model['catid_cad']]
                if name in model:
                    assert infos_model == infos[name], (infos_model,infos[name])
                else:
                    infos[name] = infos_model
    return infos

def get_model_to_infos_future3d():
    dir_shapes = '/scratches/octopus/fml35/datasets/3d_future/3D-FUTURE-model_reformatted/model/'

    infos = {}
    for cat in os.listdir(dir_shapes):
        for id in os.listdir(dir_shapes + cat):
            name = cat + '_' + id
            infos_single = {}
            infos_single['bbox'] = [0.5,0.5,0.5]
            infos_single['center'] = [0,0,0]
            infos[name] = infos_single

    return infos

def get_model_to_infos(dataset_name):
    if dataset_name == 'scannet':
        return get_model_to_infos_scannet()
    elif dataset_name == 'future3d':
        return get_model_to_infos_future3d()


def sample_Ts_ground_plane(R,height_object_center_above_ground,ground_plane_limits,category):
    # sp_rot = scipy_rot.from_matrix(R)
    # tilt,azim,elev = sp_rot.as_euler('zyx',degrees=True)
    # print('tilt,azim,elev',tilt,azim,elev)

    # if np.abs(azim) > 45:
    #     R_transform = scipy_rot.from_euler('zyx',[tilt,0,elev], degrees=True).as_matrix()
    # else:
    #     R_transform = scipy_rot.from_euler('zyx',[tilt,180,elev], degrees=True).as_matrix()
    # xs = (-0.5,0.5,3)
    # ys = (-1.8,-0.8,3) # y controls height, does it correspond exactyl to height above ground, no this is to object center need to subratct 0.5 times scale z time value 3d gt_bbox z
    # zs = (2.0,4.5,3)
    xs = ground_plane_limits["xs"]
    ys = (0,0,1) # y controls height, does it correspond exactyl to height above ground, no this is to object center need to subratct 0.5 times scale z time value 3d gt_bbox z
    zs = ground_plane_limits["zs"]
    Ts_ground = init_Ts(xs,ys,zs)
    Ts_transformed = np.matmul(R,Ts_ground.T).T

    camera_heights = np.array(ground_plane_limits["camera_heights"])

    if category == 'display':
        camera_heights = camera_heights - 0.8

    all_Ts = []
    # print(Ts_transformed)
    for camera_height in camera_heights:
        # print((camera_height - height_object_center_above_ground) * np.matmul(R,np.array([[0,-1,0]]).T).T)
        all_Ts.append(Ts_transformed + (camera_height - height_object_center_above_ground) * np.matmul(R,np.array([[0,-1,0]]).T).T )

    all_Ts = np.concatenate(all_Ts)
    return all_Ts,Ts_ground

def filter_Ts(Ts,f,sw,w,h,pred_bbox=None):
    mask_1 = Ts[:,2] > 0.
    pb = Ts / (Ts[:,2:]/f)
    px = - pb[:,0] * w/sw + w/2
    py = - pb[:,1] * w/sw + h/2

    # mask_2 = (px >= 0) & (px < w)
    # mask_3 = (py >= 0) & (py < h)
    # bb_width = pred_bbox[2] - pred_bbox[0]
    # bb_height = pred_bbox[3] - pred_bbox[1]

    # mask_2 = (px >= pred_bbox[0] - 0.2*bb_width) & (px < 1.2*w)
    mask_2 = (px >= -0.2*w) & (px < 1.2*w)
    mask_3 = (py >= -0.2*h) & (py < 1.2*h)
    

    total_mask = mask_1 & mask_2 & mask_3
    
    Ts = Ts[total_mask,:]
    return Ts