
import json
from cv2 import norm
import torch
import numpy as np
import cv2
import time

from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.dataset_points_v2 import Dataset_points
# from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.main_v2 import process_config
# from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.batch_sampler import get_batch_from_dataset
# from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.visualisation.visualisation_points_and_normals import plot_points_preds_normals

def draw_normals_on_image(points,config,size_expand=1):

    mask_channel = (torch.abs(points[:,2] - 0) < 0.1) | (torch.abs(points[:,2] - 4) < 0.1)

    points_channel = points[mask_channel,:].numpy()
    img_size = config['data']['img_size']


    mask = np.all((indices >= np.zeros(2)) & (indices < np.array([img_size[0],img_size[1]]) -1 ),axis=1)
    indices = indices[mask,:]

    img_black = np.zeros([img_size[1],img_size[0],3],dtype=np.uint8)

    normals = np.round((points_channel[:,3:6] / 2. - 0.5) * (-255))

    assert np.min(normals) >= 0 and np.max(normals) < 256
    normals = normals.astype(np.uint8)

    normals = np.repeat(normals,(2*size_expand+1)**2,axis=0)
    normals = normals[mask,:]

    img_black[indices[:,1],indices[:,0],:] = normals

    return img_black


def get_pixel_ids_and_batch_id(indices,mask,inputs):
    pixel = indices[mask,:]
    batch_ids = inputs[mask,:].to(torch.long)
    batch_ids = batch_ids[:,-1]

    assert pixel[:,0].min() >= 0 and pixel[:,0].max() < 160
    assert pixel[:,1].min() >= 0 and pixel[:,1].max() < 120, (pixel[:,1].min(),pixel[:,1].max())
    return pixel,batch_ids

def add_batch_id_to_input(inputs,device):
    batch_id = torch.arange(inputs.shape[0]).to(device)
    batch_id = batch_id.view(inputs.shape[0],1,1)
    batch_id = batch_id.repeat(1,inputs.shape[1],1)
    inputs = torch.cat((inputs,batch_id),dim=2)
    return inputs

def get_2d_and_3d_masks(inputs,indices,device,target_size):
    mask_channel_bbox = (torch.abs(inputs[:,:,2] - 1) < 0.1)
    mask_channel_2d = (torch.abs(inputs[:,:,2] - 0) < 0.1) | (torch.abs(inputs[:,:,2] - 4) < 0.1)
    mask_channel_3d = (torch.abs(inputs[:,:,2] - 2) < 0.1) | (torch.abs(inputs[:,:,2] - 3) < 0.1)

    mask_in_img = torch.all((indices >= torch.zeros(2,device=device)) & (indices < target_size-1),axis=2)

    mask_channel_bbox = mask_channel_bbox & mask_in_img
    mask_channel_2d = mask_channel_2d & mask_in_img 
    mask_channel_3d = mask_channel_3d & mask_in_img

    return mask_channel_bbox,mask_channel_2d,mask_channel_3d


def add_input_to_vgg_inputs(inputs_vgg,inputs,mask,index_start_read,index_end_read,index_start_write,index_end_write,batch_ids,pixel):
    normals_2d = inputs[mask,:]
    normals_2d = normals_2d[:,index_start_read:index_end_read]
    inputs_vgg[batch_ids,pixel[:,1],pixel[:,0],index_start_write:index_end_write] = normals_2d
    return inputs_vgg

def add_bbox_to_vgg_inputs(inputs_vgg,inputs,mask,index_start_read,index_end_read,index_start_write,index_end_write,batch_ids,pixel):
    inputs_vgg[batch_ids,pixel[:,1],pixel[:,0],index_start_write:index_end_write] = inputs_vgg[batch_ids,pixel[:,1],pixel[:,0],index_start_write:index_end_write] * 0 + 1
    return inputs_vgg

def convert_input_to_vgg_input(inputs):
    gpu_n = inputs.get_device()
    device = torch.device("cuda:{}".format(gpu_n))
    inputs_vgg = -2 * torch.ones((inputs.shape[0],120,160,12),device=device)

    target_size = torch.Tensor([160,120]).to(device)

    # add batch id to the input
    inputs = add_batch_id_to_input(inputs,device) 


    # multiply so that indices correct img size
    indices = inputs[:,:,:2] * target_size
    indices = torch.round(indices).to(torch.long)

    mask_channel_bbox,mask_channel_2d,mask_channel_3d = get_2d_and_3d_masks(inputs,indices,device,target_size)

    pixel_bbox,batch_ids_bbox = get_pixel_ids_and_batch_id(indices,mask_channel_bbox,inputs)
    pixel_2d,batch_ids_2d = get_pixel_ids_and_batch_id(indices,mask_channel_2d,inputs)
    pixel_3d,batch_ids_3d = get_pixel_ids_and_batch_id(indices,mask_channel_3d,inputs)


    inputs_vgg = add_input_to_vgg_inputs(inputs_vgg,inputs,mask_channel_2d,3,6,0,3,batch_ids_2d,pixel_2d)
    inputs_vgg = add_input_to_vgg_inputs(inputs_vgg,inputs,mask_channel_2d,6,7,3,4,batch_ids_2d,pixel_2d)
    inputs_vgg = add_input_to_vgg_inputs(inputs_vgg,inputs,mask_channel_3d,3,6,4,7,batch_ids_3d,pixel_3d)
    inputs_vgg = add_input_to_vgg_inputs(inputs_vgg,inputs,mask_channel_3d,6,7,7,8,batch_ids_3d,pixel_3d)
    # 3d camera coordinates
    inputs_vgg = add_input_to_vgg_inputs(inputs_vgg,inputs,mask_channel_3d,7,10,8,11,batch_ids_3d,pixel_3d)
    # bbox 2d
    inputs_vgg = add_bbox_to_vgg_inputs(inputs_vgg,inputs,mask_channel_bbox,0,1,11,12,batch_ids_bbox,pixel_bbox)
    # 2d rgb colors
    # inputs_vgg = add_input_to_vgg_inputs(inputs_vgg,inputs,mask_channel_2d,10,13,12,15,batch_ids_2d,pixel_2d)
    inputs_vgg = torch.permute(inputs_vgg,(0,3,1,2))

    return inputs_vgg





def vis_depth(points_channel,index):

    empty_img = np.zeros((120,160,3))
    mask = points_channel[:,:,index] > -1.5

    max_depth = 5000
    depth = (points_channel[:,:,index] * 1000) / max_depth
    depth = np.clip(depth,0,1)
    normalised_depth = (1 - depth)
    values = np.uint8(normalised_depth * 255)
    depth = cv2.applyColorMap(values, cv2.COLORMAP_JET)

    empty_img[mask] = depth[mask]

    return empty_img

def vis_normals(points_channel,index_start,index_end):
    empty_img = np.zeros((120,160,3))
    mask = points_channel[:,:,index_start:index_end] > -1.5

    normals = np.round((points_channel[:,:,index_start:index_end] / 2. - 0.5) * (-255))
    assert np.min(normals[mask]) >= 0 and np.max(normals[mask]) < 256, (np.min(normals),np.max(normals))
    normals = normals.astype(np.uint8)

    empty_img[mask] = normals[mask]
    return empty_img

def vis_rgb(points_channel,index_start,index_end):
    empty_img = np.zeros((120,160,3))
    mask = points_channel[:,:,index_start:index_end] > -1.5

    rgb = np.round((points_channel[:,:,index_start:index_end] * 255))
    rgb = rgb.astype(np.uint8)
    empty_img[mask] = rgb[mask]
    return empty_img

def vis_bbox(points_channel,index_start,index_end):
    empty_img = np.zeros((120,160,3))
    mask = points_channel[:,:,index_start:index_end] > -1.5
    mask = mask.squeeze()

    rgb = np.round((points_channel[:,:,index_start:index_end] * 255))
    gray = rgb.astype(np.uint8)
    # gray to rgb
    rgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

    empty_img[mask,:] = rgb[mask,:]
    return empty_img

def visualise_vgg_inputs(vgg_inputs):

    points_channel = vgg_inputs[0].cpu().numpy()

    # normals
    normals_2d = vis_normals(points_channel,0,3)
    normals_3d = vis_normals(points_channel,4,7)

    # depth
    depth_2d = vis_depth(points_channel,3)
    depth_3d = vis_depth(points_channel,7)

    # rgb colors
    bbox = vis_bbox(points_channel,11,12)

    # rgb colors
    rgb_2d = vis_rgb(points_channel,12,15)


    combined = cv2.hconcat([normals_2d,normals_3d,depth_2d,depth_3d,rgb_2d,bbox])
    combined = combined[:,:,::-1]

    cv2.imwrite('/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/probabilistic_pose_and_shape/train_pose_classifier_from_lines/combined_vgg.png',combined)


def save_images(images,out_path):

    combined = combine_images(images)
    combined = cv2.cvtColor(combined,cv2.COLOR_BGR2RGB)
    cv2.imwrite(out_path,combined)

def combine_images(images):

    combined_top = cv2.hconcat([images[0],images[1],images[2],images[3]])
    combined_middle = cv2.hconcat([images[4],images[5],images[6],images[7]])
    combined_bottom = cv2.hconcat([images[8],images[9],images[10],images[11]])

    combined = cv2.vconcat([combined_top,combined_middle,combined_bottom])

    combined_models = cv2.vconcat([images[14],images[15],images[16]])
    combined_models = cv2.resize(combined_models,(360,1080))

    combined_renders = cv2.vconcat([images[12],images[13],images[17]])

    combined = cv2.hconcat([combined,combined_renders,combined_models])
    return combined


if __name__ == '__main__':
    config_path = '/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/probabilistic_pose_and_shape/train_pose_classifier_from_lines/config.json'
    with open(config_path,'r') as file:
        config = json.load(file)

    config['data']['input_rgb'] = True
    config = process_config(config)
    print('use rgb')

    print('DONT ADD COLOR NEED TO ACTIVATE')
    
    kind = 'val'
    print('for vis need to disable transpose')

    dataset = Dataset_points(config,'val')

    sample_just_classifier = False
    index_infos = [0,1,2,3] * 20


    data = get_batch_from_dataset(dataset,index_infos,sample_just_classifier)
    inputs, targets, extra_infos = data

    t1 = time.time()

    inputs_vgg = convert_input_to_vgg_input(inputs)
    inputs_vgg = torch.permute(inputs_vgg,(0,2,3,1))
    t2 = time.time()
    print('convert_input_to_vgg_input took',t2-t1,'seconds')
    visualise_vgg_inputs(inputs_vgg)
    print('inputs',inputs.shape)


    probabilities = np.zeros(len(index_infos))
    labels = np.zeros(len(index_infos))
    correct = np.zeros(len(index_infos))
    t_pred = np.zeros((len(index_infos),3))
    s_pred = np.zeros((len(index_infos),3))
    r_pred = np.zeros((len(index_infos),3))

    t_correct = np.zeros((len(index_infos),3))
    s_correct = np.zeros((len(index_infos),3))
    r_correct = np.zeros((len(index_infos),3))

    images,info_vis_3d = plot_points_preds_normals(inputs.cpu(), labels,probabilities,correct,t_pred,s_pred,r_pred,t_correct,s_correct,r_correct,config,extra_infos,kind,no_render=True)
    out_path_save_comparison = '/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/probabilistic_pose_and_shape/train_pose_classifier_from_lines/comparison_'+kind+'.png'
    save_images(images[0],out_path_save_comparison)