import torch
import numpy as np
import cv2
import itertools
import sys
import GPUtil

from pytorch3d.ops import knn_points,sample_points_from_meshes
from pytorch3d.structures import Meshes


def sample_points_inside_segmentation(segmentation_mask,N_points,device):
    counter = 0
    h,w = segmentation_mask.shape[:2]
    all_pixel = np.zeros((N_points,2),dtype=int)
    while counter < N_points:
        y = np.random.randint(h)
        x = np.random.randint(w)
        if (segmentation_mask[y,x] == 255).all():
            all_pixel[counter] = np.array([y,x])
            counter += 1

    return torch.from_numpy(all_pixel).to(device).to(float)


def create_all_4_indices(n_matches,config,correct_matches):
    if config["use_correct_matches"] == "False":

        if config["leave_out_matches"] == 0:
            n_keypoints_pose = min(n_matches,config["max_number_keypoints_for_pose"])
            if n_matches < config["max_number_keypoints_for_pose"]:
                all_4_indices = [list(range(n_keypoints_pose))]
            else:
                all_4_indices = []
                for combo in itertools.combinations(range(n_matches), n_keypoints_pose):
                    all_4_indices.append(list(combo))
                np.random.shuffle(all_4_indices)
        else:
            all_4_indices = []
            if n_matches <= 4:
                n_keypoints_pose = n_matches
                all_4_indices = [list(range(n_keypoints_pose))]
            else:
                if n_matches == 5 or n_matches == 6:
                    n_keypoints_pose = n_matches - 1
                elif n_matches == 7 or n_matches == 8:
                    n_keypoints_pose = n_matches - 2
                elif n_matches == 9 or n_matches == 10:
                    n_keypoints_pose = n_matches - 3
                elif n_matches > 10:
                   n_keypoints_pose = n_matches-config["leave_out_matches"]
                
                for combo in itertools.combinations(range(n_matches), n_keypoints_pose):
                    all_4_indices.append(list(combo))
                np.random.shuffle(all_4_indices)

    elif config["use_correct_matches"] == "True":
        if n_matches < config["max_number_keypoints_for_pose"]:
            all_4_indices = [list(range(n_matches))]
            n_keypoints_pose = n_matches
        else:
            correct_matches = correct_matches.squeeze().cpu().numpy()
            gt_correct_matches = []
            false_matches = []
            for i in range(correct_matches.shape[0]):
                if correct_matches[i] == True:
                    gt_correct_matches.append(i)
                elif correct_matches[i] == False:
                    false_matches.append(i)

            
            if len(gt_correct_matches) >= config["max_number_keypoints_for_pose"]:

                all_4_indices = []
                for combo in itertools.combinations(gt_correct_matches, config["max_number_keypoints_for_pose"]):
                    all_4_indices.append(list(combo))
                np.random.shuffle(all_4_indices)

                n_keypoints_pose = len(gt_correct_matches)
            else:
                all_4_indices = []
                for combo in itertools.combinations(false_matches,config["max_number_keypoints_for_pose"] - len(gt_correct_matches)):
                    all_4_indices.append(list(combo) + gt_correct_matches)
                np.random.shuffle(all_4_indices)
                
            n_keypoints_pose = config["max_number_keypoints_for_pose"]
    
    return all_4_indices,n_keypoints_pose



def compute_rendered_pixel(predicted_r,predicted_t,world_coordinates,f,w,h,sensor_width,already_batched):

    world_coordinates= world_coordinates.float()

    # print('world_coordinates',world_coordinates)

    if already_batched == False:
        world_coordinates = world_coordinates.unsqueeze(0).repeat(predicted_r.shape[0],1,1)

    # camera_coordinates = torch.transpose(torch.matmul(torch.inverse(predicted_r), torch.transpose(world_coordinates - predicted_t,-1,-2)  ),-1,-2)


    camera_coordinates = torch.transpose(torch.matmul(predicted_r,torch.transpose(world_coordinates,-1,-2)),-1,-2) + predicted_t

    # print('camera_coordinates',camera_coordinates)


    # now get pixel bearing by dividing by z/f
    pb = camera_coordinates / (camera_coordinates[:,:,2:]/f)

    mask = (camera_coordinates[:,:,2] > 0)
    
    px = - pb[:,:,0] * w/sensor_width + w/2
    py = - pb[:,:,1] * w/sensor_width + h/2


    pixel_rendered = torch.stack((py,px),dim=-1).to(int)

    pixel_rendered[~mask] = pixel_rendered[~mask] * 0 - 100000

    # print('pixel_rendered',pixel_rendered)

    return pixel_rendered


def stretch_3d_coordinates(world_coordinates,planes,stretching):
    print('world_coordinates.shape',world_coordinates.shape)
    print('planes',planes.shape)
    print('stretchign',stretching.shape)
    for i in range(planes.shape[0]):
        n = planes[i,:3]
        d = planes[i,3]

        y = torch.sign(torch.matmul(world_coordinates,n) - d)

        sign = torch.unsqueeze(y,dim=-1)
        sign = sign.repeat((1,1,3))
        n = n.unsqueeze(0).unsqueeze(0).repeat((world_coordinates.shape[0],world_coordinates.shape[1],1))
        tau = stretching[:,i].clone().unsqueeze(1).unsqueeze(2).repeat((1,world_coordinates.shape[1],3))
        world_coordinates = world_coordinates + tau / 2 * sign * n

    return world_coordinates



def compute_rendered_pixel_shape(predicted_r,predicted_t,predicted_stretching,planes,world_coordinates,f,w,h,sensor_width,already_batched):

    world_coordinates= world_coordinates.float()
    print('wc',world_coordinates.shape)
    if already_batched == False:
        world_coordinates = world_coordinates.unsqueeze(0).repeat(predicted_r.shape[0],1,1)

    print('after wc',world_coordinates.shape)
    stretched_coordinates = stretch_3d_coordinates(world_coordinates,planes,predicted_stretching)

    camera_coordinates = torch.transpose(torch.matmul(predicted_r,torch.transpose(stretched_coordinates,-1,-2)),-1,-2) + predicted_t


    # now get pixel bearing by dividing by z/f
    pb = camera_coordinates / (camera_coordinates[:,:,2:]/f)


    mask = (camera_coordinates[:,:,2] > 0)
    
    px = - pb[:,:,0] * w/sensor_width + w/2
    py = - pb[:,:,1] * w/sensor_width + h/2


    pixel_rendered = torch.stack((py,px),dim=-1).to(int)

    pixel_rendered[~mask] = pixel_rendered[~mask] * 0 - 1000

    return pixel_rendered


def compute_distances_reprojected_keypoints(reprojected_pixel,original_pixel):
    return torch.sum((reprojected_pixel - original_pixel)**2,dim=2)**0.5

def get_points_from_predicted_mesh(predicted_obj,elev,azim,n_points,device):


    vertices,faces,properties = predicted_obj
    vertices = vertices

    mesh = Meshes(verts=[vertices],faces=[faces[0]])
    
    sample_points = sample_points_from_meshes(mesh,n_points)

    return sample_points[0]

def get_points_from_predicted_mesh_shape(predicted_obj,elev,azim,n_points,device,predicted_stretching,planes):

    vertices,faces,properties = predicted_obj

    vertices = vertices.unsqueeze(0).repeat(predicted_stretching.shape[0],1,1)
    faces = faces[0].unsqueeze(0).repeat(predicted_stretching.shape[0],1,1)

    stretched_vertices = stretch_3d_coordinates(vertices,planes,predicted_stretching)

    mesh = Meshes(verts=stretched_vertices,faces=faces)
    
    sample_points = sample_points_from_meshes(mesh,n_points)

    return sample_points


def find_distance_point_clouds(segmentation_mask,pred_points,gt_points,device,config):

    gt_points = gt_points.unsqueeze(0).repeat(pred_points.shape[0],1,1).to(float)
    pred_points = pred_points.to(float)

    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # find distance between two pointclouds
    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points,lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)
    

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points,lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    avg_dist = (torch.mean(gt_to_pred_dists,dim=1) + torch.mean(pred_to_gt_dists,dim=1))/2
 
    sorted_pred_to_gt_dists,indices_pred_to_gt_dists = torch.sort(pred_to_gt_dists,descending=True,dim=1)
    sorted_gt_to_pred_dists,indices_gt_to_pred_dists = torch.sort(gt_to_pred_dists,descending=True,dim=1)

    index = int(gt_points.shape[0] * config["fraction_of_points_for_dist"])
    # index = int(gt_points.shape[1] * config["fraction_of_points_for_dist"])

    avg_dist_furthest = (torch.mean(sorted_gt_to_pred_dists[:,:index],dim=1) + torch.mean(sorted_pred_to_gt_dists[:,:index],dim=1))/2

    


    return avg_dist, avg_dist_furthest,indices_pred_to_gt_dists,indices_gt_to_pred_dists

def compute_selection_metric(predicted_r,predicted_t,points_from_predicted_mesh,f,w,h,config,segmentation_mask,pixel_inside_segmentation,device,world_coordinates_batch,pixels_real_original_image_batch):
    avg_dist_pointclouds,avg_dist_furthest,avg_dist_reprojected_keypoints,combined = 100000*torch.ones(predicted_r.shape[0]),100000*torch.ones(predicted_r.shape[0]),100000*torch.ones(predicted_r.shape[0]),100000*torch.ones(predicted_r.shape[0])
    
    if config["choose_best_based_on"] ==  "segmentation" or config["choose_best_based_on"] ==  "combined":
        pixel_coords_random_points_predicted_mesh = compute_rendered_pixel(predicted_r,predicted_t,points_from_predicted_mesh,f,w,h,config["sensor_width"],already_batched = False)
        avg_dist_pointclouds,avg_dist_furthest,_,_ = find_distance_point_clouds(segmentation_mask,pixel_coords_random_points_predicted_mesh,pixel_inside_segmentation,device,config)      
        
    if config["choose_best_based_on"] ==  "keypoints" or config["choose_best_based_on"] ==  "combined":
        pixel_coords_rendered = compute_rendered_pixel(predicted_r,predicted_t,world_coordinates_batch,f,w,h,config["sensor_width"],already_batched = True)
        # note is really distance between reprojected points
        distances_reprojected_keypoints_predicted = compute_distances_reprojected_keypoints(pixel_coords_rendered,pixels_real_original_image_batch)
        avg_dist_reprojected_keypoints = torch.mean(distances_reprojected_keypoints_predicted,dim=1)

    if config["choose_best_based_on"] ==  "combined":
        alpha = config["weighting_seg_vs_keypoints"]
        combined = alpha * avg_dist_furthest + (1-alpha) * avg_dist_reprojected_keypoints

    return avg_dist_pointclouds,avg_dist_furthest,avg_dist_reprojected_keypoints,combined



def compute_selection_metric_shape(predicted_r,predicted_t,predicted_stretching,plane,points_from_predicted_mesh,f,w,h,config,segmentation_mask,pixel_inside_segmentation,device,world_coordinates_batch,pixels_real_original_image_batch):
    avg_dist_pointclouds,avg_dist_furthest,avg_dist_reprojected_keypoints,combined = 100000*torch.ones(predicted_r.shape[0]),100000*torch.ones(predicted_r.shape[0]),100000*torch.ones(predicted_r.shape[0]),100000*torch.ones(predicted_r.shape[0])
    
    if config["choose_best_based_on"] ==  "segmentation" or config["choose_best_based_on"] ==  "combined":
        pixel_coords_random_points_predicted_mesh = compute_rendered_pixel(predicted_r,predicted_t,points_from_predicted_mesh,f,w,h,config["sensor_width"],already_batched = True)
        
        # pixel_coords_random_points_predicted_mesh = compute_rendered_pixel_shape(predicted_r,predicted_t,predicted_stretching,plane,points_from_predicted_mesh,f,w,h,config["pose_prediction"]["real_camera_sensor_width"],already_batched = False)
        avg_dist_pointclouds,avg_dist_furthest,_,_ = find_distance_point_clouds(segmentation_mask,pixel_coords_random_points_predicted_mesh,pixel_inside_segmentation,device,config)      
        
    if config["choose_best_based_on"] ==  "keypoints" or config["choose_best_based_on"] ==  "combined":
        pixel_coords_rendered = compute_rendered_pixel_shape(predicted_r,predicted_t,predicted_stretching,plane,world_coordinates_batch,f,w,h,config["sensor_width"],already_batched = True)
        # note is really distance between reprojected points
        distances_reprojected_keypoints_predicted = compute_distances_reprojected_keypoints(pixel_coords_rendered,pixels_real_original_image_batch)
        avg_dist_reprojected_keypoints = torch.mean(distances_reprojected_keypoints_predicted,dim=1)

    if config["choose_best_based_on"] ==  "combined":
        alpha = config["weighting_seg_vs_keypoints"]
        combined = alpha * avg_dist_furthest + (1-alpha) * avg_dist_reprojected_keypoints

    return avg_dist_pointclouds,avg_dist_furthest,avg_dist_reprojected_keypoints,combined


