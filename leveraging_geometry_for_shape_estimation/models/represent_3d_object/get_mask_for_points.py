import numpy as np
import torch
from tqdm import tqdm
import os

from leveraging_geometry_for_shape_estimation.models.represent_3d_object.extract_points_from_models import find_edge_points,load_points_normals,sample_points_from_lines,repair_mesh,plot_points,different_normals_around_points,vis_representative_normals
from leveraging_geometry_for_shape_estimation.models.represent_3d_object.moller_trumbore import inside_mesh, moller_trumbore,normalize


from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj,save_ply
import pytorch3d
import k3d
import trimesh




def get_rotations(device):

    rotation_path = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_117_scannet_models/models/rotations/R_T_torch.npz'

    R_and_T = np.load(rotation_path)
    R_mesh = torch.from_numpy(R_and_T["R"]).to(device)
    T_mesh = torch.from_numpy(R_and_T["T"]).to(device)
    R_mesh = torch.inverse(R_mesh)

    elev = [0.0,15.0,30.0,45.0]
    azim = [0.0,22.5,45.0,67.5,90.0,112.5,135.0,157.5,180.0,202.5,225.0,247.5,270.0,292.5,315.0,337.5]

    return R_mesh,T_mesh,elev,azim



def reshape_distances(intersects,smaller_than_distance_face,n_points):
    max_len = torch.max(torch.bincount(intersects[:, 0]))
    point_dict_counter = {}
    smaller_than_distance_reshaped = torch.ones((n_points,max_len))

    # print('intersects',intersects.shape)
    # print('smaller_than_distance_face',smaller_than_distance_face.shape)

    for k in range(smaller_than_distance_face.shape[0]):
        index = int(intersects[k, 0].item())

        if index not in point_dict_counter:
            smaller_than_distance_reshaped[index, 0] = smaller_than_distance_face[k]
            point_dict_counter[index] = 1
        else:
            # print('yes increase')
            smaller_than_distance_reshaped[index, point_dict_counter[index]] = smaller_than_distance_face[k]
            point_dict_counter[index] += 1
    
    return smaller_than_distance_reshaped

def reshape_distances_v2(intersects,smaller_than_distance_face,n_points):
    max_len = torch.max(torch.bincount(intersects))
    point_dict_counter = {}
    smaller_than_distance_reshaped = torch.ones((n_points,max_len))

    # print('intersects',intersects.shape)
    # print('smaller_than_distance_face',smaller_than_distance_face.shape)

    for k in range(smaller_than_distance_face.shape[0]):
        index = int(intersects[k].item())

        if index not in point_dict_counter:
            smaller_than_distance_reshaped[index, 0] = smaller_than_distance_face[k]
            point_dict_counter[index] = 1
        else:
            # print('yes increase')
            smaller_than_distance_reshaped[index, point_dict_counter[index]] = smaller_than_distance_face[k]
            point_dict_counter[index] += 1
    
    return smaller_than_distance_reshaped

def get_name(total_index,elev,azim):
    elev_index = total_index // len(azim)
    # NOTE: complicated formula because of old convention
    # azim_index = (len(azim) - total_index % len(azim)) % len(azim)
    azim_index = total_index % len(azim)
    elev_current = str(int(elev[elev_index])).zfill(3)
    azim_current = str(np.round(azim[azim_index],1)).zfill(3)

    name = 'elev_{}_azim_{}.npy'.format(elev_current,azim_current)
    return name

def find_intersecting_faces(points,faces,v_mesh):
    # for each point find faces that ray from origing to point intersects
    ray_o = points * 0
    ray_d = normalize(points)
    faces_muller = v_mesh[faces]
    print('points',points.shape)
    print('faces',faces.shape)
    print('v_mesh',v_mesh.shape)
    print('ray_o',ray_o.shape)
    print('ray_d',ray_d.shape)
    print('faces_muller',faces_muller.shape)

    print('Debug transpose ??')
    faces_muller = torch.transpose(faces_muller,1,2)

    u, v, t = moller_trumbore(ray_o, ray_d, faces_muller)  # (n_rays, n_faces, 3)
    print('u,v,t',u.shape,v.shape,t.shape)
    intersection = ((t >= 0.0) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0)).bool()
    print('intersection',intersection.shape)
    intersects = torch.nonzero(intersection)
    print('intersects',intersects.shape)
    return intersects


def find_intersecting_faces_trimesh(points,faces,v_mesh,device):
    # for each point find faces that ray from origing to point intersects
    ray_o = points * 0
    ray_d = normalize(points)
    # faces_muller = v_mesh[faces]

    mesh = trimesh.Trimesh(vertices=v_mesh.cpu().numpy(), faces=faces.cpu().numpy())

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_o.cpu().numpy(),
        ray_directions=ray_d.cpu().numpy())

    # print('index ray',index_ray[:10])
    # print('locations',locations.shape)
    points_repeated_for_locations = points[index_ray]

    locations = torch.from_numpy(locations).to(points.device)
    # points_repeated_for_locations = torch.from_numpy(points_repeated_for_locations).to(device)
    index_ray = torch.from_numpy(index_ray).to(device).long()

    distances_to_orig_face_centers = torch.norm(locations, dim=1)
    distances_to_orig_points = torch.norm(points_repeated_for_locations, dim=1)

  
    # print('points_repeated_for_locations',points_repeated_for_locations.shape)
    # print()
    return distances_to_orig_face_centers,distances_to_orig_points,index_ray

def get_masks_single_orientation(R,T,vertices,faces,points_sparse,gamma,device):

    # point_index = [11]

    

    # transform mesh and points
    v_mesh = torch.transpose(torch.matmul(R[:,:],torch.transpose(vertices,0,1)),0,1) + T
    points = torch.transpose(torch.matmul(R[:,:],torch.transpose(points_sparse,0,1)),0,1) + T

    # print(points)
    # intersects = find_intersecting_faces(points,faces,v_mesh)
    # involved_points = v_mesh[faces[intersects[:, 1]]]
    # center_intersects = torch.mean(involved_points, dim=1)  # (n_faces, 3)
    # print('after center intersects',center_intersects.shape)
    # points_sparse_same_as_intersects = points[intersects[:, 0]]
    # distances_to_orig_face_centers = torch.norm(center_intersects, dim=1)
    # distances_to_orig_points = torch.norm(points_sparse_same_as_intersects, dim=1)



    distances_to_orig_face_centers,distances_to_orig_points,intersects = find_intersecting_faces_trimesh(points,faces,v_mesh,device)

    # print('v_mesh',v_mesh.shape)
    # print('faces',faces.shape)
    # print('intersects[[intersects[:, 0] == point_index[0]]]',intersects[[intersects[:, 0] == point_index[0]]])
    # # print('faces[intersects[:, 0] == point_index[0]]',faces[intersects[:, 0] == point_index[0]])
    

    # # faces_involved = faces[[56038,87597,219841],:]
    # print('faces_involved',faces_involved)
    # print('v_mesh[faces_involved]',v_mesh[faces_involved])
    # print(intersects[:100])

    # print('faces',faces.shape)
    # print('faces[intersects[:, 1]]',faces[intersects[:, 1]])

    # print('v mesh of face',v_mesh[faces[32]])

    # find distance from origin to center of intersected faces, NOTE this is not super precise as doesnt necessarily take the closest point

    # print(distances_to_orig_face_centers[intersects[:, 0] == point_index[0]])
    # print(distances_to_orig_points[intersects[:, 0] == point_index[0]])

    # print('intersects',intersects.shape)
    # print('involve points',involved_points.shape)
    # print('center intersects',center_intersects.shape)
    # print('points sparse same as intersects',points_sparse_same_as_intersects.shape)
    # print('distances to orig face centers',distances_to_orig_face_centers.shape)
    # print('distances to orig points',distances_to_orig_points.shape)




    # print('distance_to_orig_points',distances_to_orig_points.shape)

    smaller_than_distance_face = distances_to_orig_face_centers > distances_to_orig_points - gamma
    # print('after smaller than dist')
    # reshape distances
    smaller_than_distance_reshaped = reshape_distances_v2(intersects,smaller_than_distance_face,points_sparse.shape[0])
    # print('smaller_than_distance_reshaped',smaller_than_distance_reshaped[point_index])

    mask = torch.all(smaller_than_distance_reshaped,dim=1)

    # plot = k3d.plot()
    # plot += k3d.points(points.cpu().numpy(), point_size=0.01, shader="flat",color=0xffff)
    # plot += k3d.points(points[mask].cpu().numpy(), point_size=0.01, shader="flat",color=0xF62006)
    # # plot += k3d.mesh(v_mesh.cpu().numpy(), faces.cpu().numpy(), textures=None, shader="flat")

    # # print(mask[point_index])
    # # yellow 
    # plot += k3d.points(points[point_index].cpu().numpy(), point_size=0.01, shader="flat",color=0xF6E706)
    # # print('points[point_index].cpu().numpy()',points[point_index].cpu().numpy())
    # # # pink
    # # # print('involved_points[intersects[:, 0] == point_index[0]',involved_points[intersects[:, 0] == point_index[0]])
    # # # plot += k3d.points(involved_points[intersects[:, 0] == point_index[0]].view(-1,3).cpu().numpy(), point_size=0.01, shader="flat",color=0xF718E3)
    # # print('involved_points[intersects[:, 0] == point_index[0]].view(-1,3).cpu().numpy()',involved_points[intersects[:, 0] == point_index[0]].view(-1,3).cpu().numpy())
    # # # print('v_mesh',v_mesh.shape)
    # # # print('faces',faces[0].shape)
    # # # borwn
    # # # plot += k3d.points(center_intersects[intersects[:, 0] == point_index[0]].cpu().numpy(), point_size=0.01, shader="flat",color=0x07F3A3A)
    # # print('center_intersects[intersects[:, 0] == point_index[0]].cpu()',center_intersects[intersects[:, 0] == point_index[0]].cpu().numpy())
    # plot.display()


    return mask




def get_masks_all_orientations(R_mesh,T_mesh,elev,azim,model_path,point_norm_path,device,gamma):

    vertices,faces,textures = load_obj(model_path ,load_textures=False,device=device)
    faces = faces[0].long()

    point_norm = np.load(point_norm_path)
    points_sparse = torch.from_numpy(point_norm['points']).to(device)

    if points_sparse.shape[0] == 0:
        return {},None

    else:
        mask_dict = {}

        total_index = 0
        for i in range(64):
            mask = get_masks_single_orientation(R_mesh[total_index,:,:],T_mesh[total_index,:],vertices,faces,points_sparse,gamma,device)
            
            name = get_name(total_index,elev,azim)
            # print(name)
            mask_dict[name] = mask.cpu().numpy()
            total_index += 1

        return mask_dict,point_norm


def vis_masks(mask_dict,point_norm,vis_dir,model):
    
    furthest_points = point_norm['points']
    representative_normals = point_norm['normals']

    for key in mask_dict:
        mask = mask_dict[key]
        vis_points = vis_representative_normals(torch.from_numpy(furthest_points[mask,:]),torch.from_numpy(representative_normals[mask,:,:]),in_notebook=False)
        out_path = vis_dir + model.split('.')[0] + '_' + key.rsplit('.',1)[0] + '.ply'
        save_ply(out_path,vis_points.cpu())

def main():

    exp_dir = '/scratch/fml35/datasets/shapenet_v2/ShapeNetRenamed/representation_points_and_normals/exp_05_150_edge_150_random/'

    input_dir = exp_dir + 'points_and_normals/'
    output_dir = exp_dir + 'masks/'
    output_dir_vis = exp_dir + 'masks_vis/'

    os.mkdir(output_dir)
    os.mkdir(output_dir_vis)

    model_3d_dir = '/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/model/'

    gamma = 0.03

    device = torch.device("cuda:0")
    R_mesh,T_mesh,elev,azim = get_rotations(device)

    for model in tqdm(sorted(os.listdir(input_dir))):

        # if model != 'bathtub_4a6ed9c61b029d15904e03fa15bd73ec.npz':
        #     continue

        model_path = model_3d_dir + model.split('_')[0] + '/' + model.split('_')[1].split('.')[0] + '/model_normalized.obj'
        point_norm_path = input_dir + model

        mask_dict,point_norm = get_masks_all_orientations(R_mesh,T_mesh,elev,azim,model_path,point_norm_path,device,gamma) 
        np.savez(output_dir + model, **mask_dict)

        if np.random.rand() > 0.99 and point_norm != None:
            vis_masks(mask_dict,point_norm,output_dir_vis,model)

            


if __name__ == '__main__':
    main()