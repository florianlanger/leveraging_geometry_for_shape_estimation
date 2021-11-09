import numpy as np

def add_minimal_pose_info_dict(indices,predicted_r,predicted_t,avg_dist_pointclouds,avg_dist_furthest,avg_dist_reprojected_keypoints,combined,F1):

    pose_dict = {}
    pose_dict["indices"] = list(indices)
    pose_dict["predicted_r"] = [predicted_r[0,:].tolist(),predicted_r[1,:].tolist(),predicted_r[2,:].tolist()]
    pose_dict["predicted_t"] = predicted_t[0,:].tolist()

    pose_dict["avg_dist_pointclouds"] = float(avg_dist_pointclouds.item())
    pose_dict["avg_dist_furthest"] = float(avg_dist_furthest.item())
    pose_dict["avg_dist_reprojected_keypoints"] = float(avg_dist_reprojected_keypoints.item())
    pose_dict["combined"] = float(combined.item())

    if F1 != None:
        pose_dict["F1"] = float(F1.item())
    return pose_dict


def add_minimal_pose_info_dict_shape(indices,predicted_r,predicted_t,predicted_stretching,avg_dist_pointclouds,avg_dist_furthest,avg_dist_reprojected_keypoints,combined,F1):

    pose_dict = {}
    pose_dict["indices"] = list(indices)
    pose_dict["predicted_r"] = [predicted_r[0,:].tolist(),predicted_r[1,:].tolist(),predicted_r[2,:].tolist()]
    pose_dict["predicted_t"] = predicted_t[0,:].tolist()
    pose_dict["predicted_stretching"] = predicted_stretching.tolist()

    pose_dict["avg_dist_pointclouds"] = float(avg_dist_pointclouds.item())
    pose_dict["avg_dist_furthest"] = float(avg_dist_furthest.item())
    pose_dict["avg_dist_reprojected_keypoints"] = float(avg_dist_reprojected_keypoints.item())
    pose_dict["combined"] = float(combined.item())

    if F1 != None:
        pose_dict["F1"] = float(F1.item())

    return pose_dict


def get_information_best_pose(all_pose_information,which_metric,max_or_min,index):

    # print('deactivate this in normal')
    if which_metric not in ['init_pose','lower_bound','upper_bound']:
        # values = [all_pose_information[i][which_metric] for i in range(3,len(all_pose_information))]
        values = [all_pose_information[i][which_metric] for i in range(len(all_pose_information))]

        indices_sorted = np.array(values).argsort()

        if max_or_min == 'max':
            index = indices_sorted[-1*index]
        elif max_or_min == 'min':
            index = indices_sorted[index-1]

    # order_f1=None
    # if "F1" in all_pose_information[0]:
    #     order_f1 = ''
    #     for name,selector in zip(['R','P','F'],['avg_dist_reprojected_keypoints','avg_dist_pointclouds','F1']):
    #         values = [all_pose_information[i][selector] for i in range(len(all_pose_information))]
    #         order_f1 += get_index_values(values,index,name)

    pose_dict = all_pose_information[index]
    return pose_dict

    # general_information = nn_dict["info_all_poses"]
    # subset_4_indices = pose_dict["indices"]


    # # load from all poses info
    # world_coordinates_matches_all = torch.Tensor(general_information["world_coordinates_matches"]).to(device)
    # real_bearings_all = torch.Tensor(general_information["real_bearings"]).to(device)
    # indices_valid_matches_all = torch.Tensor(general_information["indices_valid_matches"]).to(device)
    # pixels_real_original_image_all = torch.Tensor(general_information["pixels_real_original_image"]).to(device).to(int)
    
    # avg_dist_pointclouds = pose_dict['avg_dist_pointclouds']
    # avg_dist_furthest = pose_dict['avg_dist_furthest']
    # avg_dist_reprojected_keypoints = pose_dict['avg_dist_reprojected_keypoints']
    # combined = pose_dict['combined']

    # world_coordinates_matches = world_coordinates_matches_all[subset_4_indices]
    # pixels_real_original_image = pixels_real_original_image_all[subset_4_indices]
    # real_bearings = real_bearings_all[subset_4_indices]
    # indices_valid_matches = indices_valid_matches_all[subset_4_indices]
    # pixels_rendered = [general_information["pixels_rendered"][int(index)] for index in subset_4_indices]
    # pixels_real = [general_information["pixels_real"][int(index)] for index in subset_4_indices]


    # predicted_r = torch.Tensor(pose_dict["predicted_r"]).to(device).unsqueeze(0)
    # predicted_t = torch.Tensor(pose_dict["predicted_t"]).to(device).unsqueeze(0)

    # if "predicted_stretching" in pose_dict:
    #     predicted_stretching = torch.Tensor([pose_dict["predicted_stretching"]]).to(device)
    #     return world_coordinates_matches,real_bearings,indices_valid_matches,pixels_rendered,pixels_real,pixels_real_original_image,predicted_r,predicted_t,avg_dist_pointclouds,avg_dist_furthest,avg_dist_reprojected_keypoints,combined, predicted_stretching, index
    # else:
    #     return world_coordinates_matches,real_bearings,indices_valid_matches,pixels_rendered,pixels_real,pixels_real_original_image,predicted_r,predicted_t,avg_dist_pointclouds,avg_dist_furthest,avg_dist_reprojected_keypoints,combined, order_f1