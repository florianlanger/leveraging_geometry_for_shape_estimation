import torch
import cv2
import numpy as np
import random

from leveraging_geometry_for_shape_estimation.vis_pose.vis_pose import plot_polygons,plot_polygons_v2, load_mesh, render_mesh, overlay_rendered_image, plot_matches_individual_color_no_endpoints


from probabilistic_formulation.factors.factors_T.factors_lines_multiple_T import get_factor_reproject_lines_multiple_T_threshold_map_single_3d_line_v2
from probabilistic_formulation.tests.test_reproject_lines import draw_factors

from leveraging_geometry_for_shape_estimation.segmentation.meshrcnn_vis_tools import draw_boxes
from leveraging_geometry_for_shape_estimation.utilities.write_on_images import draw_text_block
# from probabilistic_formulation.tests.test_factors import get_depth_and_bearings_real


def format_infos_v2(visualisation_info):
    infos_text = []

    running_sum = 0

    for i in range(visualisation_info['all_indices_2d_to_3d'].shape[0]):
        infos_text.append('2d {} to 3d {}   Area: {} Angle: {} Accepted: {}'.format(i,visualisation_info['all_indices_2d_to_3d'][i],np.round(visualisation_info['all_factors_T_2d'][i]),np.round(visualisation_info["all_angles_T_2d"][i]),visualisation_info['all_accepted_T_2d'][i]))

        if visualisation_info['all_accepted_T_2d'][i] == True:
            running_sum += visualisation_info['all_factors_T_2d'][i]

    return infos_text

def format_infos(visualisation_info,factors_all_2d_lines):
    infos_text = []
    running_sum = 0
    for i in range(factors_all_2d_lines.shape[0]):
        infos_text.append('2d {} to 3d {}   Area: {} Angle: {} Accepted: {}'.format(i,visualisation_info["indices_2d_to_3d"][i],np.round(factors_all_2d_lines[i]),np.round(visualisation_info["angles_2d_to_3d"][i]),visualisation_info['accepted_2d_line'][i]))
        if visualisation_info['accepted_2d_line'][i] == True:
            running_sum += factors_all_2d_lines[i]

    return infos_text


def get_pointsize_from_Ts(Ts):
    Ts = np.array(Ts)

    assert len(Ts.shape) == 2 and Ts.shape[1] == 3, Ts.shape
    mins = np.min(Ts,axis=0)
    maxs = np.max(Ts,axis=0)

    n_per_dim = [np.unique(Ts[:,i]).shape[0] for i in range(3)]
    ranges = maxs - mins
    step_size = ranges / n_per_dim

    return min(step_size[step_size > 0]) / 1.2



def plot_lines_T_correct_visualisation(R,T,scaling,model_path,img_path,sw,device,lines_3D,lines_2D,B,f,area_threshold,angle_threshold,only_allow_single_mapping_to_3d):
    original_image = cv2.imread(img_path)
    original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
    h,w,_= original_image.shape
    mesh = load_mesh(model_path,R[0],T[0],scaling,device)

    rendered_image = render_mesh(w,h,f,mesh,device,sw)
    original_image = cv2.imread(img_path)
    original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
    out_img = overlay_rendered_image(original_image,rendered_image)


    if len(lines_2D.shape) > 1:
        S = torch.Tensor(scaling).unsqueeze(0)

        # factors_all_2d_lines,all_3d = get_factor_reproject_lines_single_R(R[0],torch.Tensor(lines_3D[:,3:6]),lines_2D,B.cpu())
        # get slightly different results when performing computation on cpu!!!
        factors,areas,factors_all_2d_lines,visualisation_info = get_factor_reproject_lines_multiple_T_threshold_map_single_3d_line_v2(R[0],T[0],torch.from_numpy(lines_3D),lines_2D,B.cpu(),f,sw,area_threshold,angle_threshold,only_allow_single_mapping_to_3d)
        # factors,areas,factors_all_2d_lines,visualisation_info = get_factor_reproject_lines_multiple_T_threshold_map_single_3d_line_v2(R[0].to(device),T[0].to(device),torch.from_numpy(lines_3D).to(device),lines_2D.to(device),B.cpu().to(device),f,sw,area_threshold,angle_threshold,only_allow_single_mapping_to_3d)
        pix1_3d = visualisation_info['pix1_3d'][:lines_3D.shape[0]]
        pix2_3d = visualisation_info['pix2_3d'][:lines_3D.shape[0]]
        pix1_2d = visualisation_info['pix1_2d'][::lines_3D.shape[0]]
        pix2_2d = visualisation_info['pix2_2d'][::lines_3D.shape[0]]

        individual_size = (640,480)

        # draw lines 3d 
        img_lines_3d = plot_matches_individual_color_no_endpoints(out_img.copy(),pix1_3d,pix2_3d,line_colors=[255,0,0])
        img_lines_3d = draw_factors(img_lines_3d,np.arange(pix1_3d.shape[0]),(pix1_3d + pix2_3d) / 2,flip_xy=True)

        # lines 2d
        img_lines_2d = plot_matches_individual_color_no_endpoints(out_img.copy(),pix1_2d,pix2_2d,line_colors=[0,0,255])
        img_lines_2d = draw_factors(img_lines_2d,np.arange(pix1_2d.shape[0]),(pix1_2d + pix2_2d) / 2,flip_xy=True)

        # combined
        img_lines_combined = plot_matches_individual_color_no_endpoints(out_img.copy(),pix1_3d,pix2_3d,line_colors=[255,0,0])
        img_lines_combined = plot_matches_individual_color_no_endpoints(img_lines_combined,pix1_2d,pix2_2d,line_colors=[0,0,255])

        mask_plot_polygons = visualisation_info['accepted_2d_line']
        vis_max_indices = visualisation_info['max_indices'].view(lines_2D.shape[0],lines_3D.shape[0])
        img_lines_combined = plot_polygons_v2(img_lines_combined,pix1_3d,pix2_3d,pix1_2d,pix2_2d,visualisation_info['indices_2d_to_3d'],vis_max_indices,mask_plot_polygons)
        line_centers = (lines_2D[:,:2] + lines_2D[:,2:4])/2
        img_lines_combined = draw_factors(img_lines_combined,factors_all_2d_lines.squeeze(0).numpy(),line_centers.numpy())
        draw_text_block(img_lines_combined,['area threshold: '+ str(np.round(area_threshold,3)),'factor: '+str(factors.item())])
        

        # lines 2d used
        mask_2d = visualisation_info['accepted_2d_line']
        img_lines_2d_used = plot_matches_individual_color_no_endpoints(out_img.copy(),pix1_2d[mask_2d],pix2_2d[mask_2d],line_colors=[0,0,255])
        img_lines_2d_used = draw_factors(img_lines_2d_used,np.arange(pix1_2d.shape[0])[mask_2d],((pix1_2d + pix2_2d) / 2)[mask_2d],flip_xy=True)

        # draw lines 3d used
        mask_3d = np.zeros(lines_3D.shape[0], dtype=bool)
        mask_3d[np.unique(visualisation_info['indices_2d_to_3d'][visualisation_info['accepted_2d_line']])] = True
        img_lines_3d_used = plot_matches_individual_color_no_endpoints(out_img.copy(),pix1_3d[mask_3d],pix2_3d[mask_3d],line_colors=[255,0,0])
        img_lines_3d_used = draw_factors(img_lines_3d_used,np.arange(pix1_3d.shape[0])[mask_3d],((pix1_3d + pix2_3d) / 2)[mask_3d],flip_xy=True)


        # info img
        img_infos = 200 * np.ones([individual_size[1],individual_size[0],3],dtype=np.uint8)
        infos_text = format_infos(visualisation_info,factors_all_2d_lines.squeeze(0))
        draw_text_block(img_infos,infos_text,font_scale=1,font_thickness=1)

        # combine images
        top = cv2.hconcat([cv2.resize(img_lines_2d,individual_size),cv2.resize(img_lines_2d_used,individual_size),cv2.resize(img_lines_combined,individual_size)])
        bottom = cv2.hconcat([cv2.resize(img_lines_3d,individual_size),cv2.resize(img_lines_3d_used,individual_size),img_infos])
        top_bottom = cv2.vconcat([top,bottom])

    return top_bottom,infos_text


def plot_lines_T_correct_visualisation_v2(img_path,sw,device,f,factors_generic_information,factors_specific_T):
    original_image = cv2.imread(img_path)
    original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
    h,w,_= original_image.shape

    mesh = load_mesh(factors_generic_information['model_path'][0],factors_generic_information['R'],factors_specific_T['T'],factors_generic_information['S'],device)

    rendered_image = render_mesh(w,h,f,mesh,device,sw)
    original_image = cv2.imread(img_path)
    original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
    out_img = overlay_rendered_image(original_image,rendered_image)


    if len(factors_generic_information['lines_2D'].shape) > 1:
        S = torch.Tensor(factors_generic_information['S']).unsqueeze(0)

        pix1_3d = factors_specific_T['all_lines_3d_T_3D_4'][:,:2]
        pix2_3d = factors_specific_T['all_lines_3d_T_3D_4'][:,2:4]
        pix1_2d = factors_generic_information['lines_2D'][:,:2]
        pix2_2d = factors_generic_information['lines_2D'][:,2:4]

        individual_size = (640,480)

        # draw lines 3d 
        img_lines_3d = plot_matches_individual_color_no_endpoints(out_img.copy(),pix1_3d,pix2_3d,line_colors=[255,0,0])
        img_lines_3d = draw_factors(img_lines_3d,np.arange(pix1_3d.shape[0]),(pix1_3d + pix2_3d) / 2,flip_xy=True)

        # lines 2d
        img_lines_2d = plot_matches_individual_color_no_endpoints(out_img.copy(),pix1_2d,pix2_2d,line_colors=[0,0,255])
        img_lines_2d = draw_factors(img_lines_2d,np.arange(pix1_2d.shape[0]),(pix1_2d + pix2_2d) / 2,flip_xy=True)

        # combined
        img_lines_combined = plot_matches_individual_color_no_endpoints(out_img.copy(),pix1_3d,pix2_3d,line_colors=[255,0,0])
        img_lines_combined = plot_matches_individual_color_no_endpoints(img_lines_combined,pix1_2d,pix2_2d,line_colors=[0,0,255])

        mask_plot_polygons = factors_specific_T['all_accepted_T_2d']
        vis_max_indices = factors_specific_T['all_max_indices']
        img_lines_combined = plot_polygons_v2(img_lines_combined,pix1_3d,pix2_3d,pix1_2d,pix2_2d,factors_specific_T['all_indices_2d_to_3d'],vis_max_indices,mask_plot_polygons)
        line_centers = (pix1_2d + pix2_2d)/2
        img_lines_combined = draw_factors(img_lines_combined,factors_specific_T['all_factors_T_2d'],line_centers,flip_xy=True)
        draw_text_block(img_lines_combined,['area threshold: '+ str(np.round(factors_generic_information['area_threshold'],3)),'factor: '+str(factors_specific_T['n_accepted_all_Ts'])])
        

        # lines 2d used
        mask_2d = factors_specific_T['all_accepted_T_2d']
        img_lines_2d_used = plot_matches_individual_color_no_endpoints(out_img.copy(),pix1_2d[mask_2d],pix2_2d[mask_2d],line_colors=[0,0,255])
        img_lines_2d_used = draw_factors(img_lines_2d_used,np.arange(pix1_2d.shape[0])[mask_2d],((pix1_2d + pix2_2d) / 2)[mask_2d],flip_xy=True)

        # draw lines 3d used
        mask_3d = np.zeros(pix1_3d.shape[0], dtype=bool)
        mask_3d[np.unique(factors_specific_T['all_indices_2d_to_3d'][mask_2d])] = True
        img_lines_3d_used = plot_matches_individual_color_no_endpoints(out_img.copy(),pix1_3d[mask_3d],pix2_3d[mask_3d],line_colors=[255,0,0])
        img_lines_3d_used = draw_factors(img_lines_3d_used,np.arange(pix1_3d.shape[0])[mask_3d],((pix1_3d + pix2_3d) / 2)[mask_3d],flip_xy=True)


        # info img
        img_infos = 200 * np.ones([individual_size[1],individual_size[0],3],dtype=np.uint8)
        infos_text = format_infos_v2(factors_specific_T)
        draw_text_block(img_infos,infos_text,font_scale=1,font_thickness=1)

        # combine images
        top = cv2.hconcat([cv2.resize(img_lines_2d,individual_size),cv2.resize(img_lines_2d_used,individual_size),cv2.resize(img_lines_combined,individual_size)])
        bottom = cv2.hconcat([cv2.resize(img_lines_3d,individual_size),cv2.resize(img_lines_3d_used,individual_size),img_infos])
        top_bottom = cv2.vconcat([top,bottom])

    return top_bottom,infos_text