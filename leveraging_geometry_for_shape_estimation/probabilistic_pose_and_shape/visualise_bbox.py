import torch
import cv2
import numpy as np
import random

from leveraging_geometry_for_shape_estimation.vis_pose.vis_pose import plot_polygons,plot_polygons_v2, load_mesh, render_mesh, overlay_rendered_image, plot_matches_individual_color_no_endpoints


from probabilistic_formulation.factors.factors_T.factors_lines_multiple_T import get_factor_reproject_lines_multiple_T_threshold_map_single_3d_line_v2
from probabilistic_formulation.tests.test_reproject_lines import draw_factors

from leveraging_geometry_for_shape_estimation.segmentation.meshrcnn_vis_tools import draw_boxes
from leveraging_geometry_for_shape_estimation.utilities.write_on_images import draw_text_block


def plot_bbox_T_v2(img_path,sw,device,f,factors_generic_information,factors_specific_T):


    original_image = cv2.imread(img_path)
    original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
    h,w,_= original_image.shape

    mesh = load_mesh(factors_generic_information['model_path'][0],factors_generic_information['R'],factors_specific_T['T'],factors_generic_information['S'],device)

    rendered_image = render_mesh(w,h,f,mesh,device,sw)
    original_image = cv2.imread(img_path)
    original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
    out_img = overlay_rendered_image(original_image,rendered_image)


    pix1_3d = factors_specific_T['all_lines_3d_T_3D_4'][:,:2]
    pix2_3d = factors_specific_T['all_lines_3d_T_3D_4'][:,2:4]

    # draw lines 3d 
    out_img = plot_matches_individual_color_no_endpoints(out_img,pix1_3d,pix2_3d,line_colors=[255,0,0])
    # add offset so that can see
    out_img = draw_boxes(out_img,np.expand_dims(factors_specific_T["bbox_each_T"] + np.array([2,2,-2,-2]),0),thickness=3,color=(0,0,255))
    out_img = draw_boxes(out_img,np.expand_dims(factors_generic_information['pred_bbox'],0),thickness=3)

    cv2.putText(out_img, 'iou: ' + str(np.round(factors_specific_T["box_iou"],3)), (20,100), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 0, 0), 2, cv2.LINE_AA)

    return out_img