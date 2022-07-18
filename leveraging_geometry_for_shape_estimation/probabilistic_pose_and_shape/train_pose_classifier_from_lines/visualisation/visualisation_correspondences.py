from matplotlib import pyplot as plt
import numpy as np
import torch

from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.visualisation.utils import load_rgb_image, plot_points_channels,draw_points_on_image
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.dataset import plot_lines_3d_torch

def plot_offsets_preds(points, offsets,outputs,loss,pixel_dist,config,extra_infos,kind):

    fig = plt.figure(figsize=(20, 20))


    points = points * np.array(config['data']['img_size'] + [1])
    offsets = offsets * np.array(config['data']['img_size'])
    outputs = outputs * np.array(config['data']['img_size'])

    point_size = 0.01
    n_rows = min(config["training"]["n_vis"],points.shape[0])
    n_col = 6
    for idx in np.arange(n_rows):
        single_example = points[idx,:,:]

        channel_0 = single_example[single_example[:,2] == 0,:]
        channel_1 = single_example[single_example[:,2] == 1,:]
        channel_2 = single_example[single_example[:,2] == 2,:]

        n_corr = channel_2.shape[0]
        max_side_length = max(config['data']['img_size'])

        ax = fig.add_subplot(n_rows, n_col, n_col*idx+1, projection='3d')
        relevant_loss = np.round(torch.sum(loss[idx,:n_corr]).item(),4)

        title = "loss: {} n: {} loss per n: {} dist per n: {}".format(relevant_loss,n_corr,np.round(relevant_loss / n_corr,3),np.round(pixel_dist[idx],3))
        ax.set_title(title)
        ax = plot_points_channels(ax,channel_0,channel_1,channel_2,point_size)

        ax = fig.add_subplot(n_rows, n_col, n_col*idx+2, projection='3d')
        ax = plot_points_channels(ax,channel_0,channel_1,channel_2,point_size)
        ax.view_init(elev=-90., azim=270)

        ax = fig.add_subplot(n_rows, n_col, n_col*idx+3, xticks=[], yticks=[])
        img_rgb = load_rgb_image(config,kind,extra_infos["gt_name"][idx],normalised=False)
        img_drawn = draw_points_on_image(points[idx],img_rgb)
        plt.imshow(img_drawn[...,::-1])

        ax = fig.add_subplot(n_rows, n_col, n_col*idx+4, xticks=[], yticks=[])
        img_drawn = plot_correspondence(points[idx],offsets[idx],img_rgb)
        plt.imshow(img_drawn[...,::-1])

        ax = fig.add_subplot(n_rows, n_col, n_col*idx+5, xticks=[], yticks=[])
        img_drawn = plot_correspondence(points[idx],outputs[idx],img_rgb)
        plt.imshow(img_drawn[...,::-1])

    
        ax = fig.add_subplot(n_rows, n_col, n_col*idx+6, xticks=[], yticks=[])
        img_rgb = load_rgb_image(config,kind,extra_infos["gt_name"][idx])
        # print('change back')
        # img_rgb = load_rgb_image(config,'train',extra_infos["gt_name"][idx])
        plt.imshow(img_rgb[...,::-1])
        ax.set_title(extra_infos["detection_name"][idx])

    return fig

def plot_correspondence(points,offsets,img):
    start_points = points[points[:,2] == 2,:2]
    end_points = offsets[:start_points.shape[0],:] + start_points
    lines = torch.cat((start_points[:,:2],end_points[:,:2]),dim=1)
    out_img = plot_lines_3d_torch(lines,torch.from_numpy(img),lines.shape[0],samples_per_pixel=1,line_color=[255,0,0],channel=None)
    return out_img[0].numpy()