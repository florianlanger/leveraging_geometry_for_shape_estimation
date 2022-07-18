from matplotlib import pyplot as plt
import numpy as np

from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.visualisation.utils import load_rgb_image,plot_points_channels,draw_points_on_image



def plot_points_preds(points, labels,probabilities,correct,config,extra_infos,kind):

    fig = plt.figure(figsize=(10, 10))

    if config['model']['type'] == 'pointnet':
        points = points.transpose(2, 1)

    points = points * np.array(config['data']['img_size'] + [1])
    point_size = 0.01
    n_rows = min(config["training"]["n_vis"],points.shape[0])
    n_col = 4
    for idx in np.arange(n_rows):
        single_example = points[idx,:,:]

        channel_0 = single_example[single_example[:,2] == 0,:]
        channel_1 = single_example[single_example[:,2] == 1,:]
        channel_2 = single_example[single_example[:,2] == 2,:]

        ax = fig.add_subplot(n_rows, n_col, n_col*idx+1, projection='3d')
        title = "gt:{},pred:{:.4f}\nr {} sym {}".format(labels[idx].item(),probabilities[idx].item(),extra_infos["r_correct"][idx].item(),extra_infos["sym"][idx])
        ax.set_title(title,color=("green" if correct[idx] else "red"))
        ax = plot_points_channels(ax,channel_0,channel_1,channel_2,point_size)

        ax = fig.add_subplot(n_rows, n_col, n_col*idx+2, projection='3d')
        ax = plot_points_channels(ax,channel_0,channel_1,channel_2,point_size)
        ax.view_init(elev=-90., azim=270)

        ax = fig.add_subplot(n_rows, n_col, n_col*idx+3, xticks=[], yticks=[])
        img_rgb = load_rgb_image(config,kind,extra_infos["gt_name"][idx],normalised=False)
        img_drawn = draw_points_on_image(points[idx],img_rgb)
        plt.imshow(img_drawn[...,::-1])
        title = "t {}\ns {}".format(np.round(extra_infos["offset_t"][idx],2).tolist(),np.round(extra_infos["offset_s"][idx],2).tolist())
        ax.set_title(title,color="blue")


        ax = fig.add_subplot(n_rows, n_col, n_col*idx+4, xticks=[], yticks=[])
        img_rgb = load_rgb_image(config,kind,extra_infos["gt_name"][idx])
        # print('change back')
        # img_rgb = load_rgb_image(config,'train',extra_infos["gt_name"][idx])
        plt.imshow(img_rgb[...,::-1])
        ax.set_title(extra_infos["detection_name"][idx])

    return fig