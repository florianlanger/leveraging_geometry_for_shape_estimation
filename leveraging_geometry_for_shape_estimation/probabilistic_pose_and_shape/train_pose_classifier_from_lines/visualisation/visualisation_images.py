from matplotlib import pyplot as plt
import numpy as np

from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.visualisation.utils import matplotlib_imshow,load_rgb_image,show_rgb_plus_lines,load_gt_reprojection
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.dataset import get_datatype_to_channel_index,plot_lines_3d_torch


def plot_classes_preds(images, labels,probabilities,correct,config,extra_infos,kind):

    datatype_to_channel_index = get_datatype_to_channel_index(config)

    fig = plt.figure(figsize=(13, 10))
    n_images = min(config["training"]["n_vis"],images.shape[0])
    for idx in np.arange(n_images):
        ax = fig.add_subplot(5, 6, 6*idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx,:3,:,:])
        ax.set_title("gt:{},pred:{:.4f}".format(labels[idx].item(),probabilities[idx].item()),color=("green" if correct[idx] else "red"))

        if config["data"]["use_rgb"]:
            ax = fig.add_subplot(5, 6, 6*idx+2, xticks=[], yticks=[])
            start_index = datatype_to_channel_index["rgb"]
            matplotlib_imshow(images[idx,start_index:start_index + 3,:,:])

        if config["data"]["use_normals"]:
            ax = fig.add_subplot(5, 6, 6*idx+3, xticks=[], yticks=[])
            start_index = datatype_to_channel_index["normals"]
            matplotlib_imshow(images[idx,start_index:start_index + 3,:,:])

        if config["data"]["use_alpha"]:
            ax = fig.add_subplot(5, 6, 6*idx+4, xticks=[], yticks=[])
            start_index = datatype_to_channel_index["alpha"]
            matplotlib_imshow(images[idx,start_index:start_index + 3,:,:])
        
        ax = fig.add_subplot(5, 6, 6*idx+5, xticks=[], yticks=[])
        img_rgb = load_rgb_image(config,kind,extra_infos["gt_name"][idx])
        # img_rgb = load_rgb_image(config,'train',extra_infos["gt_name"][idx])
        # print('Change back')
        show_rgb_plus_lines(img_rgb,images[idx,:3,:,:])
        ax.set_title("t {} r {} \n s {} sym {}".format(np.round(extra_infos["offset_t"][idx],2).tolist(),extra_infos["r_correct"][idx].item(),np.round(extra_infos["offset_s"][idx],2).tolist(),extra_infos["sym"][idx]),color='blue')
        

        ax = fig.add_subplot(5, 6, 6*idx+6, xticks=[], yticks=[])
        img_rgb = load_gt_reprojection(config,extra_infos["gt_name"][idx])
        plt.imshow(img_rgb)
        ax.set_title(extra_infos["detection_name"][idx])

    return fig