from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cmx

from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.visualisation.utils import load_rgb_image,load_gt_reprojection,matplotlib_imshow

def plot_preds_grid(images, labels,probabilities,correct,config,extra_infos,title):

    kind = 'val'

    fig = plt.figure(figsize=(13, 10))   
    plt.title(title)
    plt.axis('off')
    plt.grid(b=None) 
    ax = fig.add_subplot(4, 6, 1, xticks=[], yticks=[])
    img_rgb = load_rgb_image(config,kind,extra_infos["gt_name"][0])
    show_rgb_plus_lines(img_rgb,images[0,:3,:,:])
    ax.set_title(extra_infos["detection_name"][0])

    ax = fig.add_subplot(4, 6,2, xticks=[], yticks=[])
    img_rgb = load_gt_reprojection(config,extra_infos["gt_name"][0])
    plt.imshow(img_rgb)
    for idx in np.arange(config["training"]["val_grid_points_per_example"]):
        ax = fig.add_subplot(4, 6, idx + 3, xticks=[], yticks=[])
        matplotlib_imshow(images[idx,:3,:,:])
        ax.set_title("gt:{},pred:{:.4f}\n{}".format(labels[idx].item(),probabilities[idx].item(),np.round(extra_infos["offset"][idx],2).tolist()),color=("green" if correct[idx] else "red"))

    return fig

def plot_grid(Ts,probabilities,title):

    max_index = np.argmax(probabilities)

    fig = plt.figure()
    plt.title(title)
    cm = plt.get_cmap('inferno')
    cNorm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    # ax = Axes3D(fig)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(Ts[max_index,0],Ts[max_index,1], Ts[max_index,2], c='cyan',s=160,depthshade=False)
    ax.scatter(Ts[:,0],Ts[:,1], Ts[:,2], c=cm(probabilities),s=160,depthshade=False)
    ax.scatter(0,0,0, c='green',s=160,depthshade=False)
    # ax.scatter(Ts[max_index,0],Ts[max_index,1], Ts[max_index,2], c='cyan',s=160,depthshade=False)


    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    draw_cube(ax,Ts[:8])
    draw_cube(ax,Ts[8:16])
    
    scalarMap.set_array(probabilities)
    fig.colorbar(scalarMap)

    return fig

def draw_cube(ax,corners):
    ax.plot(corners[[0,1],0],corners[[0,1],1],corners[[0,1],2],color='blue')
    ax.plot(corners[[0,2],0],corners[[0,2],1],corners[[0,2],2],color='blue')
    ax.plot(corners[[1,3],0],corners[[1,3],1],corners[[1,3],2],color='blue')
    ax.plot(corners[[2,3],0],corners[[2,3],1],corners[[2,3],2],color='blue')

    ax.plot(corners[[4,5],0],corners[[4,5],1],corners[[4,5],2],color='blue')
    ax.plot(corners[[4,6],0],corners[[4,6],1],corners[[4,6],2],color='blue')
    ax.plot(corners[[5,7],0],corners[[5,7],1],corners[[5,7],2],color='blue')
    ax.plot(corners[[6,7],0],corners[[6,7],1],corners[[6,7],2],color='blue')

    ax.plot(corners[[0,4],0],corners[[0,4],1],corners[[0,4],2],color='blue')
    ax.plot(corners[[1,5],0],corners[[1,5],1],corners[[1,5],2],color='blue')
    ax.plot(corners[[2,6],0],corners[[2,6],1],corners[[2,6],2],color='blue')
    ax.plot(corners[[3,7],0],corners[[3,7],1],corners[[3,7],2],color='blue')