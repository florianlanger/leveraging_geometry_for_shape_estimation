import torch
import numpy as np
from tqdm import tqdm

from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.metrics import track_cat_accuracies
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.visualisation.grid import plot_preds_grid,plot_grid

def validate_grid(data_loader,net,epoch,writer,device,config):
    
    grid_points_per_example = config["training"]["val_grid_points_per_example"]
    all_correct = []
    all_categories = []

    net.eval()

    for i, data in tqdm(enumerate(data_loader, 0)):


        inputs, labels, extra_infos = data
        inputs = torch.permute(inputs, (0,3,1,2)).to(device)
        labels = labels.to(device)
        
        outputs,embedding = net(inputs)
        probabilities = torch.sigmoid(outputs)
        correct = (probabilities.squeeze(1) > 0.5) == labels.squeeze(1)

        probabilities = probabilities.squeeze(1)

        correct_per_example,max_prob_per_example,argmax_per_example = probabilities_batch_to_correct(probabilities,config)
        all_correct += correct_per_example
        all_categories += extra_infos['category'][::grid_points_per_example]

        if i == 0:
            for j in range(int(config["training"]["batch_size_val"] / grid_points_per_example)):
                # global_step = epoch * len(data_loader) * config["training"]["batch_size"] + i * config["training"]["batch_size"] + j
                title = get_title(correct_per_example[j],max_prob_per_example[j],argmax_per_example[j])
                start,stop = j*grid_points_per_example,(j+1)*grid_points_per_example
                writer.add_figure('validate grid images', plot_preds_grid(inputs[start:stop].cpu(), labels[start:stop],probabilities[start:stop],correct[start:stop],config,extra_infos,title), epoch)
                writer.add_figure('validate grid', plot_grid(data_loader.dataset.grid_val_T,probabilities[start:stop].cpu().numpy(),title), epoch)


        for j in range(int(inputs.shape[0] / grid_points_per_example)):
            assert len(set(extra_infos['category'][j*grid_points_per_example:(j+1)*grid_points_per_example])) == 1, set(extra_infos['category'][j*grid_points_per_example:(j+1)*grid_points_per_example])

    # global_step = (epoch+1) * len(data_loader) * config["training"]["batch_size_val"]/grid_points_per_example
    all_labels = [1] * len(all_correct)
    track_cat_accuracies(all_correct,all_labels,all_categories,writer,epoch,'val',name='validate_grid')


def probabilities_batch_to_correct(probabilities,config):
    probabilities = probabilities.view(-1,config["training"]["val_grid_points_per_example"])
    max_prob_per_example,argmax_per_example = torch.max(probabilities,dim=1)
    correct = argmax_per_example < config["training"]["val_grid_points_correct"]
    return correct.tolist(),max_prob_per_example.tolist(),argmax_per_example.tolist()

def get_title(correct,max_prob,argmax):
    return "correct: {}, max prob: {:.4f}, index: {}".format(correct,max_prob,argmax)