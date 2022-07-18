
from audioop import add
from pickletools import optimize
from re import I
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from perceiver_pytorch import Perceiver, PerceiverIO
import torch_optimizer as optim_special
from transformers import PerceiverModel, PerceiverConfig
import sys
import json
from scipy.spatial.transform import Rotation as scipy_rot

import time
import psutil
import random
import itertools

# from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.model.perceiver_pytorch_local.perceiver_pytorch.perceiver_pytorch import Perceiver as Perceiver_local

from leveraging_geometry_for_shape_estimation.utilities.dicts import load_json
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.utiliies import create_directories
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.model.image_network import Classification_network
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.model.pointnet.pointnet2_cls_msg import get_model
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.batch_sampler import get_batch_from_dataset,get_index_infos,get_simple_batch_from_dataset
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.evaluate.evaluate_roca import eval_predictions
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.dataset_points_v2 import Dataset_points
# print('import dataset single normals')
# from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.dataset_points_v2_single_normals import Dataset_points
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.visualisation.visualisation_main import visualise_preds
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.visualisation.confusion_matrix import visualise_confusion_matrices
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.metrics import track_cat_accuracies, get_distances_per_point
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.visualisation.visualisation_correspondences import plot_offsets_preds
from leveraging_geometry_for_shape_estimation.data_conversion.create_dirs import dict_replace_value
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.losses.loss import get_combined_loss
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.losses.geometric_loss import geometric_loss
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.convert_batch_vgg import convert_input_to_vgg_input


def one_epoch_classifier(dataset,net,optimizer,criterion,epoch,writer,kind,device,config,exp_path):

    all_metrics = {'all_predictions':[],'all_labels':[],'all_categories':[],'all_roca_bbox':[],'all_losses':[],'all_distances':[],'all_extra_infos':[],'t_distance': [], 's_distance': [],'r_distance':[],'t_correct': [],
    's_correct': [],'running_loss':0.0,'n_correct_total':0,'counter_examples':0,'iter_refinement':[],'t_offset':[],'s_offset':[],'r_offset':[],'detection_names':[],'s_pred':[],'t_pred':[],'r_pred':[],'weighted_classification_loss':[],'weighted_t_loss':[],'weighted_s_loss':[],'weighted_r_loss':[]}
    
    set_network_state(net,kind)

    
    roca_eval_combo = None
    if kind == 'val_roca':
        roca_eval_combo = 'translation_{}_scale_{}_rotation_{}_retrieval_{}'.format(dataset.eval_params['what_translation'],dataset.eval_params['what_scale'],dataset.eval_params['what_rotation'],dataset.eval_params['what_retrieval'])

    all_indices = create_shuffled_indices(len(dataset),config,kind)

    N_total_batches = int(np.ceil(len(all_indices) / config['training']["batch_size"]))
    # N_refinements = 1 if epoch < config["training"]["use_refinement_after_which_epoch"] else config["training"]["refinement_per_object"]

    for num_batch in tqdm(range(N_total_batches)):
    # for num_batch in range(N_total_batches):
        index_infos = all_indices[num_batch*config['training']["batch_size"]:(num_batch+1)*config['training']["batch_size"]]

        N_refinements,just_classifier = get_N_refinements_and_sample_for_classification_v3(kind,config)
        # print('debug')
        # just_classifier = True
        # N_refinements = 1


        for iter_refinement in range(N_refinements):
            data = get_batch_from_dataset(dataset,index_infos,just_classifier)
            inputs, targets, extra_infos = data
            # inputs = inputs.to(device)    

            # targets = targets.to(device)

            optimizer.zero_grad()

            if config["model"]["type"] == "vgg16" or config["model"]["type"] == "resnet50":
                inputs_vgg = convert_input_to_vgg_input(inputs)
                outputs = net(inputs_vgg)
            else:
                outputs = net(inputs)
            if dataset.config['data']["sample_S"]["ignore_prediction"] == True:
                outputs[:,4:7] = outputs[:,4:7] * 0

            if dataset.config['data']["sample_T"]["ignore_prediction"] == True:
                outputs[:,1:4] = outputs[:,1:4] * 0

            if dataset.config['data']["sample_R"]["ignore_prediction"] == True:
                outputs[:,7:11] = outputs[:,7:11] * 0 + torch.Tensor([[0,0,0,1]]).repeat(outputs.shape[0],1).to(device)

            # print('remove')
            # outputs[:,4:7] = outputs[:,4:7] * 0
            # outputs[:,1:4] = outputs[:,1:4] * 0
            # outputs[:,7:11] = outputs[:,7:11] * 0 + torch.Tensor([[0,0,0,1]]).repeat(outputs.shape[0],1).to(device)

            # outputs = outputs * 0 + targets
            # outputs[:,7:16] = torch.Tensor([[0,0,-1,0,1,0,1,0,0]]).repeat(outputs.shape[0],1).to(device)

            # outputs[:,1:4] = outputs[:,1:4] * 0

            if config["loss"]["use_geometric_loss"] == False:
                loss,latest_metrics = get_combined_loss(outputs,targets,config,just_classifier)
            elif config["loss"]["use_geometric_loss"] == True:
                loss,latest_metrics = geometric_loss(outputs,targets,config,extra_infos)

            mean_loss = torch.mean(loss)
            if kind == 'train':
                mean_loss.backward()
                optimizer.step()
            t4 = time.time()
            all_metrics = update_running_metrics(all_metrics,latest_metrics,extra_infos,loss,iter_refinement)

            t5 = time.time()
            if num_batch == 0 and epoch % config['training']["vis_interval"] == 0:
                visualise_preds(writer,latest_metrics,inputs,net,config,kind,epoch,extra_infos,exp_path,num_batch,iter_refinement,roca_eval_combo)


            index_infos = get_index_infos(outputs,extra_infos,config,iter_refinement)
            t6 = time.time()
            # print('t2 - t1: {}'.format(t2 - t1))
            # print('t3 - t2: {}'.format(t3 - t2))
            # print('t4 - t3: {}'.format(t4 - t3))
            # print('t5 - t4: {}'.format(t5 - t4))
            # print('t6 - t5: {}'.format(t6 - t5))

    if epoch % config['training']["vis_interval"] == 0:
        visualise_confusion_matrices(all_metrics['all_predictions'],all_metrics['all_labels'],all_metrics['all_categories'],writer,epoch,kind)

    if kind == 'val_roca':
        eval_path = save_predictions(all_metrics,N_refinements,epoch,exp_path,dataset.eval_params,dataset.use_all_images)
        if config["general"]["run_on_octopus" ] == True:
            n_scenes_vis = config["training"]["n_vis_scenes"]
        else:
            n_scenes_vis = 0

        eval_predictions(eval_path,n_scenes_vis=n_scenes_vis,eval_all_images=dataset.use_all_images)
    track_cat_accuracies(all_metrics,writer,epoch,kind,N_refinements,config)
    writer.add_scalar(kind + ' loss',all_metrics['running_loss'] / all_metrics['counter_examples'],epoch)

    return all_metrics['running_loss'] / all_metrics['counter_examples'], all_metrics['n_correct_total']/ all_metrics['counter_examples'], all_metrics['all_extra_infos'],all_metrics['all_losses']


def save_predictions(all_metrics,N_refinements,epoch,exp_path,eval_params,use_all_images):


    extra_infos_combined = {}
    for key in all_metrics['all_extra_infos'][0]:
        extra_infos_combined[key] = np.concatenate([all_metrics['all_extra_infos'][i][key] for i in range(len(all_metrics['all_extra_infos']))])

    # for key in all_metrics['all_extra_infos'][0]:
    #     print(key)

    iter_refinement = np.array(all_metrics['iter_refinement'])

    mask = iter_refinement == N_refinements - 1
    # detection_names = np.array(all_metrics['detection_names'])[mask]
    # s_pred = np.array(all_metrics['s_pred'])[mask,:]
    # t_pred = np.array(all_metrics['t_pred'])[mask,:]

    detection_names = np.array(all_metrics['detection_names'])
    s_pred = np.array(all_metrics['s_pred'])
    t_pred = np.array(all_metrics['t_pred'])
    r_pred = np.array(all_metrics['r_pred'])

    assert detection_names.shape[0] == s_pred.shape[0] == t_pred.shape[0]
    output = {}
    for i in range(detection_names.shape[0]):
        if mask[i] == True:
            output[detection_names[i]] = {}
            output[detection_names[i]]['s'] = (extra_infos_combined['S'][i,:] * (1 + s_pred[i])).tolist()
            output[detection_names[i]]['t'] = (extra_infos_combined['T'][i,:] + t_pred[i]).tolist()

            r_offset_pred = scipy_rot.from_quat(r_pred[i])
            q = (scipy_rot.from_matrix(extra_infos_combined['R'][i])*r_offset_pred).as_quat()
            # change because different convention
            q = [q[3],q[0],q[1],q[2]]
    
            output[detection_names[i]]['q'] = q
            output[detection_names[i]]["model_id"] = extra_infos_combined['model_3d_name'][i].split('_')[1].split('.')[0]
            output[detection_names[i]]["classification_score"] = all_metrics['all_predictions'][i]

    out_dir = exp_path + '/predictions/epoch_{}'.format(str(epoch).zfill(6))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir += '/translation_{}_scale_{}_rotation_{}_retrieval_{}_all_images_{}'.format(eval_params["what_translation"],eval_params["what_scale"],eval_params["what_rotation"],eval_params["what_retrieval"],str(use_all_images))
    os.mkdir(out_dir)

    out_path = out_dir + '/our_single_predictions.json'
    with open(out_path, 'w') as f:
        json.dump(output, f)

    return out_dir



def get_N_refinements_and_sample_for_classification(kind,config):
    if kind == 'val_roca':
        N_refinements = config['training']['refinement_per_object']
        sample_for_classification = False
    else:
        N_refinements = int(np.random.choice([1,2,3,4], 1, p=[0.6, 0.2, 0.1, 0.1]))
        sample_for_classification = False
        if N_refinements == 1 and np.random.rand() > 0.5:
            sample_for_classification = True
    return N_refinements,sample_for_classification

def get_N_refinements_and_sample_for_classification_v2(kind,config):
    if kind == 'val_roca':
        N_refinements = config['training']['refinement_per_object']
        sample_for_classification = False
    else:
        # N_refinements = int(np.random.choice([1,3], 1, p=[0.5,0.5]))
        p_classifier = config['training']['p_classifier']
        N_refinements = int(np.random.choice([1,config['training']['refinement_per_object']], 1, p=[p_classifier,1-p_classifier]))
        sample_for_classification = False
        if N_refinements == 1:
            sample_for_classification = True
    return N_refinements,sample_for_classification

def get_N_refinements_and_sample_for_classification_v3(kind,config):
    if kind == 'val_roca':
        N_refinements = config['training']['refinement_per_object']
        sample_for_classification = False
    else:
        p_classifier = config['training']['p_classifier']
        sample_for_classification = np.random.choice([True,False], 1, p=[p_classifier,1-p_classifier])
        if sample_for_classification == True:
            N_refinements = 1
        elif sample_for_classification == False:
            N_refinements = config['training']['refinement_per_object']
    return N_refinements,sample_for_classification



def update_running_metrics(all_metrics,latest_metrics,extra_infos,loss,iter_refinement):


    all_metrics['all_predictions'] += latest_metrics['probabilities'].tolist()
    all_metrics['all_labels'] += latest_metrics['labels'].tolist()
    all_metrics['all_categories'] += extra_infos['category'].tolist()
    all_metrics['all_roca_bbox'] += extra_infos['roca_bbox'].tolist()
    all_metrics['all_losses'] += loss.tolist()
    all_metrics['all_extra_infos'].append(extra_infos)
    all_metrics['t_distance'] += latest_metrics['t_distance'].tolist()
    all_metrics['t_offset'] += extra_infos['offset_t'].tolist()
    all_metrics['s_distance'] += latest_metrics['s_distance'].tolist()
    all_metrics['s_offset'] += extra_infos['offset_s'].tolist()
    all_metrics['r_distance'] += latest_metrics['r_distance'].tolist()
    all_metrics['r_offset'] += extra_infos['offset_r'].tolist()
    all_metrics['t_correct'] += latest_metrics['t_correct'].tolist()
    all_metrics['s_correct'] += latest_metrics['s_correct'].tolist()
    all_metrics['iter_refinement'] += [iter_refinement] * len(latest_metrics['probabilities'])

    all_metrics['weighted_classification_loss'] += latest_metrics['weighted_classification_loss'].tolist()
    all_metrics['weighted_t_loss'] += latest_metrics['weighted_t_loss'].tolist()
    all_metrics['weighted_s_loss'] += latest_metrics['weighted_s_loss'].tolist()
    all_metrics['weighted_r_loss'] += latest_metrics['weighted_r_loss'].tolist()

    all_metrics['counter_examples'] += loss.shape[0]
    all_metrics['running_loss'] += torch.sum(loss).item()
    all_metrics['n_correct_total'] += torch.sum(latest_metrics['correct']).item()

    all_metrics['detection_names'] += extra_infos['detection_name'].tolist()
    all_metrics['s_pred'] += latest_metrics['s_pred'].tolist()
    all_metrics['t_pred'] += latest_metrics['t_pred'].tolist()
    all_metrics['r_pred'] += latest_metrics['r_pred'].tolist()

    return all_metrics



def one_epoch_offsets(data_loader,net,optimizer,criterion,epoch,writer,kind,device,config):
    
    running_loss = 0.0
    pixel_dist = 0.0

    counter_examples = 0

    if kind == 'train':
        net.train()
    elif kind == 'val':
        net.eval()

   

    for i, data in enumerate(data_loader, 0):

        inputs, offsets, extra_infos = data
        queries = torch.ones(inputs.shape[0],1000,2).to(device).float()
        inputs = inputs.to(device)        
        # labels = labels.to(device).view(-1)
        offsets = offsets.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs,queries=queries)
        outputs= outputs.view(outputs.shape[0],-1,2)

        loss = criterion(outputs,offsets)
        # print('probabilities',probabilities)
        # print('labels',labels)
        # print('loss',loss)
        mean_loss = torch.mean(loss)
    
        if kind == 'train':
            mean_loss.backward()
            optimizer.step()

        dists = get_distances_per_point(outputs,offsets,extra_infos,config)
        pixel_dist += np.sum(dists)
       
        running_loss += torch.sum(loss).item()
        counter_examples += inputs.shape[0]

        if i == 0 and epoch % config['training']["vis_interval"] == 0:
            writer.add_figure('{} predictions'.format(kind),plot_offsets_preds(inputs.cpu(),offsets.cpu(),outputs.detach().cpu(),loss.cpu(),dists,config,extra_infos,kind),epoch)

    writer.add_scalar(kind + ' loss',running_loss / counter_examples,epoch)
    writer.add_scalar(kind + ' pixel dist',pixel_dist / counter_examples,epoch)

    return running_loss / counter_examples, 0


def set_network_state(net,kind):
    if kind == 'train':
        net.train()
    elif kind == 'val' or kind == 'val_roca':
        net.eval()

def set_device(config):
    if torch.cuda.is_available():
        print('config["general"]["gpu"]',config['general']['gpu'])
        device = torch.device("cuda:{}".format(config["general"]["gpu"]))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        print('on cpu')
    return device

def create_data_loaders(config):


    if config["data"]["type"] == 'points':
        if config['data']['targets'] == 'labels':

            if config['training']['only_eval_roca'] == False:

                train_dataset = Dataset_points(config,kind='train')

                if config["training"]["validate"] == True:
                    val_dataset = Dataset_points(config,kind='val')
                    val_roca_dataset = Dataset_points(config,kind='val_roca')
                    val_roca_dataset_all_images = Dataset_points(config,kind='val_roca',use_all_images=True)
                elif config["training"]["validate"] == False:
                    val_dataset = None
                    val_roca_dataset = None
                    val_roca_dataset_all_images = None
                    print('No val roca dataset')
                
            elif config['training']['only_eval_roca'] == True:
                train_dataset = None
                val_dataset = None
                val_roca_dataset = None
                val_roca_dataset_all_images = Dataset_points(config,kind='val_roca',use_all_images=True)
                print('Only eval roca dataset')
        elif config['data']['targets'] == 'offsets':
            train_dataset = Dataset_points_correspondences(config,kind='train')
            val_dataset = Dataset_points_correspondences(config,kind='val')
            val_roca_dataset =None
    else:
        train_dataset = Dataset_lines(config,kind='train')
        val_dataset = Dataset_lines(config,kind='val')
        val_roca_dataset = None

    print('Loaded datasets')
    return train_dataset, val_dataset,val_roca_dataset, val_roca_dataset_all_images

def create_shuffled_indices(n,config,kind):
    all_indices = np.arange(n)
    if kind == 'train' or kind == 'val':
        random.shuffle(all_indices)
    all_indices = np.repeat(all_indices,config['training']["n_same_objects_per_batch"])
    return all_indices

def log_hparams(writer,config,metric_dict):

    hparam_dict = {'bs': config["training"]["batch_size"],'lr': config["training"]["learning_rate"],'optimiser':config["training"]['optimiser']}
    writer.add_hparams(hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None)

def get_optimiser(config,network):
    if config["training"]["optimiser"] == 'Adam':
        optimizer = optim.Adam([{'params': network.parameters()}], lr=config["training"]["learning_rate"])
    elif config["training"]["optimiser"] == 'SGD':
        optimizer = optim.SGD([{'params': network.parameters()}], lr=config["training"]["learning_rate"],momentum=config["training"]["momentum_sgd"])
    elif config['training']['optimiser'] == 'Lamb':
        optimizer = optim_special.Lamb([{'params': network.parameters()}], lr=config["training"]["learning_rate"])
    return optimizer

def save_checkpoint(dir_path,epoch,network,optimizer,config):
    torch.save(network.state_dict(), dir_path + '/network_last_epoch.pth'.format(str(epoch).zfill(6)))
    torch.save(optimizer.state_dict(), dir_path + '/optimizer_last_epoch.pth'.format(str(epoch).zfill(6)))
    if epoch % config["training"]["save_interval"] == 0:
        torch.save(network.state_dict(), dir_path + '/network_epoch_{}.pth'.format(str(epoch).zfill(6)))
        torch.save(optimizer.state_dict(), dir_path + '/optimizer_epoch_{}.pth'.format(str(epoch).zfill(6)))



def add_missing_keys_to_config(config):

    if "regress_offsets" not in config['model']:
        config['model']["regress_offsets"] = False
    if not "sample_wrong_R_percentage" in config["data_augmentation"]:
        config["data_augmentation"]["sample_wrong_R_percentage"] = 0.0
    if not "dir_path_scan2cad_anno" in config["data"]:
        config["data"]["dir_path_scan2cad_anno"] = "/scratch2/fml35/datasets/scannet/scan2cad_annotations/full_annotations.json"
    if 'type' not in config['data']:
        config['data']['type'] = 'lines'
    if "name_norm_folder" not in config["data"]:
        config["data"]["name_norm_folder"] = "norm"

    if "train_only_classifier" not in config["loss"]:
        config["loss"]["train_only_classifier"] = False
    if "evaluate" not in config:
        config["evaluate"] = {"rotation_index": 0}

    if "N_rotations" not in config["data_augmentation"]:
        config["data_augmentation"]["N_rotations"] = 4
    

    default_false = ['use_all_points_2d','add_random_points','input_rgb',"input_RST_and_CAD_ID","add_history"]
    default_true = ["use_3d_points","rerender_points"]

    for key in default_false:
        if key not in config['data']:
            config['data'][key] = False
        
    for key in default_true:
        if key not in config['data']:
            config['data'][key] = True

    return config

def process_config(config):

    config = add_missing_keys_to_config(config)

    n_rgb = (np.sum([config["data"]["use_rgb"],config["data"]["use_normals"],config["data"]["use_alpha"]]) + 1)
    if "n_same_objects_per_batch" in config["training"]:
        assert config["training"]["batch_size"] % config["training"]["n_same_objects_per_batch"] == 0
        assert config["training"]["batch_size_val"] % config["training"]["n_same_objects_per_batch"] == 0


    if config['model']["regress_offsets"] == False:
        config["model"]["n_outputs"] = 1
    elif config['model']["regress_offsets"] == True:
        # config["model"]["n_outputs"] = 7
        config["model"]["n_outputs"] = 11
        # change back to 11 had 16 before even with quaternion


    assert config["data_augmentation"]["change_R_angle_degree"][1] < 180 / config["data_augmentation"]["N_rotations"]


    if config["general"]["run_on_octopus"] == False:
        config = dict_replace_value(config,'/scratch/fml35/','/scratches/octopus/fml35/')
        config = dict_replace_value(config,'/scratch2/fml35/','/scratches/octopus_2/fml35/')


    if config["model"]["type"] == "vgg16" or config["model"]["type"] == "resnet50":
        config["training"]["optimiser"] = "Adam"
        config["training"]["learning_rate"] = 0.00005
    elif config["model"]["type"] == "perceiver":
        config["training"]["optimiser"] = "Lamb"
        config["training"]["learning_rate"] = 0.001

    if config["data"]["what_models"] == "lines":
        config["data"]["dims_per_pixel"] = 3
    elif config["data"]["what_models"] == "points_and_normals" and config["data"]["input_3d_coords"] ==  False and config["data"]["input_rgb"] ==  False:
        config["data"]["dims_per_pixel"] = 7
        config["data"]["indices_rgb"] = (None,None)
        config["data"]["indices_3d"] = (None,None)
    elif config["data"]["what_models"] == "points_and_normals" and config["data"]["input_3d_coords"] ==  True and config["data"]["input_rgb"] ==  False:
        config["data"]["dims_per_pixel"] = 10
        config["data"]["indices_rgb"] = (None,None)
        config["data"]["indices_3d"] = (7,10)
    elif config["data"]["what_models"] == "points_and_normals" and config["data"]["input_3d_coords"] ==  False and config["data"]["input_rgb"] ==  True:
        config["data"]["dims_per_pixel"] = 10
        config["data"]["indices_rgb"] = (7,10)
        config["data"]["indices_3d"] = (None,None)
    elif config["data"]["what_models"] == "points_and_normals" and config["data"]["input_3d_coords"] ==  True and config["data"]["input_rgb"] ==  True:
        config["data"]["dims_per_pixel"] = 13
        config["data"]["indices_rgb"] = (10,13)
        config["data"]["indices_3d"] = (7,10)

    if config["data"]["sample_what"] == 'T':
        # config["data"]["sample_T"] = {"percent_small": 0.4,"percent_large": 0.5,"threshold_correct_T": 0.2}
        config["data"]["sample_T"] = {"use_gt": False,"ignore_prediction": False,"percent_small": 0.0,"percent_large": 0.0,"threshold_correct_T": 0.2}
        # config["data"]["sample_S"] = {"use_gt": True,"percent_small": 0.7,"limit_small": 0.2,"percent_large": 0.2,"limit_large": 0.5}
        config["data"]["sample_S"] = {"use_gt": True,"ignore_prediction": True,"percent_small": 0.0,"percent_large": 0.0}
        config["data"]["sample_R"] = {"use_gt": True,"ignore_prediction": True}

        config["loss"]["constants_multiplier"]["classification"] = 0.0
        config["loss"]["constants_multiplier"]["s"] = 0.0
        config["loss"]["constants_multiplier"]["r"] = 0.0

    elif config["data"]["sample_what"] == 'T_and_R':
        # config["data"]["sample_T"] = {"percent_small": 0.4,"percent_large": 0.5,"threshold_correct_T": 0.2}
        config["data"]["sample_T"] = {"use_gt": False,"ignore_prediction": False,"percent_small": 0.0,"percent_large": 0.0,"threshold_correct_T": 0.2}
        # config["data"]["sample_S"] = {"use_gt": True,"percent_small": 0.7,"limit_small": 0.2,"percent_large": 0.2,"limit_large": 0.5}
        config["data"]["sample_S"] = {"use_gt": True,"ignore_prediction": True,"percent_small": 0.0,"percent_large": 0.0}
        config["data"]["sample_R"] = {"use_gt": False,"ignore_prediction": False}

        config["loss"]["constants_multiplier"]["classification"] = 0.0
        config["loss"]["constants_multiplier"]["s"] = 0.0

    elif config["data"]["sample_what"] == 'T_and_S':
        config["data"]["sample_T"] = {"use_gt": False,"percent_small": 0.0,"percent_large": 0.0,"threshold_correct_T": 0.2}
        # config["data"]["sample_S"] = {"use_gt": False,"percent_small": 0.0,"limit_small": 0.2,"percent_large": 0.0,"limit_large": 0.5}
        # config["data"]["sample_T"] = {"percent_small": 0.7,"percent_large": 0.2,"threshold_correct_T": 0.2}
        # config["data"]["sample_S"] = {"use_gt": False,"percent_small": 0.7,"limit_small": 0.2,"percent_large": 0.2,"limit_large": 0.5}
        config["data"]["sample_S"] = {"use_gt": False,"percent_small": 0.0,"percent_large": 0.0}

    elif config["data"]["sample_what"] == 'T_and_R_and_S':
        # config["data"]["sample_T"] = {"percent_small": 0.4,"percent_large": 0.5,"threshold_correct_T": 0.2}
        config["data"]["sample_T"] = {"use_gt": False,"ignore_prediction": False,"percent_small": 0.0,"percent_large": 0.0,"threshold_correct_T": 0.2}
        # config["data"]["sample_S"] = {"use_gt": True,"percent_small": 0.7,"limit_small": 0.2,"percent_large": 0.2,"limit_large": 0.5}
        config["data"]["sample_S"] = {"use_gt": False,"ignore_prediction": False,"percent_small": 0.0,"percent_large": 0.0}
        config["data"]["sample_R"] = {"use_gt": False,"ignore_prediction": False}



    elif config["data"]["sample_what"] == 'S':
        config["data"]["sample_T"] = {"use_gt": True,"ignore_prediction": True, "percent_small": 0.0,"percent_large": 0.0,"threshold_correct_T": 0.2}
        config["data"]["sample_S"] = {"use_gt": False,"ignore_prediction": False,"percent_small": 0.0,"percent_large": 0.0}
        config["data"]["sample_R"] = {"use_gt": True,"ignore_prediction": True}
        config["loss"]["constants_multiplier"]["classification"] = 0.0
        config["loss"]["constants_multiplier"]["t"] = 0.0
        config["loss"]["constants_multiplier"]["r"] = 0.0

    elif config["data"]["sample_what"] == 'R':
        config["data"]["sample_T"] = {"use_gt": True,"ignore_prediction": True, "percent_small": 0.0,"percent_large": 0.0,"threshold_correct_T": 0.2}
        config["data"]["sample_S"] = {"use_gt": True,"ignore_prediction": True,"percent_small": 0.0,"percent_large": 0.0}
        config["data"]["sample_R"] = {"use_gt": False,"ignore_prediction": False}
        config["loss"]["constants_multiplier"]["classification"] = 0.0
        config["loss"]["constants_multiplier"]["t"] = 0.0
        config["loss"]["constants_multiplier"]["s"] = 0.0


    return config

def load_network(config,device):
    print('loading network')
    if config["model"]["type"] == 'pointnet':
        network = get_model(num_class=1,normal_channel=False)
        network = network.to(device)

    elif config["model"]["type"] == 'perceiver':
        if config['data']['targets'] == 'labels':
            network = Perceiver(
            input_channels = config['data']["dims_per_pixel"],          # number of channels for each token of the input
            input_axis = 1,              # number of axis for input data (2 for images, 3 for video)
            num_freq_bands = config['model']['perceiver_config']['num_freq_bands'],  # number of freq bands, with original value (2 * K + 1
            max_freq = config['model']['perceiver_config']['max_freq'],              # maximum frequency, hyperparameter depending on how fine the data is
            depth = config['model']['perceiver_config']['depth'],                   # depth of net. The shape of the final attention mechanism will be:
                                        #   depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents = config['model']['perceiver_config']['num_latents'],           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = config['model']['perceiver_config']['latent_dim'],            # latent dimension
            cross_heads = config['model']['perceiver_config']['cross_heads'],             # number of heads for cross attention. paper said 1
            latent_heads = 8,            # number of heads for latent self attention, 8
            cross_dim_head = 64,         # number of dimensions per cross attention head
            latent_dim_head = 64,        # number of dimensions per latent self attention head
            num_classes = config["model"]["n_outputs"],          # output number of classes
            attn_dropout = config['model']['perceiver_config']['attn_dropout'],
            ff_dropout = config['model']['perceiver_config']['ff_dropout'],
            weight_tie_layers = config['model']['perceiver_config']['weight_tie_layers'],   # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn = 2,      # number of self attention blocks per cross attention
            final_classifier_head = True
        )
            network = network.to(device)


        elif config['data']['targets'] == 'offsets':
            network = PerceiverIO(
                dim = 3,                    # dimension of sequence to be encoded
                queries_dim = 2,            # dimension of decoder queries
                depth = config['model']['perceiver_config']['depth'],                   # depth of net
                num_latents = config['model']['perceiver_config']['num_latents'],           # number of latents, or induced set points, or centroids. different papers giving it different names
                latent_dim = config['model']['perceiver_config']['latent_dim'],            # latent dimension
                cross_heads = 1,             # number of heads for cross attention. paper said 1
                latent_heads = 8,            # number of heads for latent self attention, 8
                cross_dim_head = 64,         # number of dimensions per cross attention head
                latent_dim_head = 64,        # number of dimensions per latent self attention head
                weight_tie_layers = False    # whether to weight tie layers (optional, as indicated in the diagram)
            )
            network = network.to(device)


    else:
        network = Classification_network(config,device)
    print('loaded network')
    return network


def one_epoch(dataset,net,optimizer,criterion,epoch,writer,kind,device,config,exp_path):
    if config['data']['targets'] == 'labels':
        return one_epoch_classifier(dataset,net,optimizer,criterion,epoch,writer,kind,device,config,exp_path)
    elif config['data']['targets'] == 'offsets':
        return one_epoch_offsets(data_loader,net,optimizer,criterion,epoch,writer,kind,device,config)

def start_new():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = load_json('{}/config.json'.format(dir_path))
    config = process_config(config)
    exp_path = '{}/{}_{}'.format(config["general"]["output_dir"],datetime.now().strftime("date_%Y_%m_%d_time_%H_%M_%S"),config["general"]["name"])
    print(exp_path)
    create_directories(exp_path)
    device = set_device(config)

    network = load_network(config,device)
    optimizer = get_optimiser(config,network)
    
    start_epoch = 0

    return network,optimizer,exp_path,config,device,start_epoch

def evaluate_checkpoint(checkpoint_path,n_refinements,index_rotation,run_on_octopus=True,gpu_number=0):
    
    name_exp_eval = 'date_2022_' + checkpoint_path.split('/date_2022_')[1].split('/')[0] + '_' + checkpoint_path.split('/')[-1].split('.')[0]   + "_norm_medium" #+ '_depth_mini'
    config_path = checkpoint_path.rsplit('/',2)[0] + '/config.json'
    config = load_json(config_path)
    config = process_config(config)

    # print('evaluate mini depth')


    if run_on_octopus == True:
        config = dict_replace_value(config,'/scratches/octopus/fml35/','/scratch/fml35/')
        config = dict_replace_value(config,'/scratches/octopus_2/fml35/','/scratch2/fml35/')
    elif run_on_octopus == False:
        config = dict_replace_value(config,'/scratch/fml35/','/scratches/octopus/fml35/')
        config = dict_replace_value(config,'/scratch2/fml35/','/scratches/octopus_2/fml35/')

    config["general"]["name"] = name_exp_eval
    config["general"]["output_dir"] = '/scratches/octopus/fml35/experiments/regress_T/evals'
    config["general"]["run_on_octopus"] = run_on_octopus
    config["general"]["gpu"] = str(gpu_number)
    config["training"]["only_eval_roca"] = True
    config["training"]["vis_interval"] = config["training"]["save_interval"]

    print('DEBUG CHANGE CONFIG UNDO')
    config["data"]["input_RST"] = False
    config["data"]["input_RST_and_CAD_ID"] = True

    print('USE norm gb medium')
    config["data"]["name_norm_folder"]= "norm_gb_medium"

    config["evaluate"]["rotation_index"] = index_rotation

    # config["data"]["use_preloaded_depth_and_normals"] = False
    # config["data"]["name_depth_folder"] = "depth_gb_mini"

    config["training"]["refinement_per_object"] = int(n_refinements)
    config["training"]["n_epochs"] = 0
    config["training"]["n_vis"] = 4    

    exp_path = '{}/{}_EVAL_REFINE_{}_{}_rotation_index_{}'.format(config["general"]["output_dir"],datetime.now().strftime("date_%Y_%m_%d_time_%H_%M_%S"),n_refinements,name_exp_eval,index_rotation)
    print(exp_path)
    # print('config:',config["data"])
    create_directories(exp_path)
    device = set_device(config)

    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    network = load_network(config,device)
    network.load_state_dict(checkpoint)
    optimizer = get_optimiser(config,network)
    
    start_epoch = 0

    return network,optimizer,exp_path,config,device,start_epoch

def resume_checkpoint(checkpoint_path):
    exp_path = checkpoint_path.rsplit('/',2)[0]
    config = load_json('{}/config.json'.format(exp_path))
    config = process_config(config)
    device = set_device(config)

    checkpoint = torch.load(checkpoint_path)
    network = load_network(config,device)
    optimizer = get_optimiser(config,network)

    network.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    return network,optimizer,exp_path,config,device,start_epoch




def validate_roca(val_roca_dataset,network,optimizer,criterion,epoch,writer,device,config,exp_path):

    assert val_roca_dataset.kind == 'val_roca', val_roca_dataset.kind
    # print('in eval have kind ',val_roca_dataset.kind)

    if epoch % config["training"]["roca_eval_interval"] == 0 and epoch != 0:
    # if epoch % config["training"]["save_interval"] == 0 and epoch != 0:
        # translation, scale , rotation, retrieval
        # a = [['gt','roca','median','pred'],['lines','roca','gt'],['roca','gt']]
        
        if val_roca_dataset.use_all_images == True:
            # a = [['pred'],['pred'],['init_from_best_rotation_index'],['roca']]
            # a = [['pred'],['pred'],['init_for_classification'],['roca']]

            a = [['pred'],['pred'],['roca_init'],['roca']]
            # a = [['pred'],['pred'],['no_init'],['roca']]
            # print('eval no init undo otherwise error when loading for all images False')
            # print('debug stop')
        else:
            # a = [['gt','roca','pred'],['gt','roca','median'],['gt','roca'],['roca']]
            # a = [['gt','roca','pred'],['pred','gt','roca','median'],['roca_init','gt','roca'],['roca']]
            a = [['pred'],['pred'],['no_init'],['roca']]
            # a = [['pred'],['roca'],['roca'],['roca']]
       
        all_combinations = list(itertools.product(*a))

        for combo in all_combinations:

            if combo[0] == 'pred':
                val_roca_dataset.config["data"]["sample_T"]["ignore_prediction"] = False
            else:
                val_roca_dataset.config["data"]["sample_T"]["ignore_prediction"] = True

            if combo[1] == 'pred':
                val_roca_dataset.config["data"]["sample_S"]["ignore_prediction"] = False
            else:
                val_roca_dataset.config["data"]["sample_S"]["ignore_prediction"] = True

            if combo[2] == 'roca_init' or combo[2] == 'no_init' or combo[2] == 'init_from_best_rotation_index':
                val_roca_dataset.config["data"]["sample_R"]["ignore_prediction"] = False
            else:
                val_roca_dataset.config["data"]["sample_R"]["ignore_prediction"] = True


            eval_roca_dict = {'what_translation':combo[0],'what_scale':combo[1],'what_rotation':combo[2],'what_retrieval':combo[3]}
            print('eval dict',eval_roca_dict)
            val_roca_dataset.eval_params = eval_roca_dict
            val_loss,val_accuracy,_,_ = one_epoch(val_roca_dataset,network,optimizer,criterion,epoch,writer,"val_roca",device,config,exp_path)


def main():
    torch.manual_seed(1)
    np.random.seed(0)

    if len(sys.argv) == 1:
        # so just name of script, no args
        network,optimizer,exp_path,config,device,start_epoch = start_new()
    elif len(sys.argv) == 2:
        network,optimizer,exp_path,config,device,start_epoch = resume_checkpoint(sys.argv[1])

    elif len(sys.argv) > 2:
        checkpoint_path,n_refinements = sys.argv[1],int(sys.argv[2])
        if len(sys.argv) == 3:
            index_rotation = 0
        elif len(sys.argv) == 4:
            index_rotation = int(sys.argv[3])
        network,optimizer,exp_path,config,device,start_epoch = evaluate_checkpoint(checkpoint_path,n_refinements,index_rotation,run_on_octopus=False,gpu_number=2)

    # checkpoint_path = '/scratches/octopus/fml35/experiments/regress_T/runs_03_T_big/date_2022_05_26_time_19_04_11_three_refinements/saved_models/epoch_255/model_state_dict.pth'
    # checkpoint_path = '/scratches/octopus/fml35/experiments/regress_T/runs_03_T_big/date_2022_06_06_time_12_50_24_predicted_depth/saved_models/network_epoch_000250.pth'
    # checkpoint_path = '/scratches/octopus/fml35/experiments/regress_T/runs_03_T_big/date_2022_06_06_time_12_53_16_gt_depth/saved_models/network_epoch_000250.pth'
    # checkpoint_path = '/scratches/octopus/fml35/experiments/regress_T/runs_18_T_and_R_and_S/date_2022_07_07_time_12_27_05_3_refinements_random_points_2d_random_points_3d_no_reprojected_as_query_no_cc/saved_models/network_epoch_000350.pth'

    # assert os.path.exists(checkpoint_path),'checkpoint path does not exist'
    # checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    # print('LOAD CHECKPOINT')
    # network.load_state_dict(checkpoint)


    writer = SummaryWriter(exp_path + '/log_files',max_queue=10000, flush_secs=600)
    train_dataset,val_dataset,val_roca_dataset,val_roca_dataset_all_images = create_data_loaders(config)
    criterion = None

    # print('debug')
    # with torch.no_grad():
    #     validate_roca(val_roca_dataset_all_images,network,optimizer,criterion,config["training"]["save_interval"],writer,device,config,exp_path)
        # validate_roca(val_roca_dataset,network,optimizer,criterion,config["training"]["save_interval"],writer,device,config,exp_path)

    # print('validate at start, rename otherwise later issue')
    # validate_roca(val_roca_dataset_all_images,network,optimizer,criterion,config["training"]["save_interval"],writer,device,config,exp_path)

    if len(sys.argv) > 2:
        with torch.no_grad():
            validate_roca(val_roca_dataset_all_images,network,optimizer,criterion,config["training"]["save_interval"],writer,device,config,exp_path)
            # validate_roca(val_roca_dataset,network,optimizer,criterion,config["training"]["save_interval"],writer,device,config,exp_path)

    for epoch in tqdm(range(start_epoch,config["training"]["n_epochs"])):

        train_loss,train_accuracy,all_extra_infos,all_losses = one_epoch(train_dataset,network,optimizer,criterion,epoch,writer,'train',device,config,exp_path)
        # train_loss,train_accuracy,all_extra_infos,all_losses = one_epoch(val_dataset,network,optimizer,criterion,epoch,writer,'train',device,config,exp_path)
        metric_dict = {"train_loss_last_epoch": train_loss,"train_accuracy_last_epoch":train_accuracy}
        if config["training"]["validate"]  == True:
            with torch.no_grad():
                # val_loss,val_accuracy,_,_ = one_epoch(val_dataset,network,optimizer,criterion,epoch,writer,"val",device,config,exp_path)
                
                if config["training"]["validate_roca"] == True:
                    validate_roca(val_roca_dataset_all_images,network,optimizer,criterion,epoch,writer,device,config,exp_path)
                    # validate_roca(val_roca_dataset,network,optimizer,criterion,epoch,writer,device,config,exp_path)

        if not 'small' in config['data']["dir_path_2d_train"]:
            save_checkpoint(exp_path + '/saved_models',epoch,network,optimizer,config)

    # log_hparams(writer,config,metric_dict)

    hparam_dict = {'bs': config["training"]["batch_size"],'lr': config["training"]["learning_rate"],'optimiser':config["training"]['optimiser'],'n_epochs':epoch}
    writer.add_hparams(hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None)
    # writer.close()

if __name__ == "__main__":
    main()