# import pkg_resources
# pkg_resources.require("torch==1.10.0+cu113")


from cProfile import label
from webbrowser import get
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler
import json
import os
import numpy as np
import shutil
from tqdm import tqdm
from datetime import datetime
from torchvision.models import resnet18,vgg16
from torchvision.transforms import ColorJitter
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from perceiver_pytorch import Perceiver, PerceiverIO
import torch_optimizer as optim_special
from transformers import PerceiverModel, PerceiverConfig
import sys

import time
import psutil

# from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.model.perceiver_pytorch_local.perceiver_pytorch.perceiver_pytorch import Perceiver as Perceiver_local

from leveraging_geometry_for_shape_estimation.utilities.dicts import load_json
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.utiliies import create_directories
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.model.image_network import Classification_network
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.model.pointnet.pointnet2_cls_msg import get_model
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.dataset import Dataset_lines
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.batch_sampler import BatchSampler_repeat,SequentialSampler_custom
# from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.dataset_points_v2 import Dataset_points
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.dataset_points_correspondences import Dataset_points_correspondences
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.validation_grid import Dataset_val_grid
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.visualisation.visualisation_main import visualise_preds
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.visualisation.confusion_matrix import visualise_confusion_matrices
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.metrics import track_cat_accuracies, get_distances_per_point
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.visualisation.visualisation_correspondences import plot_offsets_preds
from leveraging_geometry_for_shape_estimation.data_conversion.create_dirs import dict_replace_value
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.validation_grid import validate_grid
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.losses.loss import get_combined_loss

def one_epoch_classifier(data_loader,net,optimizer,criterion,epoch,writer,kind,device,config,exp_path):
    

    all_metrics = {'all_predictions':[],'all_labels':[],'all_categories':[],'all_roca_bbox':[],'all_losses':[],'all_distances':[],'all_extra_infos':[],'t_distance': [], 's_distance': [],'t_correct': [], 's_correct': [],'running_loss':0.0,'n_correct_total':0,'counter_examples':0} 

    if kind == 'train':
        net.train()
    elif kind == 'val':
        net.eval()

    # for i, data in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
    for num_batch, data in enumerate(data_loader, 0):
        t1 = time.time()
        inputs, targets, extra_infos = data
        if config["data"]["type"] == 'lines':
            inputs = torch.permute(inputs, (0,3,1,2))
        elif config["data"]["type"] == 'points':
            if config['model']['type'] == 'pointnet':
                inputs = inputs.transpose(2, 1)

        inputs = inputs.to(device)    

        targets = targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)

        
        loss,latest_metrics = get_combined_loss(outputs,targets,config)
        mean_loss = torch.mean(loss)

        if kind == 'train':
            mean_loss.backward()
            optimizer.step()

        all_metrics = update_running_metrics(all_metrics,latest_metrics,extra_infos,loss)

        if num_batch == 0 and epoch % config['training']["vis_interval"] == 0:
            visualise_preds(writer,latest_metrics,inputs,net,config,kind,epoch,extra_infos,exp_path,num_batch)
    if epoch % config['training']["vis_interval"] == 0:
        visualise_confusion_matrices(all_metrics['all_predictions'],all_metrics['all_labels'],all_metrics['all_categories'],writer,epoch,kind)

    track_cat_accuracies(all_metrics,writer,epoch,kind)
    writer.add_scalar(kind + ' loss',all_metrics['running_loss'] / all_metrics['counter_examples'],epoch)
    track_gpu_usage(writer,epoch)

    return all_metrics['running_loss'] / all_metrics['counter_examples'], all_metrics['n_correct_total']/ all_metrics['counter_examples'], all_metrics['all_extra_infos'],all_metrics['all_losses']


def update_running_metrics(all_metrics,latest_metrics,extra_infos,loss):


    all_metrics['all_predictions'] += latest_metrics['probabilities'].tolist()
    all_metrics['all_labels'] += latest_metrics['labels'].tolist()
    all_metrics['all_categories'] += extra_infos['category']
    all_metrics['all_roca_bbox'] += extra_infos['roca_bbox']
    all_metrics['all_losses'] += loss.tolist()
    all_metrics['all_extra_infos'].append(extra_infos)
    all_metrics['t_distance'] += latest_metrics['t_distance'].tolist()
    all_metrics['s_distance'] += latest_metrics['s_distance'].tolist()
    all_metrics['t_correct'] += latest_metrics['t_correct'].tolist()
    all_metrics['s_correct'] += latest_metrics['s_correct'].tolist()

    all_metrics['counter_examples'] += loss.shape[0]
    all_metrics['running_loss'] += torch.sum(loss).item()
    all_metrics['n_correct_total'] += torch.sum(latest_metrics['correct']).item()

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


def set_device(config):
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config["general"]["gpu"]))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        print('on cpu')
    return device

def create_data_loaders(config):

    if config["data"]["type"] == 'points' and config["data"]["what_models"] == "lines":
        from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.dataset_points import Dataset_points
    elif config["data"]["type"] == 'points' and config["data"]["what_models"] == "points_and_normals":
        from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.dataset_points_v2 import Dataset_points

    if config["data"]["type"] == 'points':
        if config['data']['targets'] == 'labels':
            train_dataset = Dataset_points(config,kind='train')
            val_dataset = Dataset_points(config,kind='val')
        elif config['data']['targets'] == 'offsets':
            train_dataset = Dataset_points_correspondences(config,kind='train')
            val_dataset = Dataset_points_correspondences(config,kind='val')
        # val_dataset = Dataset_points(config,kind='train')
        val_loader_grid = None
    else:
        train_dataset = Dataset_lines(config,kind='train')
        val_dataset = Dataset_lines(config,kind='val')
        # val_dataset = Dataset_lines(config,kind='train')
        # val_dataset_grid = Dataset_val_grid(config,kind='val')
        # val_loader_grid = DataLoader(val_dataset_grid, batch_size = config["training"]["batch_size_val"],shuffle=False)
        val_loader_grid = None

    print('Len train dataset',len(train_dataset))
    sampler_train =SequentialSampler_custom(repeats=config["training"]["n_same_objects_per_batch"],n_examples=len(train_dataset))
    sampler_val = SequentialSampler_custom(repeats=config["training"]["n_same_objects_per_batch"],n_examples=len(val_dataset))

    # train_loader = DataLoader(train_dataset, batch_size = config["training"]["batch_size"],shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size = config["training"]["batch_size_val"],shuffle=True)

    train_loader = DataLoader(train_dataset, sampler=sampler_train,batch_size = config["training"]["batch_size"])
    val_loader = DataLoader(val_dataset, sampler=sampler_val,batch_size = config["training"]["batch_size_val"])

    return train_loader, val_loader, val_loader_grid


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
    checkpoint = {'epoch': epoch,'model_state_dict': network.state_dict(),'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, dir_path + '/last_epoch.pth'.format(str(epoch).zfill(6)))
    if epoch % config["training"]["save_interval"] == 0:
        torch.save(checkpoint, dir_path + '/epoch_{}.pth'.format(str(epoch).zfill(6)))



def process_config(config):

    n_rgb = (np.sum([config["data"]["use_rgb"],config["data"]["use_normals"],config["data"]["use_alpha"]]) + 1)
    if "n_same_objects_per_batch" in config["training"]:
        assert config["training"]["batch_size"] % config["training"]["n_same_objects_per_batch"] == 0
        assert config["training"]["batch_size_val"] % config["training"]["n_same_objects_per_batch"] == 0
    # assert config["training"]["batch_size"] % n_rgb == 0
    # assert config["training"]["batch_size_val"] % n_rgb == 0

    # if config["data"]["img_size"] == [256,192]:
    #     config["training"]["batch_size"] = config["training"]["batch_size"] / 2
    #     config["training"]["batch_size_val"] = config["training"]["batch_size_val"] / 2

    # config["training"]["batch_size"] = int(config["training"]["batch_size"] / n_rgb)
    # config["training"]["batch_size_val"] = int(config["training"]["val_grid_points_per_example"] * (10 // n_rgb))



    if config['model']["regress_offsets"] == False:
        config["model"]["n_outputs"] = 1
    elif config['model']["regress_offsets"] == True:
        config["model"]["n_outputs"] = 7

    if not "sample_wrong_R_percentage" in config["data_augmentation"]:
        config["data_augmentation"]["sample_wrong_R_percentage"] = 0.0

    if not "dir_path_scan2cad_anno" in config["data"]:
        config["data"]["dir_path_scan2cad_anno"] = "/scratch2/fml35/datasets/scannet/scan2cad_annotations/full_annotations.json"

    if config["general"]["run_on_octopus"] == False:
        config = dict_replace_value(config,'/scratch/fml35/','/scratches/octopus/fml35/')
        config = dict_replace_value(config,'/scratch2/fml35/','/scratches/octopus_2/fml35/')

    if 'type' not in config['data']:
        config['data']['type'] = 'lines'

    if config["model"]["type"] == "vgg16":
        config["training"]["optimiser"] = "Adam"
        config["training"]["learning_rate"] = 0.00005
    elif config["model"]["type"] == "perceiver":
        config["training"]["optimiser"] = "Lamb"
        config["training"]["learning_rate"] = 0.001

    if config["data"]["what_models"] == "lines":
        config["data"]["dims_per_pixel"] = 3
    elif config["data"]["what_models"] == "points_and_normals":
        config["data"]["dims_per_pixel"] = 7

    if config["data"]["sample_what"] == 'T':
        # config["data"]["sample_T"] = {"percent_small": 0.4,"percent_large": 0.5,"threshold_correct_T": 0.2}
        print('sampling for offset T')
        config["data"]["sample_T"] = {"percent_small": 0.0,"percent_large": 0.0,"threshold_correct_T": 0.2}
        config["data"]["sample_S"] = {"use_gt": True,"percent_small": 0.7,"limit_small": 0.2,"percent_large": 0.2,"limit_large": 0.5}

    elif config["data"]["sample_what"] == 'T_and_S':
        config["data"]["sample_T"] = {"percent_small": 0.7,"percent_large": 0.2,"threshold_correct_T": 0.2}
        config["data"]["sample_S"] = {"use_gt": False,"percent_small": 0.7,"limit_small": 0.2,"percent_large": 0.2,"limit_large": 0.5}

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


def one_epoch(data_loader,net,optimizer,criterion,epoch,writer,kind,device,config,exp_path):
    if config['data']['targets'] == 'labels':
        return one_epoch_classifier(data_loader,net,optimizer,criterion,epoch,writer,kind,device,config,exp_path)
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


# write a functipn that tracks gpu usage, cpu usage and memory usage in tensorboard
def track_gpu_usage(writer,step):

    gpu_stat_names = ['gpu_memory_allocated','gpu_memory_reserved','gpu_memory_allocated','gpu_max_memory_allocated','gpu_max_memory_cached','gpu_max_memory_allocated']
    gpu_stat_values = [torch.cuda.memory_allocated(),torch.cuda.memory_reserved(),torch.cuda.memory_allocated(),torch.cuda.max_memory_allocated(),torch.cuda.max_memory_cached(),torch.cuda.max_memory_allocated()]

    cpu_stat_names = ['cpu_percent']
    cpu_stat_values = [psutil.cpu_percent()]


    all_stats_values = gpu_stat_values + cpu_stat_values
    all_stats_names = gpu_stat_names + cpu_stat_names

    for i in range(len(all_stats_names)):
        writer.add_scalar(all_stats_names[i],all_stats_values[i],global_step=step)



def main():
    torch.manual_seed(1)
    np.random.seed(0)

    if len(sys.argv) == 1:
        # so just name of script, no args
        network,optimizer,exp_path,config,device,start_epoch = start_new()
    elif len(sys.argv) == 2:
        network,optimizer,exp_path,config,device,start_epoch = resume_checkpoint(sys.argv[1])

    writer = SummaryWriter(exp_path + '/log_files',max_queue=10000, flush_secs=600)
    train_loader,val_loader,val_loader_grid = create_data_loaders(config)
    criterion = None


    for epoch in tqdm(range(start_epoch,config["training"]["n_epochs"])):

        # with torch.no_grad():
        #     validate_grid(val_loader_grid,network,epoch,writer,device,config)
        train_loss,train_accuracy,all_extra_infos,all_losses = one_epoch(train_loader,network,optimizer,criterion,epoch,writer,'train',device,config,exp_path)
        train_loader.sampler.shuffle_indices()
        train_loader.dataset.update_half_width_sampling_cube(epoch)
        train_loader.dataset.update_hard_examples(all_losses,all_extra_infos)
        train_loader.dataset.epoch = epoch + 1
        metric_dict = {"train_loss_last_epoch": train_loss,"train_accuracy_last_epoch":train_accuracy}
        if config["training"]["validate"]  == True:
            with torch.no_grad():
                val_loss,val_accuracy,_,_ = one_epoch(val_loader,network,optimizer,criterion,epoch,writer,"val",device,config,exp_path)
                # if epoch % 20 == 0 and not (config["data"]["type"] == 'points'):
                #     validate_grid(val_loader_grid,network,epoch,writer,device,config)
                #     metric_dict["val_loss_last_epoch"] = val_loss
                #     metric_dict["val_accuracy_last_epoch"] = val_accuracy
            val_loader.dataset.update_half_width_sampling_cube(epoch)
            val_loader.sampler.shuffle_indices()

        if not 'small' in config['data']["dir_path_2d_train"]:
            save_checkpoint(exp_path + '/saved_models',epoch,network,optimizer,config)
    # log_hparams(writer,config,metric_dict)

    hparam_dict = {'bs': config["training"]["batch_size"],'lr': config["training"]["learning_rate"],'optimiser':config["training"]['optimiser'],'n_epochs':epoch}
    writer.add_hparams(hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None)
    # writer.close()

if __name__ == "__main__":
    # print('validate,change dataset val and train,change len data loader and getitem')
    print('put modulo back in len dataset')
    main()