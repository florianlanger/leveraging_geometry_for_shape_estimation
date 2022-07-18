
from audioop import add
from ftplib import all_errors
from pickletools import optimize
from re import I
from pytz import all_timezones
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

from leveraging_geometry_for_shape_estimation.utilities.dicts import load_json
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.main_v2 import process_config,create_shuffled_indices
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.dataset_points_v2 import Dataset_points
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.batch_sampler import get_batch_from_dataset

def create_save_dirs(exp_path):
    os.mkdir(exp_path)
    os.mkdir(exp_path+'/inputs')
    os.mkdir(exp_path+'/targets')
    os.mkdir(exp_path+'/extras')


def save_batch(list_inputs,list_targets,list_extras,i,exp_path):

    all_inputs = torch.cat(list_inputs,dim=0)
    all_targets = torch.cat(list_targets,dim=0)
    all_extras = {}
    for key in list_extras[0]:
        concat_info = np.concatenate([item[key] for item in list_extras])
        # print('concat_info.shape',concat_info.shape)
        all_extras[key] = concat_info.tolist()

    all_inputs = all_inputs.to(torch.float32)

    # print('list_inputs 0 shape: {}'.format(list_inputs[0].shape))
    # print('list targets 0 shape: {}'.format(list_targets[0].shape))

    # print('all inputs shape: {}'.format(all_inputs.shape))
    # print('all targets shape: {}'.format(all_targets.shape))

    torch.save(all_inputs,exp_path+'/inputs/{}.pt'.format(str(i).zfill(6)))
    torch.save(all_targets,exp_path+'/targets/{}.pt'.format(str(i).zfill(6)))
    with open(exp_path+'/extras/{}.json'.format(str(i).zfill(6)),'w') as f:
        json.dump(all_extras,f)


def create_all_indices(dataset,n_epochs):
    len_train_data = len(dataset)
    all_indices = np.arange(len_train_data)
    all_indices = np.repeat(all_indices,n_epochs)
    random.shuffle(all_indices)
    return all_indices

def create_train_data(dataset,config,exp_path,n_epochs,bs,save_interval_batch):
    
    all_indices = create_all_indices(dataset,n_epochs)

    N_batches = len(dataset) / bs * n_epochs
    print('N_batches: {}'.format(N_batches))
    assert N_batches.is_integer()

    
    list_inputs = []
    list_targets = []
    list_extras = []

    chunk = 0

    for i in tqdm(range(1,int(N_batches)+1)):
        indices = all_indices[i*bs:(i+1)*bs]
        input,target,extra = get_batch_from_dataset(dataset,indices)

        list_inputs.append(input.to('cpu'))
        list_targets.append(target.to('cpu'))
        list_extras.append(extra)

        if i % save_interval_batch == 0:
            print('Saving batch {}'.format(i))
            save_batch(list_inputs,list_targets,list_extras,chunk,exp_path)
            list_inputs = []
            list_targets = []
            list_extras = []
            chunk += 1


def main():

    n_epochs = 1000
    bs = 10
    save_interval_batch = 100
    # out_dir = '/scratches/octopus/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/precomputed_train_data'
    out_dir = '/data/cornucopia/fml35/datasets/own_datasets/pose_refinement_precopmuted_train_data_scannet'

    torch.manual_seed(1)
    np.random.seed(0)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = load_json('{}/../../config.json'.format(dir_path))
    config = process_config(config)
    exp_path = '{}/{}_{}'.format(out_dir,datetime.now().strftime("date_%Y_%m_%d_time_%H_%M_%S"),config["general"]["name"])
    os.environ["CUDA_VISIBLE_DEVICES"] = config["general"]["gpu"]

    dataset = Dataset_points(config,kind='train')
    create_save_dirs(exp_path)
    
    create_train_data(dataset,config,exp_path,n_epochs,bs,save_interval_batch)




if __name__ == "__main__":
    main()