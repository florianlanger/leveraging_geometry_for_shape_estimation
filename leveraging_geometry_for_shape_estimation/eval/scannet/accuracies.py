import json
import tqdm
import os
import numpy as np



def compute_accuracies(global_config):

    target_folder = global_config["target_folder"]
    for name in tqdm(os.listdir(target_folder + '/metrics_scannet')):
        with open(target_folder + '/metrics_scannet/' + name,'r') as f:
            metrics_scannet = json.load(f)

        with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
            gt_infos = json.load(f)

if __name__ == '__main__':

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    compute_accuracies(global_config)