
# The first two functions are taken from https://github.com/facebookresearch/meshrcnn/blob/main/meshrcnn/utils/VOCap.py

import torch
import numpy as np
import json
import sys
import argparse
import os
from tqdm import tqdm
import random

def compute_ap(scores, labels, npos, device=None):
    if device is None:
        device = scores.device

    if len(scores) == 0:
        return 0.0
    tp = labels == 1
    fp = labels == 0
    sc = scores
    assert tp.size() == sc.size()
    assert tp.size() == fp.size()
    sc, ind = torch.sort(sc, descending=True)
    tp = tp[ind].to(dtype=torch.float32)
    fp = fp[ind].to(dtype=torch.float32)
    tp = torch.cumsum(tp, dim=0)
    fp = torch.cumsum(fp, dim=0)

    # # Compute precision/recall
    rec = tp / npos
    prec = tp / (fp + tp)
    ap = xVOCap(rec, prec, device)

    return ap


def xVOCap(rec, prec, device):

    z = rec.new_zeros((1))
    o = rec.new_ones((1))
    mrec = torch.cat((z, rec, o))
    mpre = torch.cat((z, prec, z))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    I = (mrec[1:] != mrec[0:-1]).nonzero()[:, 0] + 1
    ap = 0
    for i in I:
        ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap



def get_metrics_by_category(all_information,metrics,thresholds,categories):
    metrics_by_category = {}
    for threshold in thresholds:
        metrics_by_category[threshold] = {}

        for metric in metrics:
            metrics_by_category[threshold][metric] = {}
            for category in categories:
                metrics_by_category[threshold][metric][category] = {}
                metrics_by_category[threshold][metric][category]["mask_scores"] = []
                metrics_by_category[threshold][metric][category]["F"] = []


    for threshold in thresholds:

        already_covered_images = {}
        for metric in metrics:
            already_covered_images[metric] = []

        for i in range(len(all_information)):
            gt_cat = all_information[i]['gt_cat']
            predicted_cat = all_information[i]['predicted_cat']


        
            for metric in metrics:
                # add mask scores
                metrics_by_category[threshold][metric][predicted_cat]["mask_scores"].append(all_information[i]["mask_score"])
                # add F score
                if gt_cat == predicted_cat and all_information[i]["img"] not in already_covered_images[metric]:

                    value = 100 * float(all_information[i][metric])
                    
                    if value > threshold:
                        already_covered_images[metric].append(all_information[i]["img"])
                else:
                    value = 0

                metrics_by_category[threshold][metric][predicted_cat]["F"].append(value)

    return metrics_by_category

def compute_all_aps(metrics_by_category,categories_true_counters,thresholds):

    aps = {}
    # could put any threshold here
    for metric in metrics_by_category[50]:
        aps[metric] = {}
        for category in metrics_by_category[50][metric]:
            aps[metric][category] = {}

            for threshold in thresholds:
                labels = torch.tensor(metrics_by_category[threshold][metric][category]["F"]) > threshold
                scores = torch.tensor(metrics_by_category[threshold][metric][category]["mask_scores"])
                indices = list(range(len(labels)))
                random.shuffle(indices)

                ap_value = compute_ap(scores[indices],labels[indices],categories_true_counters[category])
                if torch.is_tensor(ap_value):
                    ap_value = ap_value.item()
                aps[metric][category][threshold] = ap_value * 100
                
    
            aps[metric][category]["mean"] = np.mean([aps[metric][category][threshold] for threshold in aps[metric][category]])

        aps[metric]["mean"] = {}
        for threshold in range(50,100,5):
            aps[metric]["mean"][threshold] = np.mean([aps[metric][category][threshold] for category in metrics_by_category[50][metric]])
        aps[metric]["mean"]["mean"] = np.mean([aps[metric]["mean"][threshold] for threshold in aps[metric]["mean"]])

    return aps


def get_categories_counter(gt_dir_path,categories):

    categories_dict = {}
    for cat in categories:
        categories_dict[cat] = 0

    for name in os.listdir(gt_dir_path):

        with open(gt_dir_path+ '/' + name,'r') as f:
            gt_infos = json.load(f)

        categories_dict[gt_infos["category"]] += 1
    return categories_dict

def collect_all_predictions(target_folder,metrics):

    collected_predictions = []

    for name in tqdm(os.listdir(target_folder + '/cropped_and_masked')):
        
        with open(target_folder + '/segmentation_infos/' + name.split('.')[0] + '.json','r') as f:
            segmentation = json.load(f)

        with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
            gt_infos = json.load(f)

        single_crop = {}
        single_crop["gt_cat"] = gt_infos["category"]
        single_crop["img"] = gt_infos["img"]
        single_crop["predicted_cat"] = segmentation["predictions"]["category"]
        single_crop["mask_score"] = segmentation["predictions"]["score"]

        with open(target_folder + '/bbox_overlap/' + name.split('.')[0] + '.json','r') as f:
            bbox_overlap = json.load(f)

        for metric in metrics:
            single_crop[metric] = bbox_overlap[metric]

        collected_predictions.append(single_crop)

    return collected_predictions


def write_aps(global_config):

    
    
    target_folder = global_config["general"]["target_folder"]
    categories = global_config["dataset"]["categories"]
    metrics = ['mask_iou','box_iou']
    # metrics = ['box_iou']
    thresholds = global_config["evaluate_poses"]["thresholds"]

    categories_counter  = get_categories_counter(target_folder + '/gt_infos',categories)
    all_predictions = collect_all_predictions(target_folder,metrics)
    metrics_by_category = get_metrics_by_category(all_predictions,metrics,thresholds,categories)

    aps = compute_all_aps(metrics_by_category,categories_counter,thresholds)

    aps_50 = {}

    for metric in aps:
        aps_50[metric] = {}
        for category in aps[metric]:
            aps_50[metric][category] = aps[metric][category][50]

    aps_mean = {}
    for metric in aps:
        aps_mean[metric] = {}
        for category in aps[metric]:
            aps_mean[metric][category] = aps[metric][category]["mean"]

    with open(target_folder  + '/global_stats/segmentation_ap_values.json', 'w') as f:
            json.dump(aps, f, indent=4)

    with open(target_folder  + '/global_stats/segmentation_ap_50_values.json', 'w') as f:
        json.dump(aps_50, f, indent=4)

    with open(target_folder  + '/global_stats/segmentation_ap_mean_values.json', 'w') as f:
        json.dump(aps_mean, f, indent=4)

    print(aps_mean)
    

if __name__ == '__main__':

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    write_aps(global_config)

    
    
    
    
  