
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

def compute_ap_return_extra(scores, labels, npos, device=None):
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

    return ap,rec,prec


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

                    F = float(all_information[i]["metrics"][metric])
                    
                    if F > threshold:
                        already_covered_images[metric].append(all_information[i]["img"])
                else:
                    F = 0

                metrics_by_category[threshold][metric][predicted_cat]["F"].append(F)

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

def find_mean_all_F1(metrics_by_category,metric):

    all_F1 = []
    for category in metrics_by_category[50][metric]:
        for F in metrics_by_category[50][metric][category]["F"]:
            all_F1.append(float(F))

    return np.mean(np.array(all_F1)), len(all_F1)


def get_distance(x1,x2):
    return np.sum((np.array(x1) - np.array(x2))**2)**0.5

def get_angle(m1,m2):

    m = np.matmul(np.array(m1).T,np.array(m2))

    value = (np.trace(m) - 1 )/ 2

    clipped_value = np.clip(value,-0.9999999,0.999999)

    angle = np.arccos(clipped_value)

    return angle * 180 / np.pi 


def get_pose_diffs_by_category(all_information,number_nn,indicator):


    categories = ['bed','bookcase','chair','desk','misc','sofa','table','tool','wardrobe']
    angle_dist_by_category = {}
    for cat in categories:
        angle_dist_by_category[cat] = {}
        for kind in ['correct_model','correct_category']:
            angle_dist_by_category[cat][kind] = {}
            for metric in ['angle','distance']:
                angle_dist_by_category[cat][kind][metric] = []



    for i in range(len(all_information)):

        if all_information[i]["mask_info"]:

            list_of_reprojection_distances = [all_information[i]["nearest_neighbours"][j]["best_"+indicator][indicator] for j in range(min(number_nn,len(all_information[i]["nearest_neighbours"])))]
            best_value = min(list_of_reprojection_distances)
            best_index = list_of_reprojection_distances.index(best_value)

            r_pred = all_information[i]["nearest_neighbours"][best_index]["best_"+indicator]['predicted_r']
            t_pred = all_information[i]["nearest_neighbours"][best_index]["best_"+indicator]['predicted_t']
            r_gt = all_information[i]["annotations"]['rot_mat']
            t_gt = all_information[i]["annotations"]['trans_mat']

            distance = get_distance(t_pred,t_gt)
            angle = get_angle(r_pred,r_gt)

            gt_cat = all_information[i]["annotations"]['category']
            predicted_cat = all_information[i]["mask_info"][0]["predicted_category"]

            if gt_cat == predicted_cat:
                angle_dist_by_category[gt_cat]['correct_category']['angle'].append(angle)
                angle_dist_by_category[gt_cat]['correct_category']['distance'].append(distance)

            if all_information[i]["annotations"]["model_full_name"] == all_information[i]["nearest_neighbours"][best_index]["name"]:
                angle_dist_by_category[gt_cat]['correct_model']['angle'].append(angle)
                angle_dist_by_category[gt_cat]['correct_model']['distance'].append(distance)
            
    return angle_dist_by_category


def get_mean_angle_dist_by_category(angle_dist_by_category):

    mean_angle_dist_by_category = {}
    for cat in angle_dist_by_category:
        mean_angle_dist_by_category[cat] = {}
        for kind in ['correct_model','correct_category']:
            mean_angle_dist_by_category[cat][kind] = {}
            for metric in ['angle','distance']:
                mean_angle_dist_by_category[cat][kind][metric] = {}
                mean_angle_dist_by_category[cat][kind][metric]['mean'] = np.mean(angle_dist_by_category[cat][kind][metric])
                mean_angle_dist_by_category[cat][kind][metric]['total number'] = len(angle_dist_by_category[cat][kind][metric])

    return mean_angle_dist_by_category


def get_categories_counter(gt_dir_path,categories):

    categories_dict = {}
    for cat in categories:
        categories_dict[cat] = 0

    for name in os.listdir(gt_dir_path):

        with open(gt_dir_path+ '/' + name,'r') as f:
            gt_infos = json.load(f)

        categories_dict[gt_infos["category"]] += 1
    return categories_dict


def collect_all_predictions(target_folder):
    collected_predictions = []

    print('need to do change in here if want multiple correct predictions per image')
    print('collect predictions')
    for name in tqdm(os.listdir(target_folder + '/bbox_overlap')):
        with open(target_folder + '/bbox_overlap/' + name.split('.')[0] + '.json','r') as f:
            bbox_overlap = json.load(f)

        if bbox_overlap['valid']:

            with open(target_folder + '/segmentation_infos/' + name.split('.')[0] + '.json','r') as f:
                segmentation = json.load(f)

            with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
                gt_infos = json.load(f)

            single_crop = {}
            single_crop["gt_cat"] = gt_infos["category"]
            single_crop["img"] = gt_infos["img"]
            single_crop["predicted_cat"] = segmentation["predictions"]["category"]
            single_crop["mask_score"] = segmentation["predictions"]["score"]

            if bbox_overlap['correct_category']:

                with open(target_folder + '/selected_nn/' + name,'r') as f:
                    selected = json.load(f)
                

                with open(target_folder + '/metrics/' + name.split('.')[0] + '_' + str(selected["selected_nn"]).zfill(3) + '_' + str(selected["selected_orientation"]).zfill(2) + '.json','r') as f:
                    metrics = json.load(f)

                # with open(target_folder + '/metrics/' + name.split('.')[0] + '_' + str(selected["selected_nn"]).zfill(3) + '_' + str(selected["best_orientation"]).zfill(2) + '.json','r') as f:
                #     metrics = json.load(f)

                single_crop["metrics"] = metrics

            collected_predictions.append(single_crop)

    return collected_predictions


# def collect_all_predictions(target_folder,top_n_retrieval,indicator):

#     collected_predictions = []

#     print('need to do change in here if want multiple correct predictions per image')
#     print('collect predictions')
#     for name in tqdm(os.listdir(target_folder + '/cropped_and_masked')):
#         try:
#             with open(target_folder + '/segmentation_infos/' + name.split('.')[0] + '.json','r') as f:
#                 segmentation = json.load(f)

#             with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
#                 gt_infos = json.load(f)

#             single_crop = {}
#             single_crop["gt_cat"] = gt_infos["category"]
#             single_crop["img"] = gt_infos["img"]
#             single_crop["predicted_cat"] = segmentation["predictions"]["category"]
#             single_crop["mask_score"] = segmentation["predictions"]["score"]
#             single_crop["metrics"] = []

#             for i in range(top_n_retrieval):

#                 with open(target_folder + '/metrics/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json','r') as f:
#                     metrics = json.load(f)

#                 with open(target_folder + '/poses/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json','r') as f:
#                     poses = json.load(f)

#                 metrics[indicator] = poses[indicator]
                
#                 single_crop["metrics"].append(metrics)

#             collected_predictions.append(single_crop)
#         except:
#             pass

#     return collected_predictions


        
def write_aps(global_config):

    
    
    target_folder = global_config["general"]["target_folder"]

    metrics = global_config["evaluate_poses"]["metrics"]
    thresholds = global_config["evaluate_poses"]["thresholds"]
    number_nn = global_config["keypoints"]["matching"]["top_n_retrieval"]
    categories = global_config["dataset"]["categories"]

   
    categories_counter  = get_categories_counter(target_folder + '/gt_infos',categories)

    all_predictions = collect_all_predictions(target_folder)
    # filtered_predictions = filter_predictions_mask_score(all_predictions)


    metrics_by_category = get_metrics_by_category(all_predictions,metrics,thresholds,categories)
    # metrics_by_category = get_metrics_by_category(filtered_predictions,metrics,number_nn,thresholds,categories,indicator,min_or_max)
    
    mean_F1, n_examples = find_mean_all_F1(metrics_by_category,metrics[0])
    line = 'Number examples: {} , Mean {} score: {}'.format(n_examples,metrics[0],mean_F1)

    with open(target_folder  + '/global_stats/mean_F1.txt', 'w') as f:
        f.write(line)

    with open(target_folder  + '/global_stats/metrics_by_category.json', 'w') as f:
        json.dump(metrics_by_category[50], f, indent=4)

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

    with open(target_folder  + '/global_stats/ap_values.json', 'w') as f:
            json.dump(aps, f, indent=4)

    with open(target_folder  + '/global_stats/ap_50_values.json', 'w') as f:
        json.dump(aps_50, f, indent=4)

    with open(target_folder  + '/global_stats/ap_mean_values.json', 'w') as f:
        json.dump(aps_mean, f, indent=4)
    
    print(aps_mean)
    cats = ['mean'] + categories
    print(cats)
    values =''
    for cat in cats:
        values += str(np.round(aps_mean['F1@0.300000'][cat],2)).replace('.',',') + ';'
    print(values)
    # angle_dist_by_category = get_pose_diffs_by_category(all_information,config["number_nearest_neighbours"],indicator)
    # mean_angle_dist_by_category = get_mean_angle_dist_by_category(angle_dist_by_category)

    # with open(exp_path  + '/angle_dist_by_category.json','w') as f:
    #     json.dump(angle_dist_by_category, f, indent=4)
    # with open(exp_path  + '/mean_angle_dist_by_category.json','w') as f:
    #     json.dump(mean_angle_dist_by_category, f, indent=4)




if __name__ == '__main__':

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    write_aps(global_config)

    
    
    
    
  