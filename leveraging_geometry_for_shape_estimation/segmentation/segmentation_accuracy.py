import torch
import numpy as np
import json
import sys
import argparse

sys.path.append('/home/mifs/fml35/code/shape/retrieval_plus_keypoints/pose_prediction/recover_pose/code/metrics')
# from metrics.VOCap import compute_ap
from VOCap import compute_ap

def get_metrics_by_category(all_information,metrics,number_nn,indicator,thresholds):
    categories = ['bed','bookcase','chair','desk','misc','sofa','table','tool','wardrobe']
    metrics_by_category = {}
    for threshold in thresholds:
        metrics_by_category[threshold] = {}

        for metric in metrics:
            metrics_by_category[threshold][metric] = {}
            for category in categories:
                metrics_by_category[threshold][metric][category] = {}
                metrics_by_category[threshold][metric][category]["mask_scores"] = []
                metrics_by_category[threshold][metric][category]["F"] = []

    
    print('en(all_information)',len(all_information))

    for threshold in thresholds:

        already_covered_images = {}
        for metric in metrics:
            already_covered_images[metric] = []

        for i in range(len(all_information)):
            gt_cat = all_information[i]["annotations"]['category']
            predicted_cat = all_information[i]["mask_info"][0]["predicted_category"]
        
            for metric in metrics:
                # add mask scores
                if all_information[i]["mask_info"]:
                    metrics_by_category[threshold][metric][predicted_cat]["mask_scores"].append(all_information[i]["mask_info"][0]["score"])
                    # add F score
                    if gt_cat == predicted_cat and all_information[i]["img"] not in already_covered_images[metric]:

                        list_of_reprojection_distances = [all_information[i]["nearest_neighbours"][j]["best_"+indicator][indicator] for j in range(min(number_nn,len(all_information[i]["nearest_neighbours"])))]
                        best_index = min(list_of_reprojection_distances)


                        best_nn = list_of_reprojection_distances.index(best_index)

                        F = float(all_information[i]["nearest_neighbours"][best_nn]["best_"+indicator][metric])
                        
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
                if threshold == 50:
                    print(category)
                    print(np.sum((labels * 1).numpy()))
                    print(labels.shape)
                ap_value = compute_ap(torch.tensor(metrics_by_category[threshold][metric][category]["mask_scores"]),labels,categories_true_counters[category])
                if torch.is_tensor(ap_value):
                    ap_value = ap_value.item()
                aps[metric][category][threshold] = ap_value * 100
                
    
            aps[metric][category]["mean"] = np.mean([aps[metric][category][threshold] for threshold in aps[metric][category]])

        aps[metric]["mean"] = {}
        for threshold in range(50,100,5):
            aps[metric]["mean"][threshold] = np.mean([aps[metric][category][threshold] for category in metrics_by_category[50][metric]])
        aps[metric]["mean"]["mean"] = np.mean([aps[metric]["mean"][threshold] for threshold in aps[metric]["mean"]])

    return aps

def find_mean_all_F1(metrics_by_category):
    for metric in metrics_by_category[50]:
        all_F1 = []
        for category in metrics_by_category[50][metric]:
            for F in metrics_by_category[50][metric][category]["F"]:
                all_F1.append(float(F))

    return np.mean(np.array(all_F1)), len(all_F1)


 
def write_aps(config,exp_path,metrics,top_1_mask=False):

    categories_counter = {}
    categories_counter['s1'] = {'bed': 213, 'bookcase': 79, 'chair': 1165, 'desk': 154, 'misc': 20, 'sofa': 415, 'table': 419, 'tool': 11, 'wardrobe': 54}
    categories_counter['s2'] = {'bed': 218, 'bookcase': 84, 'chair': 777, 'desk': 205, 'misc': 20, 'sofa': 504, 'table': 392, 'tool': 22, 'wardrobe': 134}

    if top_1_mask == True:
        pose_path = exp_path + '/pose_information_top_1_mask.json'
        add_on = '_top_1_mask'
    elif top_1_mask == False:
        pose_path = exp_path + '/pose_information.json'
        add_on = ''

    with open(pose_path,'r') as f:
        all_information = json.load(f)

    thresholds = range(50,100,5)
    print(len(all_information[0]["nearest_neighbours"]))


    setting_to_metric = {'segmentation': 'avg_dist_furthest', 'keypoints': 'avg_dist_reprojected_keypoints', 'combined':'combined','meshrcnn':'meshrcnn','F1':'F1'}
    indicator = setting_to_metric[config["choose_best_based_on"]]
    indicators = [indicator]

    for indicator in indicators:

        metrics_by_category = get_metrics_by_category(all_information,metrics,config["number_nearest_neighbours"],indicator,thresholds)
        
        mean_F1, n_examples = find_mean_all_F1(metrics_by_category)
        line = 'Number examples: {} , Mean F1 score: {}'.format(n_examples,mean_F1)

        print(line)
        with open(exp_path  + '/mean_F1_{}top1{}.txt'.format(indicator,add_on), 'w') as f:
            f.write(line)


        with open(exp_path  + '/metrics_by_category_{}{}.json'.format(indicator,add_on), 'w') as f:
            json.dump(metrics_by_category[50], f, indent=4)

        aps = compute_all_aps(metrics_by_category,categories_counter[config["split"]],thresholds)

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
        


        print(indicator)
        print(aps_50)
        print(aps_mean)

        with open(exp_path  + '/ap_values_top_{}{}.json'.format(config["number_nearest_neighbours"],add_on), 'w') as f:
            json.dump(aps, f, indent=4)

        with open(exp_path  + '/ap_50_values_top_{}{}.json'.format(config["number_nearest_neighbours"],add_on), 'w') as f:
            json.dump(aps_50, f, indent=4)

        with open(exp_path  + '/ap_mean_values_top_{}{}.json'.format(config["number_nearest_neighbours"],add_on), 'w') as f:
            json.dump(aps_mean, f, indent=4)



if __name__ == '__main__':


    
    metrics = ['mask_iou','box_iou']

    write_aps(config,args.exp_path,metrics,args.top_1_mask)

    
    
    
    
  