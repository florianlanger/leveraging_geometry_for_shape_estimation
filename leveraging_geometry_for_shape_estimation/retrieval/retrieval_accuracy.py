import numpy as np
import sys
import os
import json

def get_first_correct(nearest_neighbours,gt_model_name):

    for i,nn in enumerate(nearest_neighbours["nearest_neighbours"]):
        if nn["name"] == gt_model_name:
            return i + 1
    return 10000  

def compute_accuracies(all_nn_infos,all_gt_models,accuracies,categories):
    first_correct_by_category = {}
    for cat in categories:
        first_correct_by_category[cat] = []


    print(first_correct_by_category)


    for i,nearest_neighbours in enumerate(all_nn_infos):
        first_correct = get_first_correct(nearest_neighbours,all_gt_models[i])
        gt_cat = all_gt_models[i].split('_')[0]
        first_correct_by_category[gt_cat].append(first_correct)
        first_correct_by_category['total'].append(first_correct)
        
    accuracies_by_category = {}
    for cat in categories:
        accuracies_by_category[cat] = {}
        for accuracy in accuracies:
            accuracies_by_category[cat][accuracy] = sum(np.array(first_correct_by_category[cat]) <= accuracy)/np.float64(len(first_correct_by_category[cat]))
        accuracies_by_category[cat]['total number'] = len(first_correct_by_category[cat])
    
    return accuracies_by_category

    



if __name__ == "__main__":    
    
    accuracies = [1,5,10,15,20,100] # top_n

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    categories = global_config["dataset"]["categories"] + ['total']
    print(categories)

    target_folder = global_config["general"]["target_folder"]


    all_nn_infos = []
    all_gt_models = []
    print('before list dir')
    for name in os.listdir(target_folder + '/nn_infos'):

        with open(target_folder + '/nn_infos/' + name,'r') as file:
            all_nn_infos.append(json.load(file))

        gt_file = name.replace('_' + name.split('_')[-1],'') + '.json'
        with open(target_folder + '/gt_infos/' + gt_file,'r') as file:  
            gt_infos = json.load(file)

        split_model = gt_infos["model"].split('/')
        gt_model = split_model[1] + '_' + split_model[2]
        all_gt_models.append(gt_model)

    accuracy_by_category = compute_accuracies(all_nn_infos,all_gt_models,accuracies,categories)

    with open(target_folder + '/global_stats/retrieval_accuracy_all_embeds.json', 'w') as f:
        json.dump(accuracy_by_category, f,indent=4)
    
    print(accuracy_by_category["total"])