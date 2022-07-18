import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import json
import numpy as np
import sys
from tqdm import tqdm


def get_categories_counter(gt_dir_path,categories):

    categories_dict = {}
    for cat in categories:
        categories_dict[cat] = 0

    for name in os.listdir(gt_dir_path):

        with open(gt_dir_path+ '/' + name,'r') as f:
            gt_infos = json.load(f)

        for object in gt_infos["objects"]:
            categories_dict[object["category"]] += 1
    return categories_dict


def plot_analysis(cats_gt,cats_correct,cats_false,cats_double,save_path):

    gt = np.array([cats_gt[cat] for cat in cats_gt])
    correct = np.array([cats_correct[cat] for cat in cats_correct]).astype(float)
    false = np.array([cats_false[cat] for cat in cats_false]).astype(float)
    double = np.array([cats_double[cat] for cat in cats_double]).astype(float)
    
    
    width = 0.2

    x = np.array(range(len(cats_gt)))
    

    ax = plt.subplot(111)
    ax.bar(x-width, correct/gt, width=width, color='g', align='center')
    ax.bar(x, false/gt, width=width, color='r', align='center')
    ax.bar(x+width, double/gt, width=width, color='b', align='center')

    # ax.set_xticklabels([cat for cat in cats_best_possible],minor=True)
    plt.xticks(x, [cat for cat in cats_gt])

    # rects = ax.patches
    # for rect, label in zip(rects, list(total)):
    #     height = rect.get_height()
    #     ax.text(
    #         rect.get_x() + rect.get_width() / 2, height + 0.05, label, ha="center", va="bottom"
    #     )

    plt.savefig(save_path)

def main():
    print('Analyse Segmentation')
    # global_info = os.path.dirname(os.path.abspath(__file__)) + '/../global_information.json'
    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    categories = global_config["dataset"]["categories"]
    target_folder = global_config["general"]["target_folder"]
    categories_counter  = get_categories_counter(target_folder + '/gt_infos',categories)

    correct_detected_gt_objects = []

    correct_cats = {}
    false_cats = {}
    double_cats = {}
    for cat in categories:
        correct_cats[cat] = 0
        false_cats[cat] = 0
        double_cats[cat] = 0

    for name in tqdm(os.listdir(target_folder + '/bbox_overlap')):

        with open(target_folder + '/bbox_overlap/' + name,'r') as f:
            bbox_overlap = json.load(f)
    
        with open(target_folder + '/segmentation_infos/' + name,'r') as f:
            seg_info = json.load(f)

        with open(target_folder + '/gt_infos/' + seg_info['img'].split('.')[0] + '.json','r') as f:
            gt_info = json.load(f)

        if bbox_overlap['valid']:
            gt_cat = gt_info["objects"][bbox_overlap['index_gt_objects']]['category']
            predicted_cat = seg_info["predictions"]['category']

            gt_object = gt_info['img'] + 'object_' + str(bbox_overlap['index_gt_objects']).zfill(2)
            if bbox_overlap['correct_category'] and gt_object not in correct_detected_gt_objects:
                correct_cats[gt_cat] += 1
                correct_detected_gt_objects.append(gt_object)
            elif bbox_overlap['correct_category'] and gt_info['img'] in correct_detected_gt_objects:
                double_cats[gt_cat] += 1
            elif not bbox_overlap['correct_category']:
                false_cats[predicted_cat] += 1

    save_path = target_folder + '/global_stats/segmentation_analysis.png'
    plot_analysis(categories_counter,correct_cats,false_cats,double_cats,save_path)
    


           
        
        
            

if __name__ == '__main__':
    main()

