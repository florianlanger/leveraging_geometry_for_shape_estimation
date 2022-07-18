from operator import inv
from unicodedata import category
import numpy as np
import quaternion
import json

from leveraging_geometry_for_shape_estimation.eval.scannet.EvaluateBenchmark_v5 import calc_rotation_diff_considering_symmetry,divide_potentially_by_0
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.evaluate.combine_predictions import combine_predictions_with_predictions_all_images


def valid_single_detection(consider_which,t,s,q,model_id,t_gt,s_gt,q_gt,model_id_gt,thresholds_dict,cadid2sym):

    threshold_translation = thresholds_dict["max_t_error"] # <-- in meter
    threshold_rotation = thresholds_dict["max_r_error"] # <-- in deg
    threshold_scale = thresholds_dict["max_s_error"] # <-- in %

    sym_pred = cadid2sym[model_id]
    sym_gt = cadid2sym[model_id_gt]

    error_translation = np.linalg.norm(np.array(t) - t_gt, ord=2)
    # error_scale = 100.0*np.abs(np.mean(np.array(s)/s_gt) - 1)
    error_scale = 100.0*np.mean(np.abs(np.array(s)/s_gt - 1))

    q = np.quaternion(q[0], q[1], q[2], q[3])
    q_gt = np.quaternion(q_gt[0], q_gt[1], q_gt[2], q_gt[3])
    error_rotation = calc_rotation_diff_considering_symmetry(q,q_gt,sym_pred,sym_gt)


    trans_correct = error_translation <= threshold_translation
    rotation_correct = error_rotation <= threshold_rotation
    scale_correct = error_scale <= threshold_scale
    retrieval_correct = model_id == model_id_gt

    if 'r' not in consider_which:
        rotation_correct = True
    if 's' not in consider_which:
        scale_correct = True
    if 't' not in consider_which:
        trans_correct = True
    if 'retrieval' not in consider_which:
        retrieval_correct = True

    is_valid_transformation = trans_correct and rotation_correct and scale_correct and retrieval_correct

    return is_valid_transformation


def compute_category_counts(single_frame_predictions_with_gt,consider_which,thresholds_dict,cadid2sym):

    cats = ["display","table","bathtub","bin","sofa","chair","cabinet","bookshelf","bed","any"]
    counter_cats = {}
    for cat in cats:
        counter_cats[cat] = {"correct":0,"total":0}

    for img in single_frame_predictions_with_gt:
        for i in range(len(single_frame_predictions_with_gt[img])):
            if single_frame_predictions_with_gt[img][i]["own_prediction"] == True:

            # also need this for eval roca
            # if single_frame_predictions_with_gt[img][i]["associated_gt_infos"]["matched_to_gt_object"]:
                
                t_pred = single_frame_predictions_with_gt[img][i]["t"]
                s_pred = single_frame_predictions_with_gt[img][i]["s"]
                q_pred = single_frame_predictions_with_gt[img][i]["q"]
                model_id_pred = single_frame_predictions_with_gt[img][i]["scene_cad_id"][1]
                t_gt = single_frame_predictions_with_gt[img][i]["associated_gt_infos"]["trans_mat"]
                s_gt = single_frame_predictions_with_gt[img][i]["associated_gt_infos"]["scaling"]
                q_gt = quaternion.from_rotation_matrix(single_frame_predictions_with_gt[img][i]["associated_gt_infos"]["rot_mat"])
                q_gt = quaternion.as_float_array(q_gt)

                # needed this for evaluating roca predictions

                # invert = np.array([[-1,0,0],[0,-1,0],[0,0,1.]])
                # t_gt = np.array(t_gt)
                # t_gt = np.matmul(invert,t_gt)

                # r_gt = np.array(single_frame_predictions_with_gt[img][i]["associated_gt_infos"]["rot_mat"])
                # r_gt = np.matmul(invert,r_gt)
                # q_gt = quaternion.from_rotation_matrix(r_gt)
                # q_gt = quaternion.as_float_array(q_gt)




                model_id_gt = single_frame_predictions_with_gt[img][i]["associated_gt_infos"]["model"].split('/')[2]
                is_valid = valid_single_detection(consider_which,t_pred,s_pred,q_pred,model_id_pred,t_gt,s_gt,q_gt,model_id_gt,thresholds_dict,cadid2sym)

                category = single_frame_predictions_with_gt[img][i]["category"]
                assert category == single_frame_predictions_with_gt[img][i]["associated_gt_infos"]["category"]
                counter_cats[category]['total'] += 1
                counter_cats[category]['correct'] += is_valid * 1
                counter_cats['any']['total'] += 1
                counter_cats['any']['correct'] += is_valid * 1
    return counter_cats
        
def get_output_text(category_counts):

    text = "*********** PER CLASS {} **************************\n"


    instance_mean_accuracy = float(category_counts['any']['correct'])/category_counts['any']['total']

    accuracy_per_class = {}
    for k,v in category_counts.items():
        # if k != 'any':
        accuracy_per_class[k] = divide_potentially_by_0(v["correct"],v["total"])
        text += "category-name: {:>20s} \t n-cads-positive: {:>4d} \t n-cads-total: {:>4d} \t accuracy: {:>4.4f}\n".format(k,  v["correct"], v["total"], accuracy_per_class[k])
            

    class_mean_accuracy = np.mean([ v for k,v in accuracy_per_class.items() if k != 'any'])
    text += "class-mean-accuracy: {:>4.4f}\n".format(class_mean_accuracy)

    cats = ["bathtub","bed","bin","bookshelf","cabinet","chair","display","sofa","table"]
    text += str(cats+['class_mean','instance_mean']) + '\n'
    values =''
    for cat in cats:
        v = category_counts[cat]
        value = divide_potentially_by_0(v["correct"],v["total"])
        values += str(np.round(100*value,3)).replace('.',',') + ';'
    
    values += str(np.round(100*class_mean_accuracy,3)).replace('.',',') + ';'
    values += str(np.round(100*instance_mean_accuracy,3)).replace('.',',') + ';'
    text += values + '\n'
    return text


def get_sym_by_cadid():

    cadid2sym = {}
    with open('/scratches/octopus_2/fml35/datasets/scannet/scan2cad_annotations/full_annotations.json','r') as f:
        annotations = json.load(f)

    for r in annotations:
        for model in r["aligned_models"]:
            id_cad = model["id_cad"]
            cadid2sym[id_cad] = model["sym"]
    return cadid2sym


def eval_single_frame_predictions(eval_path,thresholds_dict):
    roca_path_with_gt = '/scratches/octopus/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/all_detection_infos_fixed_category.json'
    single_frame_predictions_with_gt = combine_predictions_with_predictions_all_images(eval_path + '/our_single_predictions.json',roca_path_with_gt,ending_file='.json')
    with open(eval_path + '/combined_predictions_with_roca_and_gt_infos.json','w') as json_file:
        json.dump(single_frame_predictions_with_gt,json_file)

    
    cadid2sym = get_sym_by_cadid()


    for consider_which in [['r'],['t'],['s'],['retrieval'],['r','t','s'],['r','t','s','retrieval']]:
        category_counts = compute_category_counts(single_frame_predictions_with_gt,consider_which,thresholds_dict,cadid2sym)
        out_text = get_output_text(category_counts)

        add_on = '_'.join(consider_which)
        with open(eval_path + '/results_single_frame_{}.txt'.format(add_on),'w') as f:
            f.write(out_text)