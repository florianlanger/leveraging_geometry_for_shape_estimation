import numpy as np
np.warnings.filterwarnings('ignore')
import pathlib
import subprocess
import os
import collections
import shutil
import quaternion
import operator
import glob
import csv
import re
import CSVHelper
import SE3
import JSONHelper
import argparse
np.seterr(all='raise')
import argparse
import json
import sys



# get top8 (most frequent) classes from annotations. 
def get_top8_classes_scannet():                                                                                                                                                                                                                                                                                           
    top = collections.defaultdict(lambda : "other")
    top["03211117"] = "display"
    top["04379243"] = "table"
    top["02808440"] = "bathtub"
    top["02747177"] = "bin"
    top["04256520"] = "sofa"
    top["03001627"] = "chair"
    top["02933112"] = "cabinet"
    top["02871439"] = "bookshelf"
    top["02818832"] = "bed"
    return top


# helper function to calculate difference between two quaternions 
def calc_rotation_diff(q, q00):
    rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
    rotation_dot_abs = np.abs(rotation_dot)
    try:                                                                                                                                                                                                                                                                                                                      
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:
        return 0.0
    error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    error_rotation = np.rad2deg(error_rotation_rad)
    return error_rotation


def calc_rotation_diff_considering_symmetry(q,q_gt,sym_pred,sym_gt):

    if sym_gt != sym_pred:
        error_rotation = calc_rotation_diff(q, q_gt)

    else:
        if sym_gt == "__SYM_ROTATE_UP_2":
            m = 2
            tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
            error_rotation = np.min(tmp)
        elif sym_gt == "__SYM_ROTATE_UP_4":
            m = 4
            tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
            error_rotation = np.min(tmp)
        elif sym_gt == "__SYM_ROTATE_UP_INF":
            m = 36
            tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
            error_rotation = np.min(tmp)
        else:
            error_rotation = calc_rotation_diff(q, q_gt)

    return error_rotation


def reformat_results_per_scene(benchmark_per_scan):
    score_per_scan = {}
    for item in benchmark_per_scan:
        scene_dict = dict(item[1])
        scene_dict.pop('seen', None)
        if scene_dict['n_total'] != 0:
            scene_dict['accuracy'] = float(scene_dict['n_good']/scene_dict['n_total'])
        else:
            assert scene_dict['n_good'] == 0
            scene_dict['accuracy'] = 0.
        
        score_per_scan[item[0]] = scene_dict
    return score_per_scan


def evaluate(projectdir,outputdir,results_file_cats,results_file_scenes,global_config,filename_cad_appearance, filename_annotations):


    # -> define Thresholds
    threshold_translation = global_config["evaluate_poses"]["scannet"]["max_t_error"] # <-- in meter
    threshold_rotation = global_config["evaluate_poses"]["scannet"]["max_r_error"] # <-- in deg
    threshold_scale = global_config["evaluate_poses"]["scannet"]["max_s_error"] # <-- in %
    force_same_retrieval = False
    # <-
    print('Should be sym of gt model ?')
    print('Threshold translation: ',threshold_translation)
    print('Threshold rotation: ',threshold_rotation)
    print('Threshold scale: ',threshold_scale)

    print('NO GT object center')

    text = ''


    for force_same_retrieval in [False,True]:
        print('Force same retrieval: ',force_same_retrieval)
        if force_same_retrieval:
            add_on = 'with_retrieval'
        else:
            add_on = 'without_retrieval'


        appearances_cad = JSONHelper.read(filename_cad_appearance)

        appearances_cad_total = {}
        for scene in appearances_cad:
            appearances_cad_total[scene] = 0
            for model in appearances_cad[scene]:
                appearances_cad_total[scene] += appearances_cad[scene][model]

        benchmark_per_scan = collections.defaultdict(lambda : collections.defaultdict(lambda : 0)) # <-- benchmark_per_scan
        benchmark_per_class = collections.defaultdict(lambda : collections.defaultdict(lambda : 0)) # <-- benchmark_per_class

        catid2catname = get_top8_classes_scannet()
        
        groundtruth = {}
        cad2info = {}
        idscan2trs = {}
        
        testscenes = [os.path.basename(f).split(".")[0] for f in glob.glob(projectdir + "/*.csv")]
        
        testscenes_gt = []
        for r in JSONHelper.read(filename_annotations):
            id_scan = r["id_scan"]

            # NEED THIS
            if id_scan not in testscenes:
                continue

            testscenes_gt.append(id_scan)

            idscan2trs[id_scan] = r["trs"]
            
            for model in r["aligned_models"]:
                id_cad = model["id_cad"]
                catid_cad = model["catid_cad"]
                catname_cad = catid2catname[catid_cad]
                # if catname_cad == 'other':
                #     print(model["catid_cad"])
                #     print(id_cad)
                model["n_total"] = len(r["aligned_models"])
                groundtruth.setdefault((id_scan, catid_cad),[]).append(model)
                cad2info[(catid_cad, id_cad)] = {"sym" : model["sym"], "catname" : catname_cad}
                if catname_cad != 'other':
                    benchmark_per_class[catname_cad]["n_total"] += 1
                    benchmark_per_scan[id_scan]["n_total"] += 1

        projectname = os.path.basename(os.path.normpath(projectdir))


        # Iterate through your alignments
        counter = 0
        counter_objects_dont_exist = 0
        for file0 in glob.glob(projectdir + "/*.csv"):

            # if "scene0011_00" not in file0:
            #     continue

            

            alignments_compare = CSVHelper.read(file0.replace('/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_156_roca_all_vis','/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_186_roca_retrieval_gt_z_lines_octopus'))
            assert os.path.exists(file0.replace('/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_156_roca_all_vis','/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_186_roca_retrieval_gt_z_lines_octopus'))
            detections_compare = [alignment[13] for alignment in alignments_compare]
            # print(detections_compare)
            # print(DF)
            alignments = CSVHelper.read(file0)
            id_scan = os.path.basename(file0.rsplit(".", 1)[0])
            if id_scan not in testscenes_gt:
                print('id_scan',id_scan)
                print('continue')
                continue
            benchmark_per_scan[id_scan]["seen"] = 1

            all_gt = []

            appended_alignments = []

            for alignment in alignments: # <- multiple alignments of same object in scene
                counter += 1
            
                # -> read from .csv file
                catid_cad = str(alignment[0]).zfill(8)
                id_cad = alignment[1]
                cadkey = catid_cad + "_" + id_cad

                # print(alignment[13])
                # print(detections_compare)
                # print(Df)
                if alignment[13] not in detections_compare:
                    print(dfdfd)
                    continue

                # if (catid_cad, id_cad) not in cad2info:
                #     counter_objects_dont_exist += 1
                #     continue

                
                catname_cad = cad2info[(catid_cad, id_cad)]["catname"]
                sym_pred = cad2info[(catid_cad, id_cad)]["sym"]
                t = np.asarray(alignment[2:5], dtype=np.float64)
                q0 = np.asarray(alignment[5:9], dtype=np.float64)
                q = np.quaternion(q0[0], q0[1], q0[2], q0[3])
                s = np.asarray(alignment[9:12], dtype=np.float64)
                # <-

                key = (id_scan, catid_cad) # <-- key to query the correct groundtruth models


                infos_compared_to_gt = []
                had_detection = False

                for idx, model_gt in enumerate(groundtruth[key]):
                    is_same_class = model_gt["catid_cad"] == catid_cad # <-- is always true (because the way the 'groundtruth' was created
                    if is_same_class: # <-- proceed only if candidate-model and gt-model are in same class


                        Mscan = SE3.compose_mat4(idscan2trs[id_scan]["translation"], idscan2trs[id_scan]["rotation"], idscan2trs[id_scan]["scale"])
                        Mcad = SE3.compose_mat4(model_gt["trs"]["translation"], model_gt["trs"]["rotation"], model_gt["trs"]["scale"]) #,-np.array(model_gt["center"]))

                        sym_gt = cad2info[(model_gt["catid_cad"], model_gt['id_cad'])]["sym"]
                        
                        t_gt, q_gt, s_gt = SE3.decompose_mat4(np.dot(np.linalg.inv(Mscan), Mcad))

            
                        single_dict = {}
                        single_dict["category"] = catname_cad
                        single_dict["id_cad"] = id_cad
                        single_dict["r"] = quaternion.as_rotation_matrix(q_gt).tolist()
                        single_dict["t"] = t_gt.tolist()
                        single_dict["s"] = s_gt.tolist()

                        if single_dict not in all_gt:
                            all_gt.append(single_dict)



                        error_translation = np.linalg.norm(t - t_gt, ord=2)
                        # error_scale = 100.0*np.abs(np.mean(s/s_gt) - 1)
                        error_scale = 100.0*np.abs(np.mean(s/s_gt) - 1)
                        # error_scale = 100.0*np.abs(np.mean(s_gt/s) - 1)
                        # error_scale = 100.0*np.mean(np.abs(s/s_gt - 1))
                        # print('changed error in scale')

                        # --> resolve symmetry
                        # error_rotation = calc_rotation_diff_considering_symmetry(q_gt,q,sym)
                        error_rotation = calc_rotation_diff_considering_symmetry(q,q_gt,sym_pred,sym_gt)


                        trans_correct = error_translation <= threshold_translation
                        rotation_correct = error_rotation <= threshold_rotation
                        scale_correct = error_scale <= threshold_scale
                        retrieval_correct = id_cad == model_gt['id_cad']

                        if force_same_retrieval:
                            is_valid_transformation = trans_correct and rotation_correct and scale_correct and retrieval_correct
                        else:
                            is_valid_transformation = trans_correct and rotation_correct and scale_correct

                        # print([is_valid_transformation,trans_correct,rotation_correct,scale_correct,retrieval_correct,error_translation,error_rotation,error_scale])
                        infos_compared_to_gt.append([is_valid_transformation,trans_correct,rotation_correct,scale_correct,retrieval_correct,error_translation,error_rotation,error_scale])
                        
                        
                        if is_valid_transformation:

                            had_detection = True
                            index_to_gt = idx

                            # if catname_cad == 'bookshelf' and retrieval_correct == False:
                            #     print('counter',idx)
                            #     print(alignment[-1])
                            #     print(error_rotation)
                            #     print(error_translation)
                            #     print([is_valid_transformation,trans_correct,rotation_correct,scale_correct,retrieval_correct,error_translation,error_rotation,error_scale])
                        

                            benchmark_per_scan[id_scan]["n_good"] += 1
                            benchmark_per_class[catname_cad]["n_good"] += 1
                            del groundtruth[key][idx]
                            break
                            # print('HAve deactivated breaking')

                if not had_detection:
                    translation_errors = [item[5] for item in infos_compared_to_gt]
                    min_translation_error = min(translation_errors)
                    index_to_gt = translation_errors.index(min_translation_error)
                

                # if alignment[-1] in ['scene0208_00-002000_00','scene0203_02-000900_00','scene0231_02-002800_01','scene0591_02-000200_00','scene0535_00-000000_01']:
                #     print('min_index',index_to_gt)
                #     print('===============')

                appended_alignments.append(tuple(alignment + infos_compared_to_gt[index_to_gt]))


                

                
            # CSVHelper.write(outputdir + '_' + add_on + '/' + id_scan + '.csv',appended_alignments)
            # CSVHelper.write(outputdir + '_' + add_on + '_debug/' + id_scan + '.csv',appended_alignments)
                

        # print("***********")
        benchmark_per_scan = sorted(benchmark_per_scan.items(), key=lambda x: x[1]["n_good"], reverse=True)
        total_accuracy = {"n_good" : 0, "n_total" : 0, "n_scans" : 0}
        for k, v in benchmark_per_scan:
            if "seen" in v:
                total_accuracy["n_good"] += v["n_good"]
                total_accuracy["n_total"] += v["n_total"]
                total_accuracy["n_scans"] += 1

    

        score_per_scan = reformat_results_per_scene(benchmark_per_scan)
            
        with open(results_file_scenes.replace('.json','_{}_only11987.json'.format(add_on)), 'w') as f:
            json.dump(score_per_scan,f)


        
        text += "*********** PER CLASS {} **************************\n".format(add_on)

        instance_mean_accuracy = float(total_accuracy["n_good"])/total_accuracy["n_total"]

        accuracy_per_class = {}
        for k,v in benchmark_per_class.items():
            accuracy_per_class[k] = float(v["n_good"])/v["n_total"]
            text += "category-name: {:>20s} \t n-cads-positive: {:>4d} \t n-cads-total: {:>4d} \t accuracy: {:>4.4f}\n".format(k,  v["n_good"], v["n_total"], float(v["n_good"])/v["n_total"])
            

        class_mean_accuracy = np.mean([ v for k,v in accuracy_per_class.items() if k != 'other'])
        text += "class-mean-accuracy: {:>4.4f}\n".format(class_mean_accuracy)

        cats = ["bathtub","bed","bin","bookshelf","cabinet","chair","display","sofa","table"]
        text += str(cats+['class_mean','instance_mean']) + '\n'
        values =''
        for cat in cats:
            v = benchmark_per_class[cat]
            value = float(v["n_good"])/v["n_total"]
            values += str(np.round(100*value,3)).replace('.',',') + ';'
        
        values += str(np.round(100*class_mean_accuracy,3)).replace('.',',') + ';'
        values += str(np.round(100*instance_mean_accuracy,3)).replace('.',',') + ';'
        text += values + '\n'

    with open(results_file_cats, 'w') as f:
        f.write(text)

    print(text)
    print('Number objects dont exist ',counter_objects_dont_exist)
    print('Counter',counter)
    # print(total_accuracy["n_good"],total_accuracy["n_total"])
    return instance_mean_accuracy, class_mean_accuracy


if __name__ == "__main__":

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    scan2cad_dir = global_config["dataset"]["dir_path_images"] + '../scan2cad_annotations/'
    projectdir = global_config["general"]["target_folder"] + '/global_stats/eval_scannet/results_per_scene_scan2cad_constraints/'
    outputdir = global_config["general"]["target_folder"] + '/global_stats/eval_scannet/results_per_scene_flags'
    results_file_cats = global_config["general"]["target_folder"] + '/global_stats/results_scannet_scan2cad_constraints_v1_only11987.txt'
    results_file_scenes = global_config["general"]["target_folder"] + '/global_stats/results_scannet_scenes.json'

    # projectdir = '/scratch2/fml35/results/MASK2CAD/results_per_scene/'
    # results_file = '/scratch2/fml35/results/MASK2CAD/results_per_scene_no_filtering_correct_scale_error.txt'

    evaluate(projectdir,outputdir,results_file_cats,results_file_scenes,global_config, scan2cad_dir +  "cad_appearances.json", scan2cad_dir + "full_annotations.json")