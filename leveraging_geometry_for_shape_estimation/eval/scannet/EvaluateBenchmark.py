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


def evaluate(projectdir,results_file,global_config,filename_cad_appearance, filename_annotations):


    # -> define Thresholds
    threshold_translation = global_config["evaluate_poses"]["scannet"]["max_t_error"] # <-- in meter
    threshold_rotation = global_config["evaluate_poses"]["scannet"]["max_r_error"] # <-- in deg
    threshold_scale = global_config["evaluate_poses"]["scannet"]["max_s_error"] # <-- in %
    force_same_retrieval = False
    # <-
    print('Threshold translation: ',threshold_translation)
    print('Threshold rotation: ',threshold_rotation)
    print('Threshold scale: ',threshold_scale)

    print('NO GT object center')

    text = ''


    for force_same_retrieval in [False,True]:
        print('Force same retrieval: ',force_same_retrieval)
        counter_objects_dont_exist = 0


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
        for file0 in glob.glob(projectdir + "/*.csv"):
            alignments = CSVHelper.read(file0)
            id_scan = os.path.basename(file0.rsplit(".", 1)[0])
            if id_scan not in testscenes_gt:
                print('id_scan',id_scan)
                print('continue')
                continue
            benchmark_per_scan[id_scan]["seen"] = 1

            # appearance_counter = {}
            appearance_counter = 0
            all_gt = []
            for alignment in alignments: # <- multiple alignments of same object in scene
            
                # -> read from .csv file
                catid_cad = str(alignment[0]).zfill(8)
                id_cad = alignment[1]
                cadkey = catid_cad + "_" + id_cad
                #import pdb; pdb.set_trace()

                # if cadkey in appearances_cad[id_scan]:
                #     n_appearances_allowed = appearances_cad[id_scan][cadkey] # maximum number of appearances allowed
                # else:
                #     n_appearances_allowed = 0
                # appearance_counter.setdefault(cadkey, 0)
                # if appearance_counter[cadkey] >= n_appearances_allowed:
                #     continue
                # appearance_counter[cadkey] += 1


                n_appearances_allowed = appearances_cad_total[id_scan]
                if appearance_counter >= n_appearances_allowed:
                    continue
                appearance_counter += 1


                if (catid_cad, id_cad) not in cad2info:
                    counter_objects_dont_exist += 1
                    appearance_counter -= 1
                    print(catid_cad,id_cad)
                    continue
                
                catname_cad = cad2info[(catid_cad, id_cad)]["catname"]
                sym = cad2info[(catid_cad, id_cad)]["sym"]
                t = np.asarray(alignment[2:5], dtype=np.float64)
                q0 = np.asarray(alignment[5:9], dtype=np.float64)
                q = np.quaternion(q0[0], q0[1], q0[2], q0[3])
                s = np.asarray(alignment[9:12], dtype=np.float64)
                # <-

                all_ts = []
                all_rs = []
                all_ss = []
                all_cats = []
                all_ids = [] 

                key = (id_scan, catid_cad) # <-- key to query the correct groundtruth models

                for idx, model_gt in enumerate(groundtruth[key]):
                    is_same_class = model_gt["catid_cad"] == catid_cad # <-- is always true (because the way the 'groundtruth' was created
                    if is_same_class: # <-- proceed only if candidate-model and gt-model are in same class


                        Mscan = SE3.compose_mat4(idscan2trs[id_scan]["translation"], idscan2trs[id_scan]["rotation"], idscan2trs[id_scan]["scale"])
                        Mcad = SE3.compose_mat4(model_gt["trs"]["translation"], model_gt["trs"]["rotation"], model_gt["trs"]["scale"]) #,-np.array(model_gt["center"]))


                        
                        t_gt, q_gt, s_gt = SE3.decompose_mat4(np.dot(np.linalg.inv(Mscan), Mcad))

            
                        single_dict = {}
                        single_dict["category"] = catname_cad
                        single_dict["id_cad"] = id_cad
                        single_dict["r"] = quaternion.as_rotation_matrix(q_gt).tolist()
                        single_dict["t"] = t_gt.tolist()
                        single_dict["s"] = s_gt.tolist()

                        if single_dict not in all_gt:

                            all_gt.append(single_dict)
                        # all_ts.append(t_gt.tolist())
                        # all_rs.append(quaternion.as_rotation_matrix(q_gt).tolist())
                        # all_ss.append(s_gt.tolist())
                        # all_cats.append(catname_cad)
                        # all_ids.append(id_cad)




                        error_translation = np.linalg.norm(t - t_gt, ord=2)
                        error_scale = 100.0*np.abs(np.mean(s/s_gt) - 1)
                        # error_scale = 100.0*np.mean(np.abs(s/s_gt - 1))
                        # print('changed error in scale')

                        # --> resolve symmetry
                        if sym == "__SYM_ROTATE_UP_2":
                            m = 2
                            tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
                            error_rotation = np.min(tmp)
                        elif sym == "__SYM_ROTATE_UP_4":
                            m = 4
                            tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
                            error_rotation = np.min(tmp)
                        elif sym == "__SYM_ROTATE_UP_INF":
                            m = 36
                            tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
                            error_rotation = np.min(tmp)
                        else:
                            error_rotation = calc_rotation_diff(q, q_gt)

                        if force_same_retrieval:
                            retrieval_correct = id_cad == model_gt['id_cad']
                        else:
                            retrieval_correct = True

                        is_valid_transformation = error_translation <= threshold_translation and error_rotation <= threshold_rotation and error_scale <= threshold_scale and retrieval_correct

                        counter += 1
                        if is_valid_transformation:
                            benchmark_per_scan[id_scan]["n_good"] += 1
                            benchmark_per_class[catname_cad]["n_good"] += 1
                            del groundtruth[key][idx]
                            break
                            # print('HAve deactivated breaking')

                # pred_single = {'t':t.tolist(),'r':quaternion.as_rotation_matrix(q).tolist(),'s':s.tolist(),'category':catname_cad,'id_cad': id_cad}
                # with open('/scratch2/fml35/results/ROCA/debug_evaluation/single_pred_{}_{}.json'.format(id_scan,catname_cad),'w') as f:
                #     json.dump(pred_single,f)

                # gt = {'t':all_ts,'r':all_rs,'s':all_ss,'category':all_cats,'id_cad':all_ids}
                # with open('/scratch2/fml35/results/ROCA/gt_per_scene/{}.json'.format(id_scan,catname_cad),'w') as f:
                #     json.dump(all_gt,f)


                

        # print("***********")
        benchmark_per_scan = sorted(benchmark_per_scan.items(), key=lambda x: x[1]["n_good"], reverse=True)
        total_accuracy = {"n_good" : 0, "n_total" : 0, "n_scans" : 0}
        for k, v in benchmark_per_scan:
            if "seen" in v:
                total_accuracy["n_good"] += v["n_good"]
                total_accuracy["n_total"] += v["n_total"]
                total_accuracy["n_scans"] += 1
                # print("id-scan: {:>20s} \t n-cads-positive: {:>4d} \t n-cads-total: {:>4d} \t accuracy: {:>4.4f}".format(k,  v["n_good"], v["n_total"], float(v["n_good"])/v["n_total"]))
        instance_mean_accuracy = float(total_accuracy["n_good"])/total_accuracy["n_total"]
        # print("instance-mean-accuracy: {:>4.4f} \t n-cads-positive: {:>4d} \t n-cads-total: {:>4d} \t n-total-scans: {:>4d}".format(instance_mean_accuracy, total_accuracy["n_good"], total_accuracy["n_total"], total_accuracy["n_scans"]))

        if force_same_retrieval:
            add_on = 'WITH RETRIEVAL'
        else:
            add_on = 'NO RETRIEVAL'
        text += "*********** PER CLASS {} **************************\n".format(add_on)

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

        print('objects dont exist ',counter_objects_dont_exist)

    with open(results_file, 'w') as f:
        f.write(text)

    print(text)
    # print(total_accuracy["n_good"],total_accuracy["n_total"])
    return instance_mean_accuracy, class_mean_accuracy


if __name__ == "__main__":

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    scan2cad_dir = global_config["dataset"]["dir_path_images"] + '../scan2cad_annotations/'
    projectdir = global_config["general"]["target_folder"] + '/global_stats/eval_scannet/results_per_scene_filtered/'
    results_file = global_config["general"]["target_folder"] + '/global_stats/results_scannet.txt'

    # projectdir = '/scratch2/fml35/results/MASK2CAD/results_per_scene/'
    # results_file = '/scratch2/fml35/results/MASK2CAD/results_per_scene_no_filtering_correct_scale_error.txt'

    evaluate(projectdir,results_file,global_config, scan2cad_dir +  "cad_appearances.json", scan2cad_dir + "full_annotations.json")