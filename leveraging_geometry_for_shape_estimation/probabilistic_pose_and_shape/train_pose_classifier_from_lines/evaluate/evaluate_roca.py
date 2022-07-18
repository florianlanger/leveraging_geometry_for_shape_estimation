

import json
import os
import numpy as np
import quaternion

from leveraging_geometry_for_shape_estimation.eval.scannet.transform_from_single_frame_to_raw import transform_single_raw_invert_own_no_invert_roca
from leveraging_geometry_for_shape_estimation.eval.scannet.split_prediction import split
from leveraging_geometry_for_shape_estimation.eval.scannet.cluster_3d import cluster_3d
from leveraging_geometry_for_shape_estimation.eval.scannet.EvaluateBenchmark_v5 import evaluate,divide_potentially_by_0
from leveraging_geometry_for_shape_estimation.eval.scannet.scan2cad_constraint import scan2cad_constraint
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.evaluate.evaluate_single_frame import eval_single_frame_predictions
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.evaluate.combine_predictions import combine_predictions_with_predictions_all_images

from tqdm import tqdm


    

def eval_predictions(eval_path,n_scenes_vis=5,eval_all_images=False):
    roca_path = '/scratches/octopus_2/fml35/results/ROCA/per_frame_best_no_null.json'
    thresholds_dict = {"max_t_error":0.2,"max_r_error": 20,"max_s_error": 20}
    
    os.mkdir(eval_path + '/results_per_scene')
    os.mkdir(eval_path + '/results_per_scene_filtered')
    os.mkdir(eval_path + '/results_per_scene_scan2cad_constraints')
    os.mkdir(eval_path + '/results_per_scene_flags')
    os.mkdir(eval_path + '/results_per_scene_visualised')
    os.mkdir(eval_path + '/results_per_scene_filtered_visualised')
    os.mkdir(eval_path + '/results_per_scene_scan2cad_constraints_visualised')
    os.mkdir(eval_path + '/results_per_scene_single_frame_visualised')
    os.mkdir(eval_path + '/results_per_scene_visualised_combined')
    os.mkdir(eval_path + '/results_per_scene_filtered_visualised_combined')
    os.mkdir(eval_path + '/results_per_scene_scan2cad_constraints_visualised_combined')
    os.mkdir(eval_path + '/results_per_scene_single_frame_visualised_combined')

    # First need to have one run with eval_all_images True so that get predictions for all images. For next runs with gt annotation
    # then use the predictions from the previous run for objects that are not in the gt annotation.


    if eval_all_images == False:
        own_predictions_all_images_path = eval_path + '/../translation_pred_scale_pred_rotation_roca_init_retrieval_roca_all_images_True/combined_predictions_with_roca.json'
        single_frame_predictions = combine_predictions_with_predictions_all_images(eval_path + '/our_single_predictions.json',own_predictions_all_images_path)
        not_invert_previous_predictions=False

        # print('DEBUG: eval_all_images is False')
        # single_frame_predictions = combine_predictions_with_predictions_all_images(eval_path + '/our_single_predictions.json',roca_path)
        # not_invert_previous_predictions=True

        with open(eval_path + '/combined_predictions_with_roca.json','w') as json_file:
            json.dump(single_frame_predictions,json_file)

        eval_single_frame_predictions(eval_path,thresholds_dict)
        

    elif eval_all_images == True:
        single_frame_predictions = combine_predictions_with_predictions_all_images(eval_path + '/our_single_predictions.json',roca_path)

        # print('put back assertion')
        for img in single_frame_predictions:
            for i in range(len(single_frame_predictions[img])):
                assert single_frame_predictions[img][i]["own_prediction"] == True, single_frame_predictions[img]

        with open(eval_path + '/combined_predictions_with_roca.json','w') as json_file:
            json.dump(single_frame_predictions,json_file)
        not_invert_previous_predictions=True


    # transform single frame predictions to raw predictions
    dir_path_images = '/scratches/octopus_2/fml35/datasets/scannet/scannet_frames_25k/'
    scan2cad_full_annotations_path = dir_path_images + '../scan2cad_annotations/full_annotations.json'
    scan2cad_cad_appearances_path = dir_path_images + '../scan2cad_annotations/cad_appearances.json'
    with open(scan2cad_full_annotations_path) as json_file:
        scan2cad_full_annotations = json.load(json_file)
    output_path_raw = eval_path + '/raw_results.csv'


    transform_single_raw_invert_own_no_invert_roca(single_frame_predictions,scan2cad_full_annotations,output_path_raw,dir_path_images,not_invert_previous_predictions)

    # split raw predictions
    with open(output_path_raw,'r') as file:
        raw_results = file.readlines()  

    # print('DEBUG eval mask2cad ')
    # dir_path_images = '/scratches/octopus_2/fml35/datasets/scannet/scannet_frames_25k/'
    # scan2cad_full_annotations_path = dir_path_images + '../scan2cad_annotations/full_annotations.json'
    # scan2cad_cad_appearances_path = dir_path_images + '../scan2cad_annotations/cad_appearances.json'

    # with open('/scratch2/fml35/results/MASK2CAD/b5.csv','r') as file:
    #     raw_results = file.readlines()
    # with open('/scratch2/fml35/results/ROCA/raw_results.csv','r') as file:
    #     raw_results = file.readlines()

    dir_split = eval_path + '/results_per_scene/'
    split(raw_results,dir_split)

    # cluster 3d
    dir_filtered = eval_path + '/results_per_scene_filtered/'
    cluster_3d(dir_split,dir_filtered,scan2cad_full_annotations_path)

    # enforce scan2cad constraints
    dir_scan2cad_constraint = eval_path + '/results_per_scene_scan2cad_constraints/'
    scan2cad_constraint(dir_filtered,dir_scan2cad_constraint, scan2cad_cad_appearances_path,scan2cad_full_annotations_path)

    thresholds_dict = {"max_t_error":0.2,"max_r_error": 20,"max_s_error": 20}

    # evaluate
    add_ons = ["","_unseen_only"]
    unseen_onlys = [False,True]
    for only_evaluate_unseen,add_on in zip(unseen_onlys,add_ons):
        results_file_cats = eval_path + '/results_scannet_scan2cad_constraints{}.txt'.format(add_on)
        results_file_scenes = eval_path + '/results_scannet_scenes{}.json'.format(add_on)
        dir_evaluated_with_flags = eval_path + '/results_per_scene_flags' + add_on
        evaluate(dir_scan2cad_constraint,dir_evaluated_with_flags,results_file_cats,results_file_scenes,thresholds_dict,scan2cad_cad_appearances_path,scan2cad_full_annotations_path,use_tolerant_rotation=False,only_evaluate_unseen=only_evaluate_unseen)

    # visualize
    if n_scenes_vis > 0:
        visualise_predictions_3d(eval_path,single_frame_predictions,n_scenes_vis)

    
def visualise_predictions_3d(eval_path,single_frame_predictions,n_scenes_vis):

    from leveraging_geometry_for_shape_estimation.visualise_3d.visualise_predictions_from_csvs_v3 import visualise, visualise_single_images


    dir_gt_vis = '/scratch2/fml35/datasets/scannet/scenes_visualised_cameras_bbox'
    dir_roca_vis = '/scratch/fml35/experiments/regress_T/vis_roca/date_2022_06_01_time_16_42_51_visualise_all_roca/predictions/epoch_000000/scale_roca_rotation_lines_retrieval_gt/'

    shape_net_dir = "/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/model"
    with open(eval_path + '/results_scannet_scenes_without_retrieval.json','r') as f:
        scene_results = json.load(f)

    for add_on in ['','_filtered','_scan2cad_constraints']:
    # for add_on in ['_scan2cad_constraints']:
        projectdir = eval_path + '/results_per_scene' + add_on
        output_dir = projectdir + '_visualised'
        visualise(projectdir,output_dir,shape_net_dir,scene_results,single_frame_predictions,max_number_scenes=n_scenes_vis)
        dir_roca = dir_roca_vis + 'results_per_scene' + add_on + '_visualised'
        combine_ply_files(output_dir,dir_roca,dir_gt_vis,output_dir + '_combined',None)

    projectdir = eval_path + '/results_per_scene'
    output_dir = eval_path + '/results_per_scene_single_frame_visualised'
    visualise_single_images(projectdir,output_dir,shape_net_dir,single_frame_predictions,max_number_scenes=n_scenes_vis)
    dir_roca = dir_roca_vis + 'results_per_scene_single_frame_visualised'
    combine_ply_files(output_dir,dir_roca,dir_gt_vis,output_dir + '_combined',None,single_frame=True)


def combine_raw_ply_files(list_ply_files):
    all_colors = []
    all_vertices = []
    for ply_file in list_ply_files:
        colors = np.stack([ply_file['vertex']['red'],ply_file['vertex']['green'],ply_file['vertex']['blue']],axis=1)

        vertices = np.stack([ply_file['vertex']['x'],ply_file['vertex']['y'],ply_file['vertex']['z']],axis=1)
        all_colors.append(colors)
        all_vertices.append(vertices)
    all_colors = np.concatenate(all_colors,axis=0)
    all_vertices = np.concatenate(all_vertices,axis=0)

    return all_colors,all_vertices

def combine_ply_files(dir_own,dir_roca,dir_gt,out_dir,add_on,single_frame=False):

    from plyfile import PlyData
    from leveraging_geometry_for_shape_estimation.utilities_3d.utilities import writePlyFile
    roca_files = os.listdir(dir_roca)

    print('combine ply files')
    for file in tqdm(sorted(os.listdir(dir_own))):
        scene_name = file.split('_')[0] + '_' + file.split('_')[1].split('.')[0]

        path_own = os.path.join(dir_own, file)
        path_roca = os.path.join(dir_roca, find_roca_file(roca_files,scene_name))
        path_gt = os.path.join(dir_gt, scene_name + '.ply')
        if single_frame == True:
            path_gt = os.path.join(dir_gt, file.split('-')[0] + '.ply')

        assert os.path.exists(path_own), "File {} does not exist".format(path_own)
        assert os.path.exists(path_roca), "File {} does not exist".format(path_roca)
        assert os.path.exists(path_gt), "File {} does not exist".format(path_gt)

        out_path = os.path.join(out_dir, file)

        ply_own = PlyData.read(path_own)
        ply_roca = PlyData.read(path_roca)
        ply_gt = PlyData.read(path_gt)

        all_colors,all_vertices = combine_raw_ply_files([ply_own,ply_roca,ply_gt])
        writePlyFile(out_path, all_vertices, all_colors)

def find_roca_file(roca_files,scene_name):
    for file in roca_files:
        if scene_name in file:
            return file
    return None

if __name__ == '__main__':
    eval_path = '/scratch/fml35/experiments/regress_T/vis_roca/eval_roca'
    eval_predictions(eval_path,n_scenes_vis=1)