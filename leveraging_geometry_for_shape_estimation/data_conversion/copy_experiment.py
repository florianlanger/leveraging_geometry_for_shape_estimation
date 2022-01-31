import sys
import json
import os
import shutil


def flatten(t):
    return [item for sublist in t for item in sublist]

def main():
    exp_dir = '/scratch/fml35/experiments/leveraging_geometry_for_shape'

    # order is old_dir, new_dir, stage
    old_dir = exp_dir + '/' + sys.argv[1]
    new_dir = exp_dir + '/' + sys.argv[2]
    stage = sys.argv[3]

    map_stage_to_index = {'segmentation': 1,'lines': 2, 'retrieval': 3, 'keypoints': 4, 'R':5,'T':6,'eval':7}

    assert stage in map_stage_to_index
    assert os.path.exists(old_dir)

    os.mkdir(new_dir)

    stage_start = ['images','masks','gt_infos','models']
    stage_seg = ['segmentation_infos','segmentation_all_vis','segmentation_vis','cropped_and_masked','cropped_and_masked_small','segmentation_masks','bbox_overlap']
    stage_lines = ['lines_2d','lines_2d_vis','lines_2d_cropped','lines_2d_cropped_vis','lines_2d_filtered','lines_2d_filtered_vis']
    stage_retrieval = ['nn_infos','nn_vis','embedding']
    stage_keypoints = ['keypoints','keypoints_vis','matches','matches_orig_img_size','matches_vis','wc_matches','matches_quality','matches_quality_vis','wc_gt']
    stage_R = ['factors','factors_lines_vis','poses_R']
    stage_T = ['T_lines_vis','factors_T','poses_vis','poses']
    stage_eval = ['metrics','combined_vis','combined_vis_metrics_name','global_stats','global_stats/T_hists','selected_nn']

    all_stages = [stage_start,stage_seg,stage_lines,stage_retrieval,stage_keypoints,stage_R,stage_T,stage_eval]

    stage_index = map_stage_to_index[stage]
    folders_copy = flatten(all_stages[:stage_index])
    folders_create = flatten(all_stages[stage_index:])

    # make folders
    for folder in folders_copy:
        shutil.copytree(old_dir + '/' + folder,new_dir + '/' + folder)
    for folder in folders_create:
        os.mkdir(new_dir + '/' + folder)

    if stage_index >= 4:
        shutil.copy(old_dir + '/global_stats/retrieval_accuracy_all_embeds.json',new_dir + '/global_stats/retrieval_accuracy_all_embeds.json')


    # copy visualisation list
    shutil.copy(old_dir + '/global_stats/visualisation_images.json',new_dir + '/global_stats/visualisation_images.json')
    # load json and change name
    with open(old_dir + '/global_information.json','r') as f:
        global_config = json.load(f)
    global_config["general"]["target_folder"] = new_dir
    with open(new_dir + '/global_information.json','w') as f:
        json.dump(global_config,f,indent=4)

    shutil.copytree('/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation',new_dir + '/code')







if __name__ == '__main__':
    main()