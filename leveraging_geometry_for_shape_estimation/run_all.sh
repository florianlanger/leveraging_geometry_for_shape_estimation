
# target_folder=/scratch/fml35/experiments/leveraging_geometry_for_shape/exp_083_s1_top_10_4_rotations
# target_folder=/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_147_rova_plus_own_rotation
target_folder=/scratch/fml35/experiments/eval_classifier_grid/exp_007_gt_scale_gt_retrieval
# target_folder=/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_172_own_retrieval_roca_seg_lines_10_nn
# target_folder=/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_145_eval_gt_roca
# target_folder=/scratches/octopus/fml35/experiments/leveraging_geometry_for_shape/exp_106_top_10_s1_matches
# target_folder=/scratches/octopus_2/fml35/experiments/leveraging_geometry_for_shape/exp_117_debug_scannet
# target_folder=/scratch/fml35/experiments/leveraging_geometry_for_shape/exp_084_s2_top_10_4_rotations
# CODE_DIR=$target_folder/code/leveraging_geometry_for_shape_estimation
# CODE_DIR=/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_224_create_data
CODE_DIR=leveraging_geometry_for_shape_estimation

# NOW=date_`date '+%F_%H_%M_%S'`
# NOW=${NOW//-/_}

# eval "bash leveraging_geometry_for_shape_estimation/data_conversion/data_conversion.sh ${target_folder} ${CODE_DIR}"
# eval "bash leveraging_geometry_for_shape_estimation/data_conversion_scannet/data_conversion_scannet.sh ${target_folder} ${CODE_DIR}"
# eval "bash ${CODE_DIR}/models/models.sh ${target_folder} ${CODE_DIR}"
# eval "bash ${CODE_DIR}/segmentation/segmentation.sh ${target_folder} ${CODE_DIR}"
# eval "bash ${CODE_DIR}/line_detection/line_detection.sh ${target_folder} ${CODE_DIR}"
# eval "bash ${CODE_DIR}/retrieval/retrieval.sh ${target_folder} ${CODE_DIR}"
# eval "bash ${CODE_DIR}/keypoint_matching/keypoint_matching.sh ${target_folder} ${CODE_DIR}"
eval "bash ${CODE_DIR}/pose_and_shape_optimisation/pose_and_shape_optimisation.sh ${target_folder} ${CODE_DIR}"
# eval "bash ${CODE_DIR}/eval/scannet/eval_roca_metrics.sh ${target_folder} ${CODE_DIR}"

# eval "bash ${CODE_DIR}/vis_pose/vis_pose.sh ${target_folder} ${CODE_DIR}"
# eval "bash ${CODE_DIR}/combine_vis/combine_vis.sh ${target_folder} ${CODE_DIR}"

# eval "bash ${CODE_DIR}/eval/F1_score/compute_F1_score.sh ${target_folder} ${CODE_DIR}"

# eval "bash ${CODE_DIR}/combine_results/combine_results.sh ${target_folder} ${CODE_DIR}"