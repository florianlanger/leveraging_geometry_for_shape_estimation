
# target_folder=/scratch/fml35/experiments/leveraging_geometry_for_shape/exp_083_s1_top_10_4_rotations
target_folder=/scratches/octopus/fml35/experiments/leveraging_geometry_for_shape/exp_107_top_10_s2_matches_select_nn_from_keypoints
# target_folder=/scratch/fml35/experiments/leveraging_geometry_for_shape/exp_084_s2_top_10_4_rotations
CODE_DIR=$target_folder/code
# CODE_DIR=leveraging_geometry_for_shape_estimation


# eval "bash leveraging_geometry_for_shape_estimation/data_conversion/data_conversion.sh ${target_folder} ${CODE_DIR}"
# # eval "bash ${CODE_DIR}/models_flat/models.sh ${target_folder} ${CODE_DIR}"
# eval "bash ${CODE_DIR}/segmentation/segmentation.sh ${target_folder} ${CODE_DIR}"
# eval "bash ${CODE_DIR}/line_detection/line_detection.sh ${target_folder} ${CODE_DIR}"
# eval "bash ${CODE_DIR}/retrieval/retrieval.sh ${target_folder} ${CODE_DIR}"
# eval "bash ${CODE_DIR}/keypoint_matching/keypoint_matching.sh ${target_folder} ${CODE_DIR}"
eval "bash ${CODE_DIR}/pose_and_shape_optimisation/pose_and_shape_optimisation.sh ${target_folder} ${CODE_DIR}"
eval "bash ${CODE_DIR}/combine_vis/combine_vis.sh ${target_folder} ${CODE_DIR}"
eval "bash ${CODE_DIR}/combine_results/combine_results.sh ${target_folder} ${CODE_DIR}"