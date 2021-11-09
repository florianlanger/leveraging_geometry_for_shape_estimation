# This causes whole script to exit if sub script exits
set -e

target_folder=/data/cornucopia/fml35/experiments/exp_023_debug

# bash /home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/models/models.sh

# bash /home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/data_conversion/data_conversion.sh $target_folder
bash $target_folder/code/segmentation/segmentation.sh $target_folder
bash $target_folder/code/retrieval/retrieval.sh $target_folder
bash $target_folder/code/keypoint_matching/keypoint_matching.sh $target_folder
bash $target_folder/code/pose_and_shape_optimisation/pose_and_shape_optimisation.sh $target_folder