# python pose_and_shape_optimisation/main_shape.py $1

python $2/pose_and_shape_optimisation/main_pose.py $1
# python $2/probabilistic_pose_and_shape/pose.py $1
# python $2/probabilistic_pose_and_shape/rotation_from_lines.py $1
# python $2/probabilistic_pose_and_shape/translation_from_lines.py $1
# python $2/probabilistic_pose_and_shape/main_translation.py $1
python $2/pose_and_shape_optimisation/select_best.py $1
python $2/pose_and_shape_optimisation/compute_metrics.py $1
python $2/pose_and_shape_optimisation/analyse_R_and_T.py $1
python $2/pose_and_shape_optimisation/aps_new.py $1
python $2/pose_and_shape_optimisation/vis_pose.py $1