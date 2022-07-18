# python pose_and_shape_optimisation/main_shape.py $1

# python $2/pose_and_shape_optimisation/main_pose.py $1
# python $2/probabilistic_pose_and_shape/pose.py $1
# python $2/pose_and_shape_optimisation/pose_roca.py $1
# python $2/probabilistic_pose_and_shape/rotation_from_lines.py $1
# python $2/probabilistic_pose_and_shape/select_rotation.py $1
# python $2/probabilistic_pose_and_shape/translation_from_lines_no_shape.py $1
python $2/probabilistic_pose_and_shape/translation_from_lines_v9.py $1
python $2/probabilistic_pose_and_shape/translation_from_lines_visualise_v2.py $1
# python $2/probabilistic_pose_and_shape/translation_from_lines.py $1
# python $2/probabilistic_pose_and_shape/main_translation.py $1
python $2/pose_and_shape_optimisation/select_best_v2.py $1

# Note versions with iterating over all 4 rotations
# python $2/probabilistic_pose_and_shape/translation_from_lines_v6.py $1
# python $2/probabilistic_pose_and_shape/translation_from_lines_visualise_v1.py $1
# python $2/pose_and_shape_optimisation/select_best_v2.py $1