

for refine in 1
do
#    for path in /scratches/octopus/fml35/experiments/regress_T/runs_10_T_and_R_and_S/date_2022_06_16_time_17_46_25_five_refinements_depth_aug_points_in_bbox_3d_coords/saved_models/network_epoch_000150.pth
    # for path in /scratches/octopus/fml35/experiments/regress_T/runs_15_classification_big/date_2022_07_01_time_18_09_18_classification_adjusted_p_smaller_azim/saved_models/network_epoch_000100.pth
    # for path in /scratch/fml35/experiments/regress_T/runs_16_classification_and_regression/date_2022_07_03_time_12_15_58_classification_adjusted_p_smaller_azim/saved_models/network_epoch_000200.pth
    # for path in /scratch/fml35/experiments/regress_T/runs_16_classification_and_regression/date_2022_07_05_time_11_57_34_3_refinements_half_classifier_half_regressor_4_rotations/saved_models/network_epoch_000200.pth
    # for path in /scratch/fml35/experiments/regress_T/runs_16_classification_and_regression/date_2022_07_05_time_11_57_34_3_refinements_half_classifier_half_regressor_4_rotations/saved_models/network_epoch_000350.pth
    # for path in /scratch/fml35/experiments/regress_T/runs_18_T_and_R_and_S/date_2022_07_07_time_12_27_05_3_refinements_random_points_2d_random_points_3d_no_reprojected_as_query_no_cc/saved_models/network_epoch_000250.pth
    for path in /scratch/fml35/experiments/regress_T/runs_18_T_and_R_and_S/date_2022_07_05_time_17_41_58_3_refinements_no_cc/saved_models/network_epoch_000200.pth
    do
        for rot_index in 0 1 2 3
        do
            python /home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/probabilistic_pose_and_shape/train_pose_classifier_from_lines/main_v2.py $path $refine $rot_index $eval_method
        done
    done
done

