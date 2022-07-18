

# network_path=/scratch/fml35/experiments/regress_T/runs_16_classification_and_regression/date_2022_07_12_time_16_52_09_p_classifier_025_add_rgb/saved_models/network_epoch_000100.pth
# exp_name=date_2022_07_12_time_16_52_09_p_classifier_025_add_rgb_network_epoch_000100

# network_path=/scratch/fml35/experiments/regress_T/runs_16_classification_and_regression/date_2022_07_12_time_17_22_06_p_classifier_025_rotate_random_for_classifier/saved_models/network_epoch_000150.pth
# exp_name=date_2022_07_12_time_17_22_06_p_classifier_025_rotate_random_for_classifier_network_epoch_000150

base_path=/scratch/fml35/experiments/regress_T/runs_16_classification_and_regression
target_dir=/scratch/fml35/experiments/regress_T/evals_01

# experiments=(date_2022_07_16_time_20_07_02_p_classifier_025_add_rgb_add_rotation_1000_points_n_refinement_3_use_reprojected_as_query
# date_2022_07_16_time_19_17_12_p_classifier_025_add_rgb_add_rotation_1000_points_n_refinement_3_no_samples_bbox
# date_2022_07_16_time_19_07_30_p_classifier_025_add_rgb_add_rotation_100_points)

# networks=(network_epoch_000200 network_epoch_000150 network_epoch_000100)


experiments=(
# date_2022_07_14_time_16_53_00_p_classifier_025_add_rgb_add_rotation
# date_2022_07_14_time_17_04_21_p_classifier_025_add_rgb_add_rotation_1000_points
# date_2022_07_16_time_19_07_30_p_classifier_025_add_rgb_add_rotation_100_points
# date_2022_07_16_time_19_17_12_p_classifier_025_add_rgb_add_rotation_1000_points_n_refinement_3_no_samples_bbox
# date_2022_07_16_time_19_27_48_p_classifier_025_add_rgb_add_rotation_1000_points_n_refinement_3_no_depth
# date_2022_07_16_time_19_29_25_p_classifier_025_add_rgb_add_rotation_1000_points_n_refinement_3_no_normals
date_2022_07_16_time_19_31_53_p_classifier_025_add_rgb_add_rotation_1000_points_n_refinement_3_no_rgb
date_2022_07_16_time_19_37_16_p_classifier_025_add_rgb_add_rotation_1000_points_n_refinement_3_no_rst_cad_id
date_2022_07_16_time_19_43_59_p_classifier_025_add_rgb_add_rotation_1000_points_n_refinement_3_no_samples_bbox_15000_whole
date_2022_07_16_time_19_45_00_p_classifier_025_add_rgb_add_rotation_1000_points_n_refinement_3_1000_samples_bbox_15000_whole
date_2022_07_16_time_20_00_09_p_classifier_025_add_rgb_add_rotation_1000_points_n_refinement_3_vgg
)

networks=(
# network_epoch_000300
# network_epoch_000300
# network_epoch_000100
# network_epoch_000150
# network_epoch_000050
# network_epoch_000100
network_epoch_000100
network_epoch_000250
network_epoch_000100
network_epoch_000090
network_epoch_000110)



for i in 0 1 2 3 4
    do
        network_path=$base_path/${experiments[i]}/saved_models/${networks[i]}.pth
        exp_name=${experiments[i]}_${networks[i]}

        # echo $network_path
        # echo $exp_name


        out_dir=$target_dir/$exp_name
        echo $out_dir
        mkdir $out_dir

        refine=1
        eval_method=init_for_classification
        for rot_index in 0 1 2 3
            do
                python /home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/probabilistic_pose_and_shape/train_pose_classifier_from_lines/eval_v1.py $network_path $refine $rot_index $eval_method $out_dir
            done

        python /home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/probabilistic_pose_and_shape/train_pose_classifier_from_lines/evaluate/combine_classification_preds_v2.py $out_dir

        python /home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/probabilistic_pose_and_shape/train_pose_classifier_from_lines/eval_v1.py $network_path 3 0 init_from_best_rotation_index $out_dir
    done

