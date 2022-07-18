target_folder=/scratches/octopus/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/3d_future/data_01
line_exp=exp_01_filter_5
n_lines_filter=5


# python line_detection_models.py $target_folder $line_exp
python extract_3d_lines.py $target_folder $line_exp
python combine_3d_lines.py $target_folder $line_exp
python filter_3d_lines.py $target_folder $line_exp $n_lines_filter
# python reformat_lines.py $target_folder $line_exp