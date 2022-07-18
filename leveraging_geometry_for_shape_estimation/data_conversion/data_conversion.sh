# This causes whole script to exit if python script crashes
# set -e

# bash leveraging_geometry_for_shape_estimation/data_conversion/download_data_splits_and_superpoint.sh
# python leveraging_geometry_for_shape_estimation/data_conversion/download_models.py

python leveraging_geometry_for_shape_estimation/data_conversion/create_dirs.py $1
# python $2/data_conversion/reformat_pix3d_imgs.py $1
# python $2/data_conversion/create_visualisation_list.py $1
# python $2/data_conversion/get_gt_infos_multiple_correct.py $1

# python $1/leveraging_geometry_for_shape_estimation/data_conversion/create_dirs.py $1
# python $1/code/data_conversion/reformat_pix3d_imgs.py
# python $1/code/data_conversion/create_visualisation_list.py $1