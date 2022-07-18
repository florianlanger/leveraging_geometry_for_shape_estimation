
import os
import shutil

path1 = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_165_roca_retrieval_no_z_gt_scale_fix_z_limits/T_lines_vis'
path2 = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_183_roca_retrieval_gt_z_lines/T_lines_vis'
path3 = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_165_roca_retrieval_no_z_gt_scale_fix_z_limits/T_lines_vis_exp_183'
for file in os.listdir(path1):
    assert os.path.exists(path3 + '/' + file) == False 
    shutil.copy(path2 + '/' + file,path3 + '/' + file)