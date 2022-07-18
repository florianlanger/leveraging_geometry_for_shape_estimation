
import os
from glob import glob

dir_1 = '/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/model'
dir_2 = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_117_scannet_models/models/extract_from_2d/exp_03/lines_3d_filtered_vis'

for file in glob(dir_1 + '/*/*/model_normalized.obj'):
    path_2 = file.replace(dir_1,dir_2).replace('/model_normalized.obj','.ply')
    assert os.path.exists(path_2),path_2

# for file in glob(dir_2 + '/*/*'):
#     assert os.path.exists(file.replace(dir_2,dir_1))