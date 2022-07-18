
import os
from glob import glob
import numpy as np
from tqdm import tqdm

dir_1 = '/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/model'
dir_2 = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_117_scannet_models/models/extract_from_2d/exp_03/lines_3d_filtered'

counter = 0
files = glob(dir_1 + '/*/*/model_normalized.obj')
names = []

for file in tqdm(files):
    path_2 = file.replace(dir_1,dir_2).replace('/model_normalized.obj','.npy')
    lines = np.load(path_2)
    if lines.shape[0] == 0:
        names.append(path_2.replace(dir_2,''))
        counter += 1 

print(names)
print('{} objects have 0 lines out of {}'.format(counter,len(files)))

# for file in glob(dir_2 + '/*/*'):
#     assert os.path.exists(file.replace(dir_2,dir_1))