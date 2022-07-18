
from glob import glob
import os
from tqdm import tqdm

for path in tqdm(glob('/scratch/fml35/datasets/3d_future/3D-FUTURE-model_reformatted/model/*/*/normalized_model.obj')):
    os.rename(path, path.replace('normalized_model.obj', 'model_normalized.obj'))