import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import json

new_render = '/data/cornucopia/fml35/experiments/test_output_all_s2/models/render_black_background/'
old_render = '/data/cvfs/fml35/derivative_datasets/pix3d_new/own_data/rendered_models/model_blender_256_black_background/'
# new_depths = '/data/cvfs/fml35/derivative_datasets/pix3d_new/own_data/depth/dummy_test/'

all = {}
all["pixel_diffs_render"] = []


for new_path in tqdm(glob(new_render + '*/*/*')): #['/data/cornucopia/fml35/experiments/test_output_all_s2/models/render_black_background/bed/IKEA_BEDDINGE/elev_015_azim_67.5.png']: #tqdm(glob(new_render + '*/*/*')):
    old_path = new_path.replace(new_render,old_render) #.replace('2.5.','2.0.').replace('7.5.','8.0.')

    old = cv2.imread(old_path)
    new = cv2.imread(new_path)

    mask = (old != new)
    mask = np.all(mask,axis=2)
    n_diff = int(np.sum(mask))

    all["pixel_diffs_render"].append({"new_path":new_path,"old_path": old_path, "n_diff":n_diff})

with open('/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/tests/test_renders_results.json','w') as f:
    json.dump(all,f)

print('done')