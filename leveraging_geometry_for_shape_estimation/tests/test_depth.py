import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import json

threshold = 0.00001
all = {}
all["pixel_diffs_render"] = []
all["threshold"] = threshold


new_depths = '/data/cornucopia/fml35/experiments/test_output_all_s2/models/depth/'
old_depths = '/data/cvfs/fml35/derivative_datasets/pix3d_new/own_data/depth/256/'


for new_path in glob(new_depths + '*/*/*'):#['/data/cornucopia/fml35/experiments/test_output_all_s2/models/depth/bed/IKEA_BRIMNES_1/elev_015_azim_67.5.npy']:#glob(new_depths + '*/*/*'):
    old_path = new_path.replace(new_depths,old_depths).replace('2.5.','2.0.').replace('7.5.','8.0.')
    assert old_path != new_path

    old = np.load(old_path)
    new = np.load(new_path)
    mask = (old - new) < threshold
    
    diffs = (old - new)[~mask]
    diffs = diffs.tolist()

    all["pixel_diffs_render"].append({"new_path":new_path,"old_path": old_path, "diffs":diffs})

with open('/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/tests/test_depths_results.json','w') as f:
    json.dump(all,f)

print('done')