import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import json
import os

new_crops = '/data/cornucopia/fml35/experiments/test_output_all_s2/cropped_and_masked/'
old_crops = '/data/cvfs/fml35/derivative_datasets/pix3d_new/own_data/real_images/img_150/s2_swin_epoch_66/predicted_segmentation_predicted_bbox'
# new_depths = '/data/cvfs/fml35/derivative_datasets/pix3d_new/own_data/depth/dummy_test/'


# NOTE: two differences: 1. before saved note with whatever file ending had, (so many jpegs)
#                        2. before if had detection 1,2,3 but 3 was wrong class label detection were mapped 2 -> 1, 3-> 2, now dont do anymore

all = {}
all["pixel_diffs_crops"] = []

for cat in ['bookcase']:#os.listdir(old_crops):
    for img in os.listdir(old_crops + '/' + cat):
        old_path = old_crops + '/' + cat + '/' + img
        

        new_path = new_crops + cat + '_' + img.split('_')[0] + '_' + str(img.split('_')[1].split('.')[0]).zfill(2) + '.png'

        if not '_0.' in img:
            continue


        print(new_path)
        assert new_path != old_path
        old = cv2.imread(old_path)
        new = cv2.imread(new_path)

        new = new[53:203,53:203,:]

        mask = (old != new)
        mask = np.all(mask,axis=2)
        n_diff = int(np.sum(mask))

        all["pixel_diffs_crops"].append({"new_path":new_path,"old_path": old_path, "n_diff":n_diff})

        with open('/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/tests/test_crops_results.json','w') as f:
            json.dump(all,f)

print('done')