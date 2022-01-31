import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import json
import os
from matplotlib import pyplot as plt

new_masks = '/data/cornucopia/fml35/experiments/test_output_all_s2/segmentation_masks/'
new_overlap = '/data/cornucopia/fml35/experiments/test_output_all_s2/bbox_overlap/'
old_masks = '/data/cvfs/fml35/derivative_datasets/pix3d_new/own_data/masks/masks_s2_all_relevant_predictions_swin_epoch_66'
# new_depths = '/data/cvfs/fml35/derivative_datasets/pix3d_new/own_data/depth/dummy_test/'


all = {}
all["pixel_diffs_crops"] = []



for cat in os.listdir(old_masks):
    for img in os.listdir(old_masks + '/' + cat): 
        if '_0' in img:
            old_path = old_masks + '/' + cat + '/' + img
            new_path = new_masks + cat + '_' + img.split('_')[0] + '_' + str(img.split('_')[1].split('.')[0]).zfill(2) + '.' + img.split('.')[1]

            new_overlap_path = new_overlap + cat + '_' + img.split('_')[0] + '_' + str(img.split('_')[1].split('.')[0]).zfill(2) + '.json'

            with open(new_overlap_path,'r') as f:
                overlap = json.load(f)

            if overlap["valid"] == False:
                new_path = new_masks + cat + '_' + img.split('_')[0] + '_' + str(img.split('_')[1].split('.')[0] + 1).zfill(2) + '.' + img.split('.')[1]


            assert new_path != old_path


            print(new_path)
            if os.path.exists(new_path):
                old = cv2.imread(old_path)
                new = cv2.imread(new_path)

                # plt.imshow(old)
                # plt.show()

                # plt.imshow(new)
                # plt.show()

                mask = (old != new)
                mask = np.all(mask,axis=2)
                # plt.imshow(mask)
                # plt.show()
                n_diff = int(np.sum(mask))
                print(n_diff)

                all["pixel_diffs_crops"].append({"new_path":new_path,"old_path": old_path, "n_diff":n_diff})

            with open('/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/tests/test_masks_results.json','w') as f:
                json.dump(all,f)

print('done')