import os
import json

list_imgs = os.listdir('/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/3d_future/data_01/images')

with open('/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/3d_future/data_01/global_stats/visualisation_images.json','w') as f:
    json.dump(list_imgs,f)
