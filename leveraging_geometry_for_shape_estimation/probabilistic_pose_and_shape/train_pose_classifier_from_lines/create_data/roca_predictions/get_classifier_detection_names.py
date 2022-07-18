
import json
import os

dir = '/scratch/fml35/experiments/eval_classifier_grid/exp_006_roca_scale_gt_retrieval/T_lines_vis'
out_path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/classifier_roca_scale_detections.json'

detections = []

for file in os.listdir(dir):
    if 'closest_T' not in file:
        det_name = file.rsplit('_',7)[0]
        detections.append(det_name)

with open(out_path,'w') as file:
    json.dump(detections, file,indent=4)
   