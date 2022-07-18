
import json

scenes_1_path = '/scratch2/fml35/results/ROCA/roca_evaluation_code/per_scene.json'
with open(scenes_1_path,'r') as f:
    scenes_1 = json.load(f)

scenes_2_path = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_156_roca_all_vis/global_stats/results_scannet_scenes_without_retrieval.json'
with open(scenes_2_path,'r') as f:
    scenes_2 = json.load(f)

keys_1 = sorted([key for key in scenes_1])
keys_2 = sorted([key for key in scenes_2])

assert keys_1 == keys_2

for key in scenes_1:
    print(key)
    assert scenes_1[key]["n_good"] == scenes_2[key]["n_good"],(key,scenes_1[key],scenes_2[key])
    assert scenes_1[key]["n_total"] == scenes_2[key]["n_total"]