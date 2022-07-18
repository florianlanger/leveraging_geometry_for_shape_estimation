from glob import glob
import json
from tqdm import tqdm

sizes = []
counter = 0
for file in tqdm(glob('/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_203_new_lines_from_2d_all_vis/gt_infos/*.json')):
    with open(file,'r') as f:
        gt_infos = json.load(f)
    size = gt_infos["img_size"]
    if size not in sizes:
        sizes.append(size)

    if size == [1296,968]:
        counter += 1

print(sizes)