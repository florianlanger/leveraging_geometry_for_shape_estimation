
import json


with open('/scratches/octopus_2/fml35/datasets/scannet/scan2cad_annotations/full_annotations.json','r') as f:
    annotations = json.load(f)

with open('/scratches/octopus_2/fml35/datasets/scannet/data_splits/scannetv2_train.txt','r') as f:
    train = f.readlines()
train_scenes = [x.strip() for x in train]

seen_cad_ids = []


for i in range(len(annotations)):
    
    scene = annotations[i]['id_scan']
    if scene not in train_scenes:
        continue
    for j in range(len(annotations[i]['aligned_models'])):
        catid = annotations[i]['aligned_models'][j]['catid_cad']
        seen_cad_ids.append(annotations[i]['aligned_models'][j]['id_cad'])

with open('/scratch2/fml35/datasets/scannet/data_splits/seen_cad_ids.json','w') as f:
    json.dump(seen_cad_ids,f)