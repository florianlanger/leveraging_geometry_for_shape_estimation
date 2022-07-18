import json
import os
from tqdm import tqdm
import shutil

with open('/scratch2/fml35/datasets/shapenet_v2/ShapeNetCore.v2/taxonomy.json') as file:
    shapenet_taxonomy = json.load(file)
catid_to_cat = {}
for item in shapenet_taxonomy:
    catid_to_cat[item["synsetId"]] = item["name"]

with open('/scratch2/fml35/datasets/scannet/scan2cad_annotations/cad_appearances.json','r') as file:
    cad_appearances = json.load(file)

cats = ["bathtub,bathing tub,bath,tub","bed","ashcan,trash can,garbage can,wastebin,ash bin,ash-bin,ashbin,dustbin,trash barrel,trash bin","bookshelf","cabinet","chair","display,video display","sofa,couch,lounge","table"]
cats_renamed = ["bathtub","bed","bin","bookshelf","cabinet","chair","display","sofa","table"]

orig_dir = '/scratch2/fml35/datasets/shapenet_v2/ShapeNetCore.v2/'
target_dir = '/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/'

# for cat in cats_renamed:
#     os.mkdir(target_dir + cat)

for scene in tqdm(cad_appearances):
    for object in cad_appearances[scene]:
        catid = object.split('_')[0]
        model = object.split('_')[1]

        cat = catid_to_cat[catid]

        if cat in cats:

            index = cats.index(cat)
            new_cat_name = cats_renamed[index]

            if not os.path.exists(target_dir + new_cat_name + '/' + model):
                os.mkdir(target_dir + new_cat_name + '/' + model)

            old_path = orig_dir + catid + '/' + model + '/models/model_normalized.obj'
            new_path = target_dir + new_cat_name + '/' + model + '/model_normalized.obj'

            shutil.copy(old_path,new_path)
