
import json
import sys
import os
import sys
from turtle import shearfactor

def get_models_want():
    with open('/scratches/octopus/fml35/datasets/3d_future/3D-FUTURE-scene/GT/model_infos.json','r') as f:
        model_infos = json.load(f)
    models = [item['model_id'] for item in model_infos if item['is_train'] == 0]
    return models

def list_models(shape_dir,out_file):
    models = {}
    models["models"] = []

    models_want = get_models_want()


    for cat in os.listdir(shape_dir):
        for model in os.listdir(shape_dir + '/' + cat):

            if model not in models_want:
                continue
            model_dict = {}
            model_dict["name"] = cat + '_' + model
            model_dict["category"] = cat
            model_dict["model"] = 'model/' + cat + '/' + model + '/model_normalized.obj'
            # model_dict["model"] = 'model/' + cat + '/' + model + '/raw_model.obj'
            models["models"].append(model_dict)

    with open(out_file,'w') as json_file:
        json.dump(models, json_file, indent=4)

if __name__ == '__main__':

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    shape_dir = global_config["dataset"]["dir_path"] + 'model/'

    out_file = global_config["general"]["target_folder"] + '/models/model_list.json'

    list_models(shape_dir,out_file)