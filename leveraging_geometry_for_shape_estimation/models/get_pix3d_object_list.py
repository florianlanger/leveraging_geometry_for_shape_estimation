
import json
import sys
import os
import sys

def list_models(pix3d_file,out_file):
    # load all images in pix3d 
    with open(pix3d_file,'r') as json_file:
        pix3d_list = json.load(json_file)
    
    
    models = {}
    models["models"] = []
    names_already = []

    for info in pix3d_list[:5]:

        model_path = info["model"]
        split_path = model_path.split('/')
        model_dict = {}
        new_name = split_path[1] + '_' + split_path[2]
        
        if new_name not in names_already:
            model_dict["name"] = new_name
            model_dict["category"] = split_path[1]
            model_dict["model"] = model_path
            models["models"].append(model_dict)
            names_already.append(new_name)

    with open(out_file,'w') as json_file:
        json.dump(models, json_file, indent=4)

if __name__ == '__main__':

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)


    pix3d_file = global_config["dataset"]["dir_path"]  + 'pix3d.json'
    out_file = global_config["general"]["target_folder"] + '/models/model_list.json'

    list_models(pix3d_file,out_file)