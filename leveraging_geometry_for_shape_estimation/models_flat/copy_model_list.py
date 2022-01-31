import sys
import json
import shutil



if __name__ == '__main__':

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)


    in_file = global_config["dataset"]["pix3d_path"]  + '../model_info.json'
    out_file = global_config["general"]["target_folder"] + '/models/model_list.json'
    shutil.copy(in_file,out_file)