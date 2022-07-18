import numpy as np
import torch
import json

def create_dict(*args):
    print(locals())
    print(args)
    return dict({i:eval(i,locals()) for i in args})

def load_json(path):
    with open(path,'r') as f:
        loaded = json.load(f)
    return loaded
# def test_in_func():
#     n_accepted_all_Ts = "yo"
#     b = torch.Tensor([0,4])
#     print(dict_of("n_accepted_all_Ts", "b"))
#     print(create_dict("n_accepted_all_Ts", "b"))

# def test_2():
#     fruitdict = {}
#     for i in ('apple', 'banana', 'carrot'):
#         fruitdict[i] = locals()[i]
#     print(fruitdict)
# # test_2()
# test_in_func()

def open_json_precomputed_or_current(end_of_path,global_config,which_data):

    if which_data in global_config["general"]["use_existing"]:
        path_base = global_config["general"]["path_to_existing_data"]
    else:
        path_base = global_config["general"]["target_folder"]

    with open(path_base + end_of_path,'r') as open_f:
        return json.load(open_f)

def determine_base_dir(global_config,which_stage):
    if which_stage in global_config["general"]["use_existing"]:
        path_base = global_config["general"]["path_to_existing_data"]
    else:
        path_base = global_config["general"]["target_folder"]
    return path_base