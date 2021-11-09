import json

path = '/data/cornucopia/fml35/experiments/test_output_all_s2/models/model_list.json'

path_with_order = '/data/cvfs/fml35/derivative_datasets/pix3d/model_list_big.json'

with open(path,'r') as f:
    file = json.load(f)

with open(path_with_order,'r') as f:
    file_order = json.load(f)


assert len(file_order["models"]) == len(file["models"])

# dict = {}
# for i in range(len(file_order["models"]):
#     dict[file_order["models"][i]['category'] + '_' + file_order["models"][i]['name']] = file["models"][i]["name"]

new_models = []
for i in range(len(file_order["models"])):
    for j in range(len(file["models"])):
        if file["models"][j]["name"] == file_order["models"][i]['category'] + '_' + file_order["models"][i]['name']:
            new_models.append(file["models"][j])
            break

new_dict = {"models": new_models}

with open(path.replace('model_list','model_list_old_order'),'w') as f:
    json.dump(new_dict,f)
