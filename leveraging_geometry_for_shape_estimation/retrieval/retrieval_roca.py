import json
import sys
from tqdm import tqdm
import os

def main():
    # print('BUG ignore NONE prediction')
    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["dataset"]["which_dataset"] == 'scannet' and global_config["retrieval"]["gt"] == 'roca':

        target_folder = global_config["general"]["target_folder"]

        with open(global_config["dataset"]["roca_results"],'r') as file:
            roca = json.load(file)


        for img in tqdm(roca):

            counter = 0
            for detection in roca[img]:

                if not os.path.exists(target_folder + '/gt_infos/' + img.split('/')[0] + '-' + img.split('/')[2].split('.')[0] + '.json'):
                    continue

                detection_name = img.split('/')[0] + '-' + img.split('/')[2].split('.')[0] + '_' + str(counter).zfill(2)

                cat = detection["category"].replace('bookcase','bookshelf')
                nn_dict = {}
                nn_dict["model"] = "model/" + cat + '/' + detection["scene_cad_id"][1] + '/model_normalized.obj'
                nn_dict["category"] = cat
                nn_dict["name"] = cat + '_' + detection["scene_cad_id"][1]
                
                nn_all = {}
                nn_all["nearest_neighbours"] = []
                nn_all["nearest_neighbours"].append(nn_dict)

                counter += 1

                # if not os.path.exists(target_folder + '/gt_infos/' + img.split('/')[0] + '-' + img.split('/')[2].split('.')[0] + '.json'):
                #     continue

                with open(target_folder + '/nn_infos/' + detection_name + '.json','w') as file:
                    json.dump(nn_all,file)
                
                
    

if __name__ == '__main__':
    main()