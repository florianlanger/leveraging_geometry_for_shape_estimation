import sys
import json
import os
from tqdm import tqdm


def main():
    print('create gt infos')
    path_pix3d_file = '/data/cvfs/fml35/derivative_datasets/pix3d_new/pix3d.json'

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    target_folder = global_config["general"]["target_folder"]

    with open(path_pix3d_file,'r') as file:
        pix3d = json.load(file)

    list_images = os.listdir(global_config["general"]["image_folder"])


    for data in tqdm(pix3d):

        if data["img"].replace('img/','').replace('/','_') in list_images:
            name = data['category'] + '_' + data['img'].split('/')[2].split('.')[0]

            out_path = target_folder + '/gt_infos/' + name + '.json'
            if os.path.exists(out_path):
                continue

            infos = {}
            infos['name'] = name
            infos['img']  = data['category'] + '_' + data['img'].split('/')[2]

            for key in ['category','bbox','model','focal_length','rot_mat','trans_mat','img_size']:
                infos[key] = data[key] 
            
            with open(out_path ,'w') as file:
                json.dump(infos,file)



if __name__ == '__main__':
    main()