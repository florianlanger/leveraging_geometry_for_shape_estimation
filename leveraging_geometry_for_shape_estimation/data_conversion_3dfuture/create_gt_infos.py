import json
from tqdm import tqdm
import numpy as np

def main():
    print('FIGURE OUT SW')
    sw = 1200
    outdir = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/3d_future/data_01'

    path_future = '/scratch/fml35/datasets/3d_future/3D-FUTURE-scene/GT/test_set.json'
    with open(path_future,'r') as f:
        annos = json.load(f)

    all_new_annos =  {}

    for i in tqdm(range(len(annos['annotations'][:1000000]))):

        image_id = annos['annotations'][i]['image_id'] - 1
        image_name = annos['images'][image_id]['file_name']

        fov = annos['annotations'][i]['fov']

        if fov != None and fov != '':

            if image_id not in all_new_annos:

                infos = {}
                infos['name'] = annos['images'][image_id]['file_name']
                infos['img'] = annos['images'][image_id]['file_name'] + '.jpg'
                infos['img_size'] = [annos['images'][image_id]['width'],annos['images'][image_id]['height']]
                fov = annos['annotations'][i]['fov']
                # infos['focal_length'] = sw / (2 *np.arctan(fov/2))
                infos['focal_length'] = (sw /2) / np.tan(fov/2)
                # infos['fov'] = fov
                # infos['sw'] = np.tan(fov/2) * 2 * infos['focal_length']
                infos['objects'] = []
                all_new_annos[image_id] = infos

            obj = {}

            rotate = np.array([[-1,0,0],[0,1,0],[0,0,-1]],dtype=np.float32)
            R = np.matmul(rotate,annos['annotations'][i]['pose']['rotation'])
            T = np.matmul(rotate,np.transpose([annos['annotations'][i]['pose']['translation']]))
            obj['rot_mat'] = R.tolist()
            obj['trans_mat'] = np.transpose(T)[0].tolist()
            obj['scaling'] = [1.,1.,1.]
            # print(obj['rot_mat'])
            # print(obj['trans_mat'])
            # print(sfd)

            # print(annos['annotations'][i]['category_id'])
            category = annos['categories'][annos['annotations'][i]['category_id']-1]['super-category']
            category = category.replace('/','').lower()

            obj['model'] = "model/{}/{}/raw_model.obj".format(category,annos['annotations'][i]['model_id'])
            obj['category'] = category
            obj['bbox'] = annos['annotations'][i]['bbox'][:2] + (np.array(annos['annotations'][i]['bbox'][2:]) + np.array(annos['annotations'][i]['bbox'][:2])).tolist()              
            obj['mask_path'] = image_name + '_' + str(len(all_new_annos[image_id]['objects'])).zfill(2) + '.png'
    
            all_new_annos[image_id]['objects'].append(obj)

    
    for key in tqdm(all_new_annos):
        out_path = outdir + '/gt_infos/' + all_new_annos[key]['name'] + '.json'
        with open(out_path,'w') as f:
            json.dump(all_new_annos[key],f,indent=4)
    

if __name__ == '__main__':
    main()