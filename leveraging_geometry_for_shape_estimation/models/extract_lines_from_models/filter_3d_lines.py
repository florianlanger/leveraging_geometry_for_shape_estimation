
import numpy as np

from glob import glob
import json
from tqdm import tqdm

def get_id_to_category():
    top = {}
    top["03211117"] = "display"
    top["04379243"] = "table"
    top["02808440"] = "bathtub"
    top["02747177"] = "bin"
    top["04256520"] = "sofa"
    top["03001627"] = "chair"
    top["02933112"] = "cabinet"
    top["02871439"] = "bookshelf"
    top["02818832"] = "bed"

    return top

def get_model_to_scaling(path_annot_file):

    with open(path_annot_file,'r') as f:
        annot = json.load(f)

    id_to_category = get_id_to_category()

    model_to_scaling = {}

    for scene in annot:
        for model in scene["aligned_models"]:
            # print(model)
            if model["catid_cad"] in id_to_category:
                name = id_to_category[model["catid_cad"]] + '_' + model["id_cad"]
                model_to_scaling[name] = model['trs']["scale"]
    return model_to_scaling


def main(input_dir,output_dir,min_length):
    path_annot_file = '/scratch2/fml35/datasets/scannet/scan2cad_annotations/full_annotations.json'
    model_to_scaling = get_model_to_scaling(path_annot_file)


    for path in tqdm(glob(input_dir + '/*')):
        lines_3D = np.load(path)

        model_name = path.rsplit('/',1)[1].split('.')[0]
        scaling = model_to_scaling[model_name]

        lines_3D_for_filter = lines_3D * np.array(scaling + scaling).astype(np.float32)
        mask_length = np.sum((lines_3D_for_filter[:,:3] - lines_3D_for_filter[:,3:6])**2,axis=1)**0.5 > min_length
        lines_3D = lines_3D[mask_length]
        np.save(path.replace(input_dir,output_dir),lines_3D)




if __name__ == '__main__':

    input_dir = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_117_scannet_models/models/lines'
    output_dir = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_117_scannet_models/models/lines_filtered'
    min_length = 0.1
    main(input_dir,output_dir,min_length)