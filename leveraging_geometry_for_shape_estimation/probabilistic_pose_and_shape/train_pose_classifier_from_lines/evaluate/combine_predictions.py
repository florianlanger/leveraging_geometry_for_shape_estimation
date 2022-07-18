from enum import Flag
import json
import numpy as np

from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.ground_plane import get_model_to_infos_scannet_just_id



def combine_predictions_with_predictions_all_images(eval_path,predictions_all_images_path,ending_file='.jpg'):

    print('un normalise predictions here')

    with open(eval_path,'r') as f:
        predictions_own = json.load(f)

    with open(predictions_all_images_path,'r') as f:
        predictions_all_images = json.load(f)

    for img in predictions_all_images:
        for i in range(len(predictions_all_images[img])):
            if "own_prediction" in predictions_all_images[img][i]:
                predictions_all_images[img][i].pop("own_prediction")

    model_to_infos_scannet = get_model_to_infos_scannet_just_id()

    for detection in predictions_own:
        detection_id = int(detection.split('_')[-1].split('.')[0])
        if ending_file == '.jpg':
            gt_name = detection.split('-')[0] + '/color/' + detection.split('-')[1].split('_')[0] + ending_file
        elif ending_file == '.json':
            # print('detection',detection)
            # print('')
            gt_name = detection.split('-')[0] + '-' + detection.split('-')[1].split('_')[0] + ending_file
            # print('gt_name',gt_name)
        # for key in ['q','t','s']:
        for key in ['q','t']:
            predictions_all_images[gt_name][detection_id][key] = predictions_own[detection][key]

        factor = np.array(model_to_infos_scannet[predictions_own[detection]["model_id"]]['bbox']) * 2
        predictions_all_images[gt_name][detection_id]['s'] = (predictions_own[detection]['s'] / factor).tolist()

        predictions_all_images[gt_name][detection_id]["scene_cad_id"][1] = predictions_own[detection]["model_id"]
        predictions_all_images[gt_name][detection_id]["own_prediction"] = True


    for img in predictions_all_images:
        # print('img',img)
        for i in range(len(predictions_all_images[img])):
            if "own_prediction" not in predictions_all_images[img][i]:
                predictions_all_images[img][i]["own_prediction"] = False
            if predictions_all_images[img][i]["category"] == 'bookcase':
                predictions_all_images[img][i]["category"] = 'bookshelf'

            if ending_file == '.jpg':
                predictions_all_images[img][i]['detection'] = img.split('/')[0] + '-' + img.split('/')[2].split('.')[0] + '_' + str(i).zfill(2)
                

    return predictions_all_images

# def combine_predictions_with_roca_annos(eval_path,roca_path):


#     with open(eval_path,'r') as f:
#         predictions_own = json.load(f)

#     with open(roca_path,'r') as f:
#         predictions_all_images = json.load(f)


#     for detection in predictions_own:
#         detection_id = int(detection.split('_')[-1].split('.')[0])
#         gt_name = detection.split('-')[0] + '-' + detection.split('-')[1].split('_')[0] + ending_file
#         for key in ['q','t','s']:
#             predictions_all_images[gt_name][detection_id][key] = predictions_own[detection][key]
#         predictions_all_images[gt_name][detection_id]["scene_cad_id"][1] = predictions_own[detection]["model_id"]
#         predictions_all_images[gt_name][detection_id]["own_prediction"] = True

#     for img in predictions_all_images:
#         # print('img',img)
#         for i in range(len(predictions_all_images[img])):
#             if "own_prediction" not in predictions_all_images[img][i]:
#                 predictions_all_images[img][i]["own_prediction"] = False
#             if predictions_all_images[img][i]["category"] == 'bookcase':
#                 predictions_all_images[img][i]["category"] = 'bookshelf'

#             if ending_file == '.jpg':
#                 predictions_all_images[img][i]['detection'] = img.split('/')[0] + '-' + img.split('/')[2].split('.')[0] + '_' + str(i).zfill(2)
                

#     return predictions_all_images