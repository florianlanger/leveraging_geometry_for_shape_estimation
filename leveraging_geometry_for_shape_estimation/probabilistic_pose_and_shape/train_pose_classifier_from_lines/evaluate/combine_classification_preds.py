

import json
import os

# eval_names = ['date_2022_07_02_time_13_27_29_EVAL_REFINE_1_date_2022_07_01_time_18_09_18_classification_adjusted_p_smaller_azim_network_epoch_000100_norm_medium_rotation_index_0',
#             'date_2022_07_02_time_13_34_23_EVAL_REFINE_1_date_2022_07_01_time_18_09_18_classification_adjusted_p_smaller_azim_network_epoch_000100_norm_medium_rotation_index_1',
#             'date_2022_07_02_time_13_42_05_EVAL_REFINE_1_date_2022_07_01_time_18_09_18_classification_adjusted_p_smaller_azim_network_epoch_000100_norm_medium_rotation_index_2',
#             'date_2022_07_02_time_13_49_27_EVAL_REFINE_1_date_2022_07_01_time_18_09_18_classification_adjusted_p_smaller_azim_network_epoch_000100_norm_medium_rotation_index_3']

eval_names = ['date_2022_07_08_time_18_07_18_EVAL_REFINE_1_date_2022_07_05_time_11_57_34_3_refinements_half_classifier_half_regressor_4_rotations_network_epoch_000200_norm_medium_rotation_index_0',
'date_2022_07_08_time_18_18_27_EVAL_REFINE_1_date_2022_07_05_time_11_57_34_3_refinements_half_classifier_half_regressor_4_rotations_network_epoch_000200_norm_medium_rotation_index_1',
'date_2022_07_08_time_18_28_11_EVAL_REFINE_1_date_2022_07_05_time_11_57_34_3_refinements_half_classifier_half_regressor_4_rotations_network_epoch_000200_norm_medium_rotation_index_2',
'date_2022_07_08_time_18_38_14_EVAL_REFINE_1_date_2022_07_05_time_11_57_34_3_refinements_half_classifier_half_regressor_4_rotations_network_epoch_000200_norm_medium_rotation_index_3']
# eval_names = ['date_2022_07_05_time_09_19_57_EVAL_REFINE_1_date_2022_07_03_time_12_15_58_classification_adjusted_p_smaller_azim_network_epoch_000200_norm_medium_rotation_index_0',
# 'date_2022_07_05_time_09_26_53_EVAL_REFINE_1_date_2022_07_03_time_12_15_58_classification_adjusted_p_smaller_azim_network_epoch_000200_norm_medium_rotation_index_1',
# 'date_2022_07_05_time_09_33_00_EVAL_REFINE_1_date_2022_07_03_time_12_15_58_classification_adjusted_p_smaller_azim_network_epoch_000200_norm_medium_rotation_index_2',
# 'date_2022_07_05_time_09_39_05_EVAL_REFINE_1_date_2022_07_03_time_12_15_58_classification_adjusted_p_smaller_azim_network_epoch_000200_norm_medium_rotation_index_3']


dir_path = '/scratch/fml35/experiments/regress_T/evals/'

add_on = '/predictions/epoch_000050/translation_pred_scale_pred_rotation_init_for_classification_retrieval_roca_all_images_True/our_single_predictions.json'


out_path = dir_path + eval_names[0] + add_on.replace('our_single_predictions', 'best_rotation_index')
assert os.path.exists(out_path) == False

predictions = {}
for file in eval_names:
    rotation_index = int(file.split('_')[-1])
    with open(dir_path + file + add_on, 'r') as f:
        predictions[rotation_index] = json.load(f)

for key in predictions:
    assert set(predictions[0]) == set(predictions[key])

rotation_index_per_detection_with_max_score = {}

for detection in predictions[0]:
    max_score = -1
    max_rotation_index = -1
    for rotation_index in predictions:
        score = predictions[rotation_index][detection]["classification_score"][0]
        if score > max_score:
            max_score = score
            max_rotation_index = rotation_index
    rotation_index_per_detection_with_max_score[detection] = max_rotation_index

with open(out_path, 'w') as f:
    json.dump(rotation_index_per_detection_with_max_score, f)
