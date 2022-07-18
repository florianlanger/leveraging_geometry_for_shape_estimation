import cv2
import os
from tqdm import tqdm


dir_classifier = '/scratch/fml35/experiments/eval_classifier_grid/exp_006_roca_scale_gt_retrieval/T_lines_vis'
dir_vis = '/scratch/fml35/experiments/regress_T/runs_03_T_big/date_2022_05_31_time_16_57_26_EVAL_scale_roca_date_2022_05_26_time_19_04_11_three_refinements_epoch_255/vis'
out_dir = '/scratch/fml35/experiments/regress_T/runs_03_T_big/date_2022_05_31_time_16_57_26_EVAL_scale_roca_date_2022_05_26_time_19_04_11_three_refinements_epoch_255/vis_combined_classifier'
# os.mkdir(out_dir)

all_files_vis = os.listdir(dir_vis)

def find_file_vis(all_files_vis,detection):
    for file in all_files_vis:
        if detection in file:
            return file

for file in tqdm(sorted(os.listdir(dir_classifier))):
    if 'closest_T' not in file:
        img_path = os.path.join(dir_classifier,file)
        img = cv2.imread(img_path)

        for i in range(3):

            detection = "refinement_" + str(i).zfill(2) + '_' + file.rsplit('_',7)[0] + '_combined'
            file_vis = find_file_vis(all_files_vis,detection)
            vis_path = os.path.join(dir_vis,file_vis)

            assert os.path.exists(vis_path),'vis path does not exist {}'.format(vis_path)
            vis = cv2.imread(vis_path)

            vis[360*2:360*3,480*4:] = img[0:360,0:480+360]

            cv2.imwrite(os.path.join(out_dir,file_vis),vis)

