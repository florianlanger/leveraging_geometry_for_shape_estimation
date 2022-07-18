
import numpy as np
from glob import glob
from tqdm import tqdm



def compare_dirs(dir_1,dir_2):

    for file in tqdm(glob(dir_1 + '*')):
        # print(file)
        with open(file) as f:
            anno1 = f.readlines()

        with open(file.replace(dir_1,dir_2)) as f:
            anno2 = f.readlines()

        # assert len(anno1) == len(anno2),(file,len(anno1),len(anno2))
        if len(anno1) != len(anno2):
            print(file)
        #     continue
        # print()
        for i in range(len(anno1)):

            split_1 = anno1[i].replace('\n','').split(',')
            split_2 = anno2[i].replace('\n','').split(',')

            assert split_1[0].zfill(8) == split_2[0],(split_1[0],split_2[0],i)
            # for j in [0,2]:
            #     assert split_1[j] == split_2[j],(split_1[j],split_2[j])
            for j in range(2,13):
                if np.abs(float(split_1[j]) - float(split_2[j])) > 0.00001:
                    print(i,split_1[3:6])
                    print(i,split_2[3:6])

if __name__ == '__main__':
    # dir_1 = '/scratch2/fml35/results/ROCA/roca_evaluation_code/filtered_scenes/'
    # dir_2 = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_156_roca_all_vis/global_stats/eval_scannet/results_per_scene_filtered/'

    # dir_1 = '/scratch2/fml35/results/ROCA/roca_evaluation_code/filtered_scenes_masked_too_many_objects/'
    # dir_2 = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_156_roca_all_vis/global_stats/eval_scannet/results_per_scene_kept_scan2cad_constraints/'

    dir_1 = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_156_roca_all_vis/global_stats/eval_scannet/results_per_scene_scan2cad_constraints/'
    dir_2 = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_156_roca_all_vis/global_stats/eval_scannet/results_per_scene_kept_scan2cad_constraints/'
    compare_dirs(dir_1,dir_2)