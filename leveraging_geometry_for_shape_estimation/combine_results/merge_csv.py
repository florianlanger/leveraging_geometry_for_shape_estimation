

from tqdm import tqdm
import os
import json
import cv2
import numpy as np
from numpy.lib.utils import info
import torch
import os
import sys
import json
import pandas as pd


def combine_information(exp_1,exp_2,exp_3):

    path_1 = exp_1 + '/global_stats/all_infos_small_v3_tolerant_rotation.csv'
    path_2 = exp_2 + '/global_stats/all_infos_small_v3.csv'
    path_3 = exp_3 + '/global_stats/all_infos_small_v3.csv'

    name_1 = exp_1.split('/')[-1]
    name_2 = exp_2.split('/')[-1]
    name_3 = exp_3.split('/')[-1]

    out_path = exp_1 + '/global_stats/merge_v4_{}_{}_{}.csv'.format(name_1,name_2,name_3)

    df_1 = pd.read_csv(path_1)
    df_2 = pd.read_csv(path_2)
    df_3 = pd.read_csv(path_3)


    assert (df_1['detection'] == df_2['detection']).all()
    assert (df_1['detection'] == df_3['detection']).all()
    
    col_1 = df_1.columns

    new_col_1 = {col: name_1 + '_' + col for col in col_1}
    df_1 = df_1.rename(columns=new_col_1)

    col_2= df_2.columns
    new_col_2 = {col:name_2 + '_' + col for col in col_2}
    df_2 = df_2.rename(columns=new_col_2)

    col_3= df_3.columns
    new_col_3 = {col:name_3 + '_' + col for col in col_3}
    df_3 = df_3.rename(columns=new_col_3)


    merged = df_1.join(df_2)
    merged = merged.join(df_3)
    merged.to_csv(out_path)

    
        

            





def main():

    # exp_1 = sys.argv[1]
    # exp_2 = sys.argv[2]
    # exp_3 = sys.argv[3]

    exp_1 = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_186_roca_retrieval_gt_z_lines_octopus'
    exp_2 = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_178_gt_retrieval_matches_gt_z'
    exp_3 = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_156_roca_all_vis'

    combine_information(exp_1,exp_2,exp_3)

if __name__ == '__main__':
    main()