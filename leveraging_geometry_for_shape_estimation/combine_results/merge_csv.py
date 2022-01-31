

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


def combine_information(exp_1,exp_2):

    path_1 = exp_1 + '/global_stats/all_infos.csv'
    path_2 = exp_2 + '/global_stats/all_infos.csv'

    out_path = exp_1 + '/global_stats/merge_{}_{}.csv'.format(exp_1.split('/')[-1],exp_2.split('/')[-1])

    df_1 = pd.read_csv(path_1)
    df_2 = pd.read_csv(path_2)

    print(df_1['detection'])

    assert (df_1['detection'] == df_2['detection']).all()
    
    col_1 = df_1.columns
    new_col_1 = {col:'df_1_' + col for col in col_1}
    df_1 = df_1.rename(columns=new_col_1)

    col_2= df_2.columns
    new_col_2 = {col:'df_2_' + col for col in col_2}
    df_2 = df_2.rename(columns=new_col_2)

    merged = df_1.join(df_2)
    merged.to_csv(out_path)

    
        

            





def main():

    exp_1 = sys.argv[1]
    exp_2 = sys.argv[2]

    combine_information(exp_1,exp_2)

if __name__ == '__main__':
    main()