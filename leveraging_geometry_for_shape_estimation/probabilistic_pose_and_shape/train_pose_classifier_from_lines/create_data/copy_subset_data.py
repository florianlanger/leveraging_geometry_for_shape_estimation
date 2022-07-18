import os
import shutil
from tqdm import tqdm


def main():

    source_dir = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/train/'
    target_dir = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_04_small/train/'

    os.mkdir(target_dir)

    names_to_copy = ['scene0010_00-000100','scene0008_00-000600','scene0017_00-001000','scene0054_00-001300','scene0061_00-000200','scene0703_00-000500','scene0675_01-000600']

    ignore_dirs = ['global_information.json','global_stats','code']

    for dir in tqdm(os.listdir(source_dir)):
        if dir not in ignore_dirs:
            os.mkdir(target_dir + dir)
            for file in os.listdir(source_dir + dir):

                base_name = file.split('_')[0] + '_' + file.split('_')[1].split('.')[0]

                if base_name in names_to_copy:
                    shutil.copy(source_dir + dir + '/' + file, target_dir + dir + '/' + file)

    shutil.copytree(source_dir + 'global_stats', target_dir + 'global_stats')


if __name__ == '__main__':
    main()