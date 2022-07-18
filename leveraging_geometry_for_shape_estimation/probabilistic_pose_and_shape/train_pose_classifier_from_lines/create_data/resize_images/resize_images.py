import cv2
from glob import glob
import os
from tqdm import tqdm


main_path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/'


for size in [(160,120)]:
    for type in ['val']:

        in_dir = main_path + '{}/norm_gb_medium_480_360'.format(type)
        out_dir = in_dir.rsplit('_',2)[0] + '_' + str(size[0]) + '_' + str(size[1])
        os.mkdir(out_dir)
        print('outdir',out_dir)

        for path in tqdm(sorted(glob(in_dir + '/*'))):
            image = cv2.imread(path,cv2.IMREAD_UNCHANGED)
            resized = cv2.resize(image,size)
            cv2.imwrite(path.replace(in_dir,out_dir),resized)