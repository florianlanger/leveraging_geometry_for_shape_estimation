
import os
import cv2


seg_1 = '/data/cornucopia/fml35/experiments/debug_segmentation_1/segmentation_masks'
seg_2 = '/data/cornucopia/fml35/experiments/test_output_all_s2/segmentation_masks'

for name in os.listdir(seg_1):
    
    path_1 = seg_1 + '/' + name
    path_2 = seg_2 + '/' + name

    print(path_1)

    assert path_1 != path_2
    img_1 = cv2.imread(path_1)
    img_2 = cv2.imread(path_2)

    assert (img_1 == img_2).all() , print(path_1)