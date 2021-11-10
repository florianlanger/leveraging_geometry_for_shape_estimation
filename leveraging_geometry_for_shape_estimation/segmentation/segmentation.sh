# python $1/code/segmentation/get_gt_infos.py $1
# python $1/code/segmentation/gt_masks.py $1
# python $1/code/segmentation/segmentation.py $1
# python $1/code/segmentation/check_iou.py $1
# python $1/code/segmentation/mask_and_crop.py $1

python leveraging_geometry_for_shape_estimation/segmentation/get_gt_infos.py $1
python leveraging_geometry_for_shape_estimation/segmentation/gt_masks.py $1
python leveraging_geometry_for_shape_estimation/segmentation/segmentation.py $1
python leveraging_geometry_for_shape_estimation/segmentation/check_iou.py $1
python leveraging_geometry_for_shape_estimation/segmentation/mask_and_crop.py $1