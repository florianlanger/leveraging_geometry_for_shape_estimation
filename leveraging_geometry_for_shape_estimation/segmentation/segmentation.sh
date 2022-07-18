# python $2/segmentation/get_gt_infos.py $1
# python $2/segmentation/gt_masks.py $1
# python $2/segmentation/segmentation_roca.py $1
# source "~/environments/mmdetection/bin/activate"
python $2/segmentation/segmentation.py $1
# echo "HERE"
# source "/home/mifs/fml35/environments/shape_env/bin/activate"
python $2/segmentation/check_iou.py $1
python $2/segmentation/mask_and_crop.py $1
python $2/segmentation/analyse_segmentation.py $1