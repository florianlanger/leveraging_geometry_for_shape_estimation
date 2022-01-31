# python $2/keypoint_matching/detect_keypoints_whole_img.py $1
# python $2/keypoint_matching/filter_points.py $1
python $2/keypoint_matching/detect_keypoints.py $1
# python $2/keypoint_matching/keypoints_orig_size.py $1
python $2/keypoint_matching/match_keypoints_2d.py $1
python $2/keypoint_matching/get_matches_3d.py $1
python $2/keypoint_matching/quality_matches.py $1
