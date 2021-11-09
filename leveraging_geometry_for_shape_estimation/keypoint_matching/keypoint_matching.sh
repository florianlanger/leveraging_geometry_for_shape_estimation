python $1/code/keypoint_matching/detect_keypoints.py $1
# echo 'no keypoint detection'
python $1/code/keypoint_matching/match_keypoints_2d.py $1
python $1/code/keypoint_matching/get_matches_3d.py $1
python $1/code/keypoint_matching/quality_matches.py $1
