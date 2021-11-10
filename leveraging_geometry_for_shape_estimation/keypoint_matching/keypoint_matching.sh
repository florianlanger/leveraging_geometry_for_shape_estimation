# python $1/code/keypoint_matching/detect_keypoints.py $1
# python $1/code/keypoint_matching/match_keypoints_2d.py $1
# python $1/code/keypoint_matching/get_matches_3d.py $1
# python $1/code/keypoint_matching/quality_matches.py $1

python leveraging_geometry_for_shape_estimation/keypoint_matching/detect_keypoints.py $1
python leveraging_geometry_for_shape_estimation/keypoint_matching/match_keypoints_2d.py $1
python leveraging_geometry_for_shape_estimation/code/keypoint_matching/get_matches_3d.py $1
python leveraging_geometry_for_shape_estimation/code/keypoint_matching/quality_matches.py $1
