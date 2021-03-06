
target_folder=output

bash leveraging_geometry_for_shape_estimation/models/models.sh $target_folder
bash leveraging_geometry_for_shape_estimation/data_conversion/data_conversion.sh $target_folder
bash leveraging_geometry_for_shape_estimation/segmentation/segmentation.sh $target_folder
bash leveraging_geometry_for_shape_estimation/retrieval/retrieval.sh $target_folder
bash leveraging_geometry_for_shape_estimation/keypoint_matching/keypoint_matching.sh $target_folder
bash leveraging_geometry_for_shape_estimation/pose_and_shape_optimisation/pose_and_shape_optimisation.sh $target_folder