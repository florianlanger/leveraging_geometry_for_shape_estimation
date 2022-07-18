path_to_blender=~/applications/blender-2.91.2-linux64/blender

# python leveraging_geometry_for_shape_estimation/models/create_rotations.py $1
# python leveraging_geometry_for_shape_estimation/models/get_pix3d_object_list.py $1
# python leveraging_geometry_for_shape_estimation/models_scannet/get_scannet_object_list.py $1
# $path_to_blender -b --python leveraging_geometry_for_shape_estimation/models/render_blender.py
python leveraging_geometry_for_shape_estimation/models/black_background.py $1
# python leveraging_geometry_for_shape_estimation/models/remesh_models.py $1
# python leveraging_geometry_for_shape_estimation/models/render_depth.py $1
# python leveraging_geometry_for_shape_estimation/models/render_wc_gt.py $1

python leveraging_geometry_for_shape_estimation/models/detect_keypoints_models.py $1

python leveraging_geometry_for_shape_estimation/models/extract_3d_points.py $1
python leveraging_geometry_for_shape_estimation/models/filter_3d_points.py $1
# python leveraging_geometry_for_shape_estimation/models/extract_lines_from_models/extract_lines_from_surface_normal_faces.py $1