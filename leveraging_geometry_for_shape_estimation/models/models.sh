# python $1/code/models/create_rotations.py $1
# python $1/code/models/get_pix3d_object_list.py $1
# ~/applications/blender-2.91.2-linux64/blender -b --python $1/code/models/render_blender.py $1
# python $1/code/models/black_background.py $1
# python $1/code/models/remesh_models.py $1
# python $1/code/models/render_depth.py $1
# python $1/code/models/render_wc_gt.py $1
path_to_blender=~/applications/blender-2.91.2-linux64/blender

python leveraging_geometry_for_shape_estimation/models/create_rotations.py $1
python leveraging_geometry_for_shape_estimation/models/get_pix3d_object_list.py $1
$path_to_blender -b --python leveraging_geometry_for_shape_estimation/models/render_blender.py
python leveraging_geometry_for_shape_estimation/models/black_background.py $1
python leveraging_geometry_for_shape_estimation/models/remesh_models.py $1
python leveraging_geometry_for_shape_estimation/models/render_depth.py $1
python leveraging_geometry_for_shape_estimation/models/render_wc_gt.py $1