python $1/code/models/create_rotations.py $1
python $1/code/models/get_pix3d_object_list.py $1
~/applications/blender-2.91.2-linux64/blender -b --python $1/code/models/render_blender.py $1
python $1/code/models/black_background.py $1
python $1/code/models/remesh_models.py $1
python $1/code/models/render_depth.py $1
python $1/code/models/render_wc_gt.py $1