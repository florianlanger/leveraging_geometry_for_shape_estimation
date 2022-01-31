import bpy
import os
import numpy as np
import sys
import socket
import pickle
import math
import os
import json
import mathutils


def set_camera_6dof(scene,x,y,z,rx,ry,rz):
    scene.camera.location.x = x
    scene.camera.location.y = y
    scene.camera.location.z = z
    scene.camera.rotation_euler[0] = rx
    scene.camera.rotation_euler[1] = ry
    scene.camera.rotation_euler[2] = rz
    
    
def render_object(obj_object,R,T,output_path,scene,rotation_mode):
    obj_object.rotation_mode = rotation_mode
    obj_object.location = tuple(T)
    obj_object.rotation_euler = tuple(R)
    scene.render.filepath = output_path
    bpy.ops.render.render( write_still=True )

def enable_gpus(device_type,device_list):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cuda_devices, opencl_devices = cycles_preferences.get_devices()
 
    if device_type == "CUDA":
        devices = cuda_devices
    elif device_type == "OPENCL":
        devices = opencl_devices
    else:
        raise RuntimeError("Unsupported device type")
 
    activated_gpus = []
 
    for i,device in enumerate(devices):
        if (i in device_list):
            device.use = True
            activated_gpus.append(device.name)
        else:
            device.use = False
 
    cycles_preferences.compute_device_type = device_type
    for scene in bpy.data.scenes:
        scene.cycles.device = "GPU"
 
    return activated_gpus






def main():

    gpus = enable_gpus("CUDA", [1])
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    out_folder = "/scratches/octopus/fml35/datasets/pix3d_new/own_data/real_images_3d/normals_debug_remeshed"
    target_folder = "/data/cornucopia/fml35/experiments/exp_024_debug"    

    global_info = target_folder + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    pi = 3.14159265

    

    # Blender
    scene = bpy.data.scenes["Scene"]

    print(scene.view_layers)

    scene.view_layers[0].use_pass_normal = True

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.max_bounces = 5
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 1
    bpy.context.scene.cycles.transmission_bounces = 1
    bpy.context.scene.render.tile_x = 16
    bpy.context.scene.render.tile_y = 16
    bpy.context.scene.cycles.samples = 64
    
    for o in bpy.context.scene.objects:
        if o.type != 'CAMERA':
            o.select_set(True)
        else:
            o.select_set(False)

    # Call the operator only once
    bpy.ops.object.delete()

    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    
    scene.camera.rotation_mode = 'ZYX'
    

    # set camera
    set_camera_6dof(scene,0,0,0,pi,0,pi)
    

    # load model list
    with open(global_config["dataset"]["pix3d_path"] + "/pix3d.json",'r') as f:
        pix3d = json.load(f)

    for j in range(0,len(pix3d)):
        cat_path = out_folder + "/" + pix3d[j]["category"]
        if not os.path.exists(cat_path):
            os.mkdir(cat_path)


        w,h = pix3d[j]["img_size"]

        scene.render.resolution_x = 2 * w
        scene.render.resolution_y = 2 * h
        if w >= h:
            fov = np.arctan((global_config["pose_and_shape"]["pose"]["sensor_width"]/2) / pix3d[j]["focal_length"] ) * 2
        else:
            fov = np.arctan((global_config["pose_and_shape"]["pose"]["sensor_width"]*h/(2*w)) / pix3d[j]["focal_length"] ) * 2
        scene.camera.data.angle = fov # *(pi/180.0)
    
        imported_object = bpy.ops.import_scene.obj(filepath=global_config["dataset"]["pix3d_path"] + '/' + pix3d[j]["model"])
        # imported_object = bpy.ops.import_scene.obj(filepath='/scratch/fml35/experiments/leveraging_geometry_for_shape/test_output_all_s2/models/remeshed/' + pix3d[j]["model"].replace('model/',''))
        obj_object = bpy.context.selected_objects[0]

        # remove material
        for ob in bpy.context.selected_editable_objects:
            ob.active_material_index = 0
            for i in range(len(ob.material_slots)):
                bpy.ops.object.material_slot_remove({'object': ob})
            
            mat = bpy.data.materials.get("Material")
            if ob.data.materials:
                # assign to 1st material slot
                ob.data.materials[0] = mat
            else:
                # no slots
                ob.data.materials.append(mat)

        output_path = out_folder + '/' + pix3d[j]["category"] + '/' + pix3d[j]["img"].split('/')[2].split('.')[0] + '.png'


        newMat = mathutils.Matrix(pix3d[j]["rot_mat"])
        R = newMat.to_euler('ZYX')
        T = pix3d[j]["trans_mat"]

        # rotation_mode = ['ZYX','ZXY','YXZ','YZX','XYZ','XZY']
        rotation_mode = ['ZYX']
        for rot_mode in rotation_mode:
            # output_path = out_folder + '/' + pix3d[j]["category"] + '/' + pix3d[j]["img"].split('/')[2].split('.')[0] + '_1.png'
            render_object(obj_object,R,T,output_path,scene,rot_mode)
        
        # remove object
        bpy.ops.object.select_all(action='DESELECT')
        obj_object.select_set(True)
        bpy.ops.object.delete()
    
main()