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

    dataset_dir = '/scratch/fml35/datasets/cubes_01_large/'
    sw = 32

    # gpus = enable_gpus("CUDA", [3])
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    pi = 3.14159265

    

    # Blender
    scene = bpy.data.scenes["Scene"]
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
    
    # Lighting
    light_data = bpy.data.lights.new(name="light_1", type='POINT')
    light_data.energy = 20
    for i in [-1,1]:
        for j in [-1,1]:
            light_object = bpy.data.objects.new(name="light_1", object_data=light_data)
            bpy.context.collection.objects.link(light_object) 
            bpy.context.view_layer.objects.active = light_object
            light_object.location = (i, j, 0)
    light_object = bpy.data.objects.new(name="light_1", object_data=light_data)
    bpy.context.collection.objects.link(light_object) 
    bpy.context.view_layer.objects.active = light_object
    light_object.location = (0, 2, 1)
    

    # load model list
    with open(dataset_dir + 'img_info.json','r') as f:
        img_infos = json.load(f)

    for j in range(0,len(img_infos)):


        w,h = img_infos[j]["w"],img_infos[j]["h"]
        f = img_infos[j]["f"]

        scene.render.resolution_x = w
        scene.render.resolution_y = h

        if w >= h:
            fov = np.arctan((sw/2) / f ) * 2
        else:
            fov = np.arctan((sw*h/(2*w)) / f ) * 2

        scene.camera.data.angle = fov # *(pi/180.0)




    
        imported_object = bpy.ops.import_scene.obj(filepath=dataset_dir + 'models/' + img_infos[j]["model"])
        obj_object = bpy.context.selected_objects[0]

        # remove material
        for ob in bpy.context.selected_editable_objects:
            ob.active_material_index = 0
            for i in range(len(ob.material_slots)):
                bpy.ops.object.material_slot_remove({'object': ob})

        newMat = mathutils.Matrix(img_infos[j]["rot_mat"])
        R = newMat.to_euler('ZYX')
        rotation_mode = 'ZYX'
        T = img_infos[j]["trans_mat"]

        output_path = dataset_dir + 'img/' + img_infos[j]["img"]
        render_object(obj_object,R,T,output_path,scene,rotation_mode)
        
        # remove object
        bpy.ops.object.select_all(action='DESELECT')
        obj_object.select_set(True)
        bpy.ops.object.delete()
    
main()