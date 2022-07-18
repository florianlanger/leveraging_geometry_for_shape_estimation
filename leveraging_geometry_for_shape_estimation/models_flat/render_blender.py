import bpy
import os
import numpy as np
import sys
import socket
import pickle
import math
import os
import json


def set_camera_6dof(scene,x,y,z,rx,ry,rz):
    scene.camera.location.x = x
    scene.camera.location.y = y
    scene.camera.location.z = z
    scene.camera.rotation_euler[0] = rx
    scene.camera.rotation_euler[1] = ry
    scene.camera.rotation_euler[2] = rz
    
    
def render_object(obj_object,R,T,elev,azim,output_dir,scene):
    obj_object.rotation_mode = 'ZYX'
    obj_object.location = tuple(T)

    elev = str(int(elev)).zfill(3)
    azim = str(np.round(azim,1)).zfill(3)
    name = 'elev_{}_azim_{}.png'.format(elev,azim)
    
    obj_object.rotation_euler = tuple(R)
    scene.render.filepath = output_dir + '/' + name
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

    # gpus = enable_gpus("CUDA", [3])
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    target_folder = "/scratch/fml35/experiments/exp_005_cubes" 

    global_info = target_folder + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    pi = 3.14159265
    fov = global_config["models"]["fov"]
    img_size = global_config["models"]["img_size"]
    

    # Blender
    scene = bpy.data.scenes["Scene"]
    scene.render.resolution_x = img_size
    scene.render.resolution_y = img_size
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
    scene.camera.data.angle = fov*(pi/180.0)
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
    
    
    # load R_and_T
    R_and_T = np.load(global_config["general"]["target_folder"] + '/models/rotations/R_T_blender.npz')
    elev = global_config["models"]["elev"]
    azim = global_config["models"]["azim"]

    # load model list
    with open(global_config["general"]["target_folder"] + "/models/model_list.json",'r') as f:
        model_list = json.load(f)

    for j in range(0,len(model_list)):

        model_path = global_config["general"]["target_folder"] + "/models/render_no_background/" +  model_list[j]["name"]
        if not os.path.exists(model_path):
            os.mkdir(model_path)
    
        imported_object = bpy.ops.import_scene.obj(filepath=global_config["dataset"]["dir_path"] + model_list[j]["name"] + '.obj')
        obj_object = bpy.context.selected_objects[0]

        # remove material
        for ob in bpy.context.selected_editable_objects:
            ob.active_material_index = 0
            for i in range(len(ob.material_slots)):
                bpy.ops.object.material_slot_remove({'object': ob})
        for i in range(R_and_T["R"].shape[0]):
            elev_index = i // len(azim)
            # NOTE: complicated formula because of old convention
            azim_index = (len(azim) - i % len(azim)) % len(azim)
            print(model_path)
            render_object(obj_object,R_and_T["R"][i],R_and_T["T"][i],elev[elev_index],azim[azim_index],model_path,scene)
        
        # remove object
        bpy.ops.object.select_all(action='DESELECT')
        obj_object.select_set(True)
        bpy.ops.object.delete()
    
main()