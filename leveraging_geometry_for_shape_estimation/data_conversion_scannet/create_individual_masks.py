import numpy as np
import os
import json
from matplotlib import pyplot as plt
import cv2
from scipy.interpolate import splprep, splev

from mask_images import mask_images_single_scene

def separate_frame(frame_path,frame_path_depth,n_models):

    depths = []
    for i in range(n_models):
        depths.append(np.load(frame_path_depth + '/object_{}.npy'.format(str(i).zfill(3))))
    depths = np.stack(depths)
    masks = depths > -1

    depths[depths == -1.] = 1000
    indices = np.argmax(-1 * depths, axis = 0)

    for i in range(n_models):
        mask = np.logical_and(indices == i, masks[i])
        cv2.imwrite(frame_path + '/object_{}.png'.format(i),mask*255)
    

def masks_single_scene(scene_info,dir_path):

    frames_dir = os.listdir(dir_path + scene_info['id_scan'] + '/color')
    frames = [frame.replace('.jpg','') for frame in frames_dir]
    path_masks = dir_path + scene_info['id_scan'] + '/masks'

    n_models = scene_info["n_aligned_models"]
    if not os.path.exists(path_masks):
        os.mkdir(path_masks)

    for frame in frames:
        frame_path = path_masks + '/' + frame
        frame_path_depth = dir_path + scene_info['id_scan'] + '/depth_for_masks/' + frame
        if not os.path.exists(frame_path):
            os.mkdir(frame_path)
        
        separate_frame(frame_path,frame_path_depth,n_models)
            

def main():

    with open('/data/cornucopia/fml35/scannet/scan2cad_annotations/full_annotations.json') as json_file:
        all_data = json.load(json_file)
    
    dir_path = '/data/cornucopia/fml35/scannet/25kframes/tasks/scannet_frames_25k/'

    for i in range(len(all_data)):
        scene_info = all_data[i]
        print(scene_info["id_scan"])

        masks_single_scene(scene_info,dir_path)
        mask_images_single_scene(scene_info,dir_path)



if __name__ == '__main__':
    main()





















def create_markers(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.threshold(gray, 16, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    markers = np.zeros_like(gray).astype(np.int32)
    for i, cnt in enumerate(cnts):
        markers = cv2.drawContours(markers, [cnt], -1, i+1, cv2.FILLED)

    print(markers[300,300])
    return markers


def separate_mask(image,color,markers):

    # Assuming we only have markers now; iterate all values and crop image part
    # for i in np.arange(1, np.max(markers[:, :])+1):
    #     pixels = np.array(np.where(markers == i)).astype(np.int32)
    #     x1 = np.min(pixels[1, :])
    #     x2 = np.max(pixels[1, :])
    #     y1 = np.min(pixels[0, :])
    #     y2 = np.max(pixels[0, :])
        # cv2.imwrite(str(i) + '.png', image[y1:y2, x1:x2, :])

    image_0_to_1 = image/255.
    print(np.max(image))
    print(image[300,300])
    mask = np.all(np.abs(image_0_to_1 - np.array(color)) < 0.05, axis=-1)
    print(mask)
    print(mask.shape)
    mask = (mask * 255).astype(np.uint8)

    mask_contour = (mask/255.).astype(cv2.float32)
    print(mask_contour.dtype)
    # contours,hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours,hierachy = cv2.findContours(mask_contour, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

    print(contours)
    print(len(contours))
    smoothened = []
    for contour in contours:
        x,y = contour.T
        print(x)
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        print(x,y)
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        tck, u = splprep([x,y], u=None, s=1.0, per=1)
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        u_new = np.linspace(u.min(), u.max(), 25)
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
        x_new, y_new = splev(u_new, tck, der=0)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
        smoothened.append(np.asarray(res_array, dtype=np.int32))

    # Overlay the smoothed contours on the original image

    
    empty_image = np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)
    final_mask = cv2.drawContours(empty_image, smoothened, -1, (255), cv2.FILLED)
    return final_mask #mask#image[y1:y2, x1:x2, :]
