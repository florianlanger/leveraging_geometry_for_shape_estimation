
import json
import os
from detectron2.structures import Boxes, BoxMode, pairwise_iou
import json
import torch
from tqdm import tqdm
import numpy as np
import cv2

from leveraging_geometry_for_shape_estimation.utilities.write_on_images import  draw_text_block
# go through own predictions and if have at least bbox overlap of 30 percent take roca, else augment own bbox

def draw_boxes(img, boxes, thickness=1,color=(0,255,0)):
    """
    Draw boxes on an image.

    Args:
        boxes (Boxes or ndarray): either a :class:`Boxes` instances,
            or a Nx4 numpy array of XYXY_ABS format.
        thickness (int): the thickness of the edges
    """
    img = img.astype(np.uint8)
    if isinstance(boxes, Boxes):
        boxes = boxes.clone("xyxy")
    else:
        assert boxes.ndim == 2, boxes.shape
    for box in boxes:
        (x0, y0, x1, y1) = (int(x + 0.5) for x in box)
        img = cv2.rectangle(img, (x0, y0), (x1, y1), color=color, thickness=thickness)
    return img

def calculate_box_overlap(gt_bbox,pred_bbox):
    gt_bbox = Boxes(torch.tensor([gt_bbox], dtype=torch.float32))
    pred_bbox = Boxes(torch.tensor([pred_bbox], dtype=torch.float32))

    boxiou = pairwise_iou(gt_bbox, pred_bbox)

    return boxiou.item()

def augment_bbox(bbox,infos,change_bbox_size_percent_img):

    min_bbox_size = 10

    max_change = change_bbox_size_percent_img * max(infos["img_size"])

    valid_bbox = False
    while valid_bbox == False:
        offset = np.random.uniform(-max_change,max_change,size=4)
        new_bbox = offset + np.array(bbox)
        new_bbox = np.clip(new_bbox,np.zeros(4),np.array(infos["img_size"] + infos["img_size"])-1)
        new_bbox = np.round(new_bbox)
        if (new_bbox[:2] + min_bbox_size < new_bbox[2:4]).all():
            valid_bbox = True
    return new_bbox


def get_best_overlaps(gt_bboxes,roca_bboxes):
    ious = []
        
    for bbox1 in gt_bboxes:
        ious.append([])
        for bbox2 in roca_bboxes:
            bbox_overlap = calculate_box_overlap(bbox1,bbox2)
            ious[-1].append(bbox_overlap)

    max_indices = [np.argmax(single_ious) for single_ious in ious]
    max_values = [np.max(single_ious) for single_ious in ious]
    return max_indices,max_values,ious

def load_empty_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.hconcat([img,img])
    return img

def roca_bboxes(roca,gt_infos,threshold_accept,change_bbox_size_percent_img,img_dir_path):
    img_size_orig = gt_infos["img_size"]
    img_path_vis = os.path.join(img_dir_path,gt_infos["img"])

    resize = np.array([img_size_orig[0]/480.,img_size_orig[1]/360.])
    resize = np.concatenate([resize,resize],axis=0)

    new_bboxes = []
    use_roca = []
    roca_objects = []
    gt_bboxes = [object['bbox'] for object in gt_infos['objects']]

    if len(gt_bboxes) == 0:
        empty_img = load_empty_img(img_path_vis)
        return [],[],[],[],empty_img

    name_for_roca = gt_infos["name"].split('-')[0]  + '/color/' + gt_infos["name"].split('-')[1] + '.jpg'

    if name_for_roca in roca:
        roca_img_info = roca[name_for_roca]
        roca_bboxes = [(resize*object['bbox']).tolist() for object in roca_img_info]
    else:
        roca_bboxes = [[0,0,1,1]]

    
    _,_,ious = get_best_overlaps(gt_bboxes,roca_bboxes)
    max_indices,max_values = get_max_indices_max_values_where_one_gt_only_one_roca(ious)


    for i in range(len(gt_bboxes)):
        if max_values[i] > threshold_accept:
            new_bboxes.append(roca_bboxes[max_indices[i]])
            use_roca.append(True)
            roca_objects.append(roca_img_info[max_indices[i]])
        else:
            augmented = augment_bbox(gt_bboxes[i],gt_infos,change_bbox_size_percent_img)
            new_bboxes.append(augmented.tolist())
            use_roca.append(False)
            roca_objects.append(None)

    vis_img = vis_all_bboxes(gt_bboxes,roca_bboxes,new_bboxes,img_path_vis,img_size_orig,use_roca)
    # vis_img = None

    indices_orig_objects = [gt_infos['objects'][i]['index'] for i in range(len(gt_infos['objects']))]

    return new_bboxes,use_roca,indices_orig_objects,roca_objects,vis_img

def get_max_indices_max_values_where_one_gt_only_one_roca(ious):
    

    ious = np.array(ious)

    indices_gt = []
    indices = []
    max_values_all = []

    # print(ious)
    for j in range(len(ious)):
        max_values = []
        for i in range(len(ious)):
            max_values.append(np.max(ious[i]))
        
        index_gt = np.argmax(max_values)

        indices_gt.append(index_gt)

        index_roca = np.argmax(ious[index_gt])
        indices.append(index_roca)
        max_values_all.append(max_values[index_gt])
        ious[:,index_roca] = ious[:,index_roca] * 0

        # needed ?
        ious[index_gt,:] = ious[index_gt,:] * 0

        if np.all(np.abs(ious) < 0.00001):
            break


    final_max_values_all = np.zeros(len(ious))
    final_max_indices_all = np.zeros(len(ious),dtype=int)

    final_max_values_all[indices_gt] = max_values_all
    final_max_indices_all[indices_gt] = indices
    # print('indices_gt',indices_gt)
    # print('indices',indices)
    # print('max_values_all',max_values_all)



    # max_values_all = (np.array(max_values_all)[indices_gt]).tolist()
    # max_indices_all = (np.array(indices)[indices_gt]).tolist()
    # print('max_values_all',max_values_all)
    # print('max_indices_all',max_indices_all)
    # print(S)
    return final_max_indices_all ,final_max_values_all

def vis_all_bboxes(gt_bboxes,roca_bboxes,new_bboxes,img_path,img_size_orig,use_roca):
    assert os.path.exists(img_path), img_path
    img = cv2.imread(img_path)
    img_2 = img.copy()
    img_size_loaded = [img.shape[1],img.shape[0]]

    resize = np.array([img_size_loaded[0]/img_size_orig[0],img_size_loaded[1]/img_size_orig[1]])
    resize = np.concatenate([resize,resize],axis=0)

    roca_used = []
    for i in range(len(use_roca)):
        if use_roca[i] == True:
            roca_used.append(new_bboxes[i])


    gt_bboxes = np.array(gt_bboxes) * resize
    roca_bboxes = np.array(roca_bboxes) * resize
    new_bboxes = np.array(new_bboxes) * resize
    

    img = draw_boxes(img,gt_bboxes,color=(0,255,0))
    img = draw_boxes(img,roca_bboxes,color=(0,0,255))
    img_2 = draw_boxes(img_2,new_bboxes,color=(0,255,0))
    # img_2 = draw_boxes(img_2,gt_bboxes,color=(0,0,255))


    if roca_used != []:
        roca_used = np.array(roca_used) * resize
        img_2 = draw_boxes(img_2,roca_used,color=(0,0,255))

    draw_text_block(img,['n gt: {}'.format(gt_bboxes.shape[0]),'n roca: {}'.format(roca_bboxes.shape[0])],top_left_corner=(20,20),font_scale=1,font_thickness=1)
    draw_text_block(img_2,['n roca used: {}'.format(len(roca_used))],top_left_corner=(20,20),font_scale=1,font_thickness=1)

    concated_img = cv2.hconcat([img,img_2])
    return concated_img




def main():
    dir_path_base = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val/'
    dir_path_gt = dir_path_base + 'gt_infos_center_in_image'
    img_dir_path = dir_path_base + 'images_480_360'
    out_dir_bboxes = dir_path_base + 'bboxes_roca_with_object_infos'
    out_dir_vis_boxes = dir_path_base + 'bboxes_roca_with_object_infos_vis'


    threshold_accept = 0.3
    change_bbox_size_percent_img = 0.1


    roca_path = '/scratch2/fml35/results/ROCA/per_frame_best_no_null.json'
    with open(roca_path, 'r') as f:
        roca = json.load(f)

    # roca = {"scene0663_01/color/000000.jpg": [{"score": 0.9960930943489075, "bbox": [318.9104309082031, 1.0792236328125, 479.2693786621094, 141.2074737548828], "t": [0.5776125192642212, -0.28210917115211487, 1.3746026754379272], "q": [0.06005612388253212, -0.29392731189727783, 0.19125893712043762, -0.9345694780349731], "s": [0.7281332612037659, 0.7198944687843323, 0.7434131503105164], "category": "display", "scene_cad_id": ["03211117", "830b29f6fe5d9e51542a2cfb40957ec8"]}]}

    for file in tqdm(sorted(os.listdir(dir_path_gt))):
    # for file in tqdm(sorted([dir_path_gt+'/scene0663_01-000000.json'])):
        with open(os.path.join(dir_path_gt, file), 'r') as f:
            gt_infos = json.load(f)

        bboxes,use_roca,indices_orig_objects,roca_objects,vis_img = roca_bboxes(roca,gt_infos,threshold_accept,change_bbox_size_percent_img,img_dir_path)
        if np.any(vis_img != None):
            cv2.imwrite(os.path.join(out_dir_vis_boxes,file.replace('.json','.png')),vis_img)
        with open(os.path.join(out_dir_bboxes,file), 'w') as f:
            json.dump({'bboxes':bboxes,'use_roca':use_roca,"indices_orig_objects":indices_orig_objects,"roca_objects":roca_objects},f)


if __name__ == '__main__':
    main()