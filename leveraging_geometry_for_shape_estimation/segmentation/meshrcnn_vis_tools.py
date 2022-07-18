# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import itertools
import numpy as np
import os
import pycocotools.mask as mask_util
from detectron2.data import detection_utils
from detectron2.structures import Boxes, BoxMode
from detectron2.utils.colormap import colormap
from pytorch3d.io import save_obj
from pytorch3d.ops import sample_points_from_meshes
from termcolor import colored

from tabulate import tabulate

try:
    import cv2  # noqa
except ImportError:
    # If opencv is not available, everything else should still run
    pass


_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)


def draw_text(img, pos, text, font_scale=0.35,color_bg=_GREEN,color_font=_GRAY,font=cv2.FONT_HERSHEY_SIMPLEX,thickness=None):
    """
    Draw text on an image.

    Args:
        pos (tuple): x, y; the position of the text
        text (str):
        font_scale (float):
    """
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, 1)
    # Place text background.
    if x0 + text_w > img.shape[1]:
        x0 = img.shape[1] - text_w
    if y0 - int(1.2 * text_h) < 0:
        y0 = int(1.2 * text_h)
    back_topleft = x0, y0 - int(1.3 * text_h)
    back_bottomright = x0 + text_w, y0
    cv2.rectangle(img, back_topleft, back_bottomright, color_bg, -1)
    # Show text.
    text_bottomleft = x0, y0 - int(0.2 * text_h)
    if thickness != None:
        cv2.putText(img, text, text_bottomleft, font, font_scale, color_font, lineType=cv2.LINE_AA,thickness=thickness)
    else:
        cv2.putText(img, text, text_bottomleft, font, font_scale, color_font, lineType=cv2.LINE_AA)
    return img


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


def draw_mask(img, mask, color, alpha=0.4, draw_contours=False):
    """
    Draw (overlay) a mask on an image.

    Args:
        mask (ndarray): an (H, W) array of the same spatial size as the image.
            Nonzero positions in the array are considered part of the mask.
        color: a BGR color
        alpha (float): blending efficient. Smaller values lead to more transparent masks.
        draw_contours (bool): whether to also draw the contours of every
            connected component (object part) in the mask.
    """
    img = img.astype(np.float32)

    idx = np.nonzero(mask)
    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * color

    if draw_contours:
        # opencv func signature has changed between versions
        contours = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2]
        cv2.drawContours(img, contours, -1, _WHITE, 1, cv2.LINE_AA)
    return img.astype(np.uint8)

def draw_segmentation_prediction(img,mask,color,boxes,text,font_scale=0.6):

    img = draw_mask(img,mask,color)
    img = draw_boxes(img,boxes)
    img = draw_text(img,(boxes[0,0],boxes[0,1] - 2), text,font_scale=font_scale)
    return img