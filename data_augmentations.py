import random
import numpy as np
import cv2
import math
import torch
import torch.nn as nn

def bbox_remedy_on_corner(bbox,dot_target,one_bbox,size):
        xmin, ymin, xmax, ymax, label = one_bbox[0], one_bbox[1], one_bbox[2], one_bbox[3], one_bbox[4]
        temp_x = []
        temp_y = []
        for j in range(ymin,ymax):
            for k in range(xmin,xmax):
                if dot_target[j-1][k-1] == 1:
                    temp_x.append(k)
                    temp_y.append(j)
        if len(temp_y) == 0:#Remove clusters that contain too few berries
            return bbox
        else: 
            ymax_limit = min(max(temp_y)+15,size-1)
            ymin_limit = max(min(temp_y)-15,0)
            xmax_limit = min(max(temp_x)+15,size-1)
            xmin_limit = max(min(temp_x)-15,0)
            bbox.append([xmin_limit,ymin_limit,xmax_limit,ymax_limit,label])
            return bbox

def random_perspective(im,
                       probmap,
                       dotmap,
                       bbox,
                       degrees=10,
                       translate=.1,
                       scale=.1,
                       shear=10,
                       perspective=0.0,
                       border=(0, 0)):
    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    dotmap = nn.functional.max_pool2d(torch.from_numpy(dotmap).unsqueeze(0).to(torch.float32), 3, stride=1, padding=1).numpy()[0].astype('float32')

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
            probmap = cv2.warpPerspective(probmap, M, dsize=(width, height), borderValue=0)
            dotmap = cv2.warpPerspective(dotmap, M, dsize=(width, height), borderValue=0)
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            probmap = cv2.warpAffine(probmap, M[:2], dsize=(width, height), borderValue=0)
            dotmap = cv2.warpAffine(dotmap, M[:2], dsize=(width, height), borderValue=0)

    # Transform label coordinates
    n = len(bbox)
    if n:
        new = np.zeros((n, 4))
        xy = np.ones((n * 4, 3))
        xy[:, :2] = bbox[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=bbox[:, 0:4].T * s, box2=new.T, area_thr= 0.10)
        bbox = bbox[i]
        bbox[:, 0:4] = new[i]

    new_bboxs = []
    for i in range(len(bbox)):
        one_bbox = bbox[i]
        if 0 in one_bbox[:-1] or height in one_bbox[:-1]:
            new_bboxs = bbox_remedy_on_corner(new_bboxs,dotmap,one_bbox,height)
        else:
            new_bboxs.append(one_bbox)
    bbox = np.array(new_bboxs)

    return im, probmap, dotmap, bbox
    
def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

def letterbox(im, probmap, dotmap, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        probmap = cv2.resize(probmap, new_unpad, interpolation=cv2.INTER_LINEAR)
        dotmap = cv2.resize(dotmap, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    probmap = cv2.copyMakeBorder(probmap, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    dotmap = cv2.copyMakeBorder(dotmap, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return im, probmap, dotmap, ratio, (dw, dh)

def padding_label(x, w, h, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0]) + padw  # top left x
    y[:, 1] = h * (x[:, 1]) + padh  # top left y
    y[:, 2] = w * (x[:, 2]) + padw  # bottom right x
    y[:, 3] = h * (x[:, 3]) + padh  # bottom right y
    return y

class Albumentations:
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A
            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)]  # transforms
            self.transform = A.Compose(T)
        except ImportError:  # package not installed, skip
            pass
    def __call__(self, im, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im)  # transformed
            im= new['image']
        return im

def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed