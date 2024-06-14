from torch.utils.data import Dataset
import torch

import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import cv2
import numpy as np
import h5py
import json
import math
import xml.etree.ElementTree as ET

from config import dataset_config, dataset_config_test
from data_augmentations import Albumentations, random_perspective, letterbox, padding_label, augment_hsv
from data_utils import image_resize, preprocess_input, gaussian_radius, draw_gaussian

class MTLDataset(Dataset):
    def __init__(self, root, dataset_config = {}, train = False):
        print('Preparing dataset...')

        #load dataset config
        self.dataset_config = dataset_config
        
        #generate dataset config
        self.dataset_config['mosaic_border'] = [-dataset_config['img_size'] // 2, -dataset_config['img_size'] // 2]
        root = root*self.dataset_config['root_mut']
        #random.shuffle(root)
        self.dataset_config['nSamples'] = len(root)
        self.dataset_config['roots'] = root
        self.dataset_config['train'] = train
        bi = np.floor(np.arange(self.dataset_config['nSamples']) / self.dataset_config['batch_size']).astype(np.int64)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.dataset_config['batch'] = bi  # batch index of image

        #albumentation method define
        self.albumentations = Albumentations() if self.dataset_config['augment'] else None

        print('Done')

    def __len__(self):
        return self.dataset_config['nSamples']

    def __getitem__(self, index):
        if self.dataset_config['mosaic']:
            img, probmap, dotmap, bbox, ori_shape = self.load_mosaic(index)
        else:
            img, probmap, dotmap, bbox, ori_shape, (h, w) = self.load(index)

            shape = self.dataset_config['img_size']  # final letterboxed shape
            img, probmap, dotmap, ratio, pad = letterbox(img, probmap, dotmap, (shape*0.75,shape), auto=False, scaleup=self.dataset_config['augment'])
            if len(bbox):
                bbox[:, :-1] = padding_label(bbox[:, :-1], ratio[0], ratio[1], padw=pad[0], padh=pad[1])
        
        if self.dataset_config['augment']:
            # Albumentations
            img = self.albumentations(img)

            # HSV color-space
            augment_hsv(img, hgain=self.dataset_config['hsv_h'], sgain=self.dataset_config['hsv_s'], vgain=self.dataset_config['hsv_v'])

            # Flip up-down
            if random.random() < self.dataset_config['flipud']:
                img = np.flipud(img)
                probmap = np.flipud(probmap)
                dotmap = np.flipud(dotmap)
                if len(bbox):
                    bbox[:, 1] = self.dataset_config['img_size'] - bbox[:, 1]
                    bbox[:, 3] = self.dataset_config['img_size'] - bbox[:, 3]
                    bbox = bbox[:,[0,3,2,1,4]]

            # Flip left-right
            if random.random() < self.dataset_config['fliplr']:
                img = np.fliplr(img)
                probmap = np.fliplr(probmap)
                dotmap = np.fliplr(dotmap)
                if len(bbox):
                    bbox[:, 0] = self.dataset_config['img_size'] - bbox[:, 0]
                    bbox[:, 2] = self.dataset_config['img_size'] - bbox[:, 2]
                    bbox = bbox[:,[2,1,0,3,4]]

        # Convert     
        h,w = img.shape[0],img.shape[1]
        if not self.dataset_config['for_vis']:
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB

        if self.dataset_config['normalize']:
            img = img/255
        
        # change data type
        label_new = []
        bbox_new = []
        if len(bbox):
            for one_bbox in bbox:
                label_new.append(int(one_bbox[4]))
                bbox_new.append([int(one_bbox[0]),int(one_bbox[1]),int(one_bbox[2]),int(one_bbox[3])])
        
        batch_hm = np.zeros((h, w, 1), dtype=np.float32)

        for i in range(len(bbox)):
            x1, y1, x2, y2, cls_id = bbox[i]
            h, w = y2 - y1, x2 - x1
            if h > 0 and w > 0:
                if self.dataset_config['gaussian_type'] == 'rectangle':
                    radius = (math.ceil(w/self.dataset_config['probmap_radius']),math.ceil(h/self.dataset_config['probmap_radius']))
                elif self.dataset_config['gaussian_type'] == 'square':
                    r = math.ceil(((h+w)/2)/self.dataset_config['probmap_radius'])
                    radius = (r,r)

                # Calculates the feature points of the real box
                ct = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                #print(ct_int)

                # Get gaussian heat map
                batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)

        return np.ascontiguousarray(img), batch_hm,  np.ascontiguousarray(probmap), np.ascontiguousarray(dotmap), np.array(bbox_new), ori_shape

    #Used to maintain the consistency of cluster and berry label and remove cluster labels that do not contain berries
    def bbox_remedy(self, dot_target, bbox):
        bbox_new = []
        for i in range(len(bbox)):
            box = bbox[i]
            xmin, ymin, xmax, ymax = box[0], box[1], box [2], box[3]
            temp_x = []
            temp_y = []
            for j in range(ymin,ymax):
                for k in range(xmin,xmax):
                    if dot_target[j-1][k-1] == 1:
                        temp_x.append(k)
                        temp_y.append(j)
            if len(temp_y) == 0:#Remove clusters that contain too few berries
                continue
            else: 
                bbox_new.append(box)
        return bbox_new

    def bbox_remedy_on_corner(self,bbox,dot_target,one_bbox,size):
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
            ymax_limit = min(max(temp_y)+int(12*((size/2)/1280)),size-1)
            ymin_limit = max(min(temp_y)-int(12*((size/2)/1280)),0)
            xmax_limit = min(max(temp_x)+int(12*((size/2)/1280)),size-1)
            xmin_limit = max(min(temp_x)-int(12*((size/2)/1280)),0)
            bbox.append([xmin_limit,ymin_limit,xmax_limit,ymax_limit,label])
            return bbox
    
    def load(self, index):
        img_path = self.dataset_config['roots'][index]
        img_path = os.path.join(self.dataset_config['base_dir'],img_path)
        prob_path = img_path.replace('.jpg','_probmap.h5')
        dot_path = img_path.replace('.jpg','_dotmap.h5')

        im = cv2.imread(img_path)
        prob_file = h5py.File(prob_path, 'r')
        prob_target = np.asarray(prob_file['density'])
        dot_file = h5py.File(dot_path, 'r')
        dot_target = np.asarray(dot_file['density'])
        VOC_BBOX_LABEL_NAMES = ('cluster')
        bbox_path = img_path.replace('.jpg','.xml')
        bbox = list()
        anno = ET.parse(bbox_path)
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')
            bbox.append([int(bndbox_anno.find(tag).text) for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            name = obj.find('name').text.lower().strip()
            bbox[-1].append(VOC_BBOX_LABEL_NAMES.index(name))
            
        bbox = self.bbox_remedy(dot_target, bbox)#Remove clusters that contain too few berries
        
        h0, w0 = im.shape[:2]
        r = self.dataset_config['img_size'] / max(h0, w0)  # ratio
        if r != 1:
            interp = cv2.INTER_LINEAR if (self.dataset_config['augment'] or r > 1) else cv2.INTER_AREA
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)

            prob_max = prob_target.max()
            prob_target = cv2.resize(prob_target, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_CUBIC)
            prob_target = prob_target * prob_max / float(prob_target.max() + 1e-6)

            shape = prob_target.shape
            dot_target_new = np.zeros(shape)
            for i in range(h0):
                for j in range(w0):
                    if dot_target[i][j] == 1:
                        dot_target_new[int(i*r)-1][int(j*r)-1] = 1
            dot_target = dot_target_new

            bbox_new = []
            for i in range(len(bbox)):
                box = bbox[i]
                xmin, ymin, xmax, ymax, label = int(box[0]*r), int(box[1]*r), int(box [2]*r), int(box[3]*r), box[4]
                bbox_new.append([xmin, ymin, xmax, ymax, label])
            bbox = bbox_new
        im, prob_target, dot_target, bbox = np.array(im), np.array(prob_target), np.array(dot_target), np.array(bbox)
        return im, prob_target, dot_target, bbox, [h0,w0], [int(h0*r), int(w0*r)]

    #Mosaic data augment method adopt from yolov5
    def load_mosaic(self, index):
        bbox4 = []
        s = self.dataset_config['img_size']
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.dataset_config['mosaic_border'])
        indices = [index] + random.choices(range(self.dataset_config['nSamples']), k=3)
        random.shuffle(indices)
        for i, index in enumerate(indices):
            img, probmap, dotmap, bbox, ori_shape, alt_shape = self.load(index)
            h,w = alt_shape[0],alt_shape[1]

            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                probmap4 = np.full((s * 2, s * 2), 0.0, dtype = np.float64)  # base probmap with 4 tiles
                dotmap4 = np.full((s * 2, s * 2), 0, dtype = np.uint8)  # base dotmap with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            probmap4[y1a:y2a, x1a:x2a] = probmap[y1b:y2b, x1b:x2b]
            dotmap4[y1a:y2a, x1a:x2a] = dotmap[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            for j in range(len(bbox)):
                one_bbox = bbox[j]
                xmin, ymin, xmax, ymax, label = one_bbox[0], one_bbox[1], one_bbox[2], one_bbox[3], one_bbox[4]
                if xmax <= x1b or ymax <= y1b or xmin >= x2b or ymin >= y2b:
                    continue
                elif xmin >= x1b and ymin >= y1b and xmax <= x2b and ymax <= y2b:
                    bbox4.append([xmin+padw, ymin+padh, xmax+padw, ymax+padh, label])
                else:
                    bbox4 = self.bbox_remedy_on_corner(bbox4,dotmap4,[max(xmin,x1b)+padw, max(ymin,y1b)+padh, min(xmax,x2b)+padw, min(ymax,y2b)+padh, label],s * 2)

        img4, probmap4, dotmap4, bbox4 = np.array(img4), np.array(probmap4), np.array(dotmap4), np.array(bbox4)
        img4, probmap4, dotmap4, bbox4 = random_perspective(img4,probmap4,dotmap4,bbox4,
            self.dataset_config['degrees'],self.dataset_config['translate'],self.dataset_config['scale'],
            self.dataset_config['shear'],self.dataset_config['perspective'],self.dataset_config['mosaic_border'])

        return img4, probmap4, dotmap4, bbox4, ori_shape

#test code
if __name__ == '__main__':
    train = True
    base_dir = sys.path[0]
    json_file = os.path.join(base_dir,'dataset/WGISD/test_WGISD.json')
    if train:
        dataset_config['base_dir'] = base_dir
        dataset_config['for_vis'] = True
        dataset_config['normalize'] = False
        with open(json_file, 'r') as outfile:  
            file_list = json.load(outfile)
        dataset = MTLDataset(file_list, dataset_config = dataset_config, train = True)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_config['batch_size'], shuffle=False, num_workers=1)
        for i, (img, batch_hm, probmap, _, _, _) in enumerate(train_loader):
            print(img.shape,batch_hm.shape,probmap.shape)
            for j in range(len(img)):
                img_to_draw, batch_hm_to_draw, probmap_to_draw = img[j].numpy(),batch_hm[j].numpy(), probmap[j].numpy()
                cv2.imwrite(os.path.join(base_dir,'runs','dataset')+'/img_'+str(i)+'_'+str(j)+'.jpg',img_to_draw)
                cv2.imwrite(os.path.join(base_dir,'runs','dataset')+'/batch_hm_'+str(i)+'_'+str(j)+'.jpg',batch_hm_to_draw*255)
                cv2.imwrite(os.path.join(base_dir,'runs','dataset')+'/probmap_'+str(i)+'_'+str(j)+'.jpg',probmap_to_draw*255)
            if i == 2:
                break
    else:
        dataset_config_test['base_dir'] = base_dir
        dataset_config_test['for_vis'] = True
        dataset_config_test['normalize'] = False
        with open(json_file, 'r') as outfile:  
            file_list = json.load(outfile)
        dataset = MTLDataset(file_list, dataset_config = dataset_config_test, train = True)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_config_test['batch_size'], shuffle=False, num_workers=1)
        for i, (img, batch_hm, probmap, _, _, _) in enumerate(test_loader):
            print(img.shape,batch_hm.shape,probmap.shape)
            for j in range(len(img)):
                img_to_draw, batch_hm_to_draw, probmap_to_draw = img[j].numpy(),batch_hm[j].numpy(), probmap[j].numpy()
                cv2.imwrite(os.path.join(base_dir,'runs','dataset')+'/img_'+str(i)+'_'+str(j)+'.jpg',img_to_draw)
                cv2.imwrite(os.path.join(base_dir,'runs','dataset')+'/batch_hm_'+str(i)+'_'+str(j)+'.jpg',batch_hm_to_draw*255)
                cv2.imwrite(os.path.join(base_dir,'runs','dataset')+'/probmap_'+str(i)+'_'+str(j)+'.jpg',probmap_to_draw*255)
            if i == 2:
                break
