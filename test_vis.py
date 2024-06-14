import torch
import torch.nn as nn
import torchvision

import os
import sys
import time

import json
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import random
from tqdm import tqdm

from model_BC_mtl import BC_mtl
from data_dataset import MTLDataset
from config import dataset_config_test, evaluate_config, dev

def faster_up_hill_pred(object_pred,parts_pred,evaluate_config):
    threshold_parts = evaluate_config['threshold_parts']
    step = evaluate_config['step']
    filter = evaluate_config['filter']
    threshold_filter = evaluate_config['threshold_filter']
    
    H,W = object_pred.shape[2:]

    avg_pooled_parts_probmap = nn.functional.avg_pool2d(parts_pred[0], 3, stride=1, padding=1)
    max_pooled_parts_probmap = nn.functional.max_pool2d(avg_pooled_parts_probmap, 3, stride=1, padding=1)
    candidate_parts_peak_unfiltered = torch.where(avg_pooled_parts_probmap==max_pooled_parts_probmap, avg_pooled_parts_probmap, torch.full_like(parts_pred[0], 0))
    candidate_parts_peak_unfiltered = torch.where(candidate_parts_peak_unfiltered>=threshold_parts, torch.full_like(parts_pred[0], 1), torch.full_like(parts_pred[0], 0))
    parts_points_indices_unfiltered = torch.nonzero(torch.reshape(candidate_parts_peak_unfiltered[0],(W*H,))).squeeze(1).cpu().numpy()
    parts_points_unfiltered = torch.nonzero(candidate_parts_peak_unfiltered[0]).cpu().numpy()
    
    avg_pooled_object_probmap = nn.functional.avg_pool2d(object_pred[0], 3, stride=1, padding=1)
    if filter:
        candidate_parts_peak = torch.where(avg_pooled_object_probmap>=threshold_filter, candidate_parts_peak_unfiltered, torch.full_like(parts_pred[0], 0))
        parts_points_indices = torch.nonzero(torch.reshape(candidate_parts_peak[0],(W*H,))).squeeze(1).cpu().numpy()
        parts_points = torch.nonzero(candidate_parts_peak[0]).cpu().numpy()
    else:
        parts_points_indices = parts_points_indices_unfiltered
        parts_points = parts_points_unfiltered

    max_pooled_object_indices = []
    for one_step in step:
        max_pooled_object_probmap,indices = nn.functional.max_pool2d(avg_pooled_object_probmap, 2*one_step+1, stride=1, padding=one_step, return_indices=True)
        max_pooled_object_indices.append(torch.reshape(indices,(1,W*H))[0].cpu().numpy())
    avg_pooled_object_probmap = avg_pooled_object_probmap[0].cpu().numpy()

    len_step = len(step)
    divide_parts = {}
    divide_tags = []
    divide_results = []
    moved_value = []
    for i in range(parts_points_indices.shape[0]):
        epoch = 0
        while True:
            object_indices = max_pooled_object_indices[epoch if epoch < len_step else len_step-1]
            ori_indice = parts_points_indices[i]
            moved_indice = object_indices[ori_indice]
            parts_points_indices[i] = moved_indice
            if epoch >= len_step-1 and object_indices[moved_indice] == moved_indice:
                moved_point = [moved_indice//W,moved_indice%W]
                if moved_point not in divide_tags:
                    divide_tags.append(moved_point)
                    divide_parts[str(moved_point)] = []
                    moved_value.append(avg_pooled_object_probmap[moved_point[0],moved_point[1]])
                divide_parts[str(moved_point)].append(parts_points[i].tolist())
                break
            else:
                epoch+=1

    for i in range(len(divide_tags)):
        divide_results.append(divide_parts[str(divide_tags[i])])
    
    return divide_tags, divide_results, parts_points, moved_value

def get_bbox(divide_results,H,W):
    bbox_preds = []
    for i in range(len(divide_results)):
        y_min,y_max,x_min,x_max = 0,0,0,0
        for j in range(len(divide_results[i])):
            if j == 0:
                y_min,y_max = divide_results[i][j][0],divide_results[i][j][0]
                x_min,x_max = divide_results[i][j][1],divide_results[i][j][1]
            y_min,y_max = min(y_min,divide_results[i][j][0]), max(y_max,divide_results[i][j][0])
            x_min,x_max = min(x_min,divide_results[i][j][1]), max(x_max,divide_results[i][j][1])
        bbox_preds.append([max(0,x_min-5),max(0,y_min-5),min(W-1,x_max+5),min(H-1,y_max+5)])#xmin,ymin,xmax,ymax
    return bbox_preds

def get_bbox_with_map(divide_results,H,W,parts_pred,evaluate_config):
    score_or_multiple = evaluate_config['score_or_multiple']
    score_alter = evaluate_config['score_alter']
    multiple_alter = evaluate_config['multiple_alter']

    bbox_preds = []
    avg_pooled_parts_probmap = nn.functional.avg_pool2d(parts_pred[0], 3, stride=1, padding=1)[0].cpu().numpy()
    for i in range(len(divide_results)):
        y_min_c,y_max_c,x_min_c,x_max_c = [],[],[],[]
        y_min_v,y_max_v,x_min_v,x_max_v = 0,0,0,0
        y_min_a,y_max_a,x_min_a,x_max_a = 0,0,0,0
        for j in range(len(divide_results[i])):
            if j == 0:
                y_min_c,y_max_c = divide_results[i][j].copy(),divide_results[i][j].copy()
                x_min_c,x_max_c = divide_results[i][j].copy(),divide_results[i][j].copy()
                y_min_v,y_max_v = avg_pooled_parts_probmap[y_min_c[0]][y_min_c[1]],avg_pooled_parts_probmap[y_max_c[0]][y_max_c[1]]
                x_min_v,x_max_v = avg_pooled_parts_probmap[x_min_c[0]][x_min_c[1]],avg_pooled_parts_probmap[x_max_c[0]][x_max_c[1]]
            else:
                if y_min_c[0]>divide_results[i][j][0]:
                    y_min_c = divide_results[i][j].copy()
                    y_min_v = avg_pooled_parts_probmap[y_min_c[0]][y_min_c[1]]
                if y_max_c[0]<divide_results[i][j][0]:
                    y_max_c = divide_results[i][j].copy()
                    y_max_v = avg_pooled_parts_probmap[y_max_c[0]][y_max_c[1]]
                if x_min_c[1]>divide_results[i][j][1]:
                    x_min_c = divide_results[i][j].copy()
                    x_min_v = avg_pooled_parts_probmap[x_min_c[0]][x_min_c[1]]
                if x_max_c[1]<divide_results[i][j][1]:
                    x_max_c = divide_results[i][j].copy()
                    x_max_v = avg_pooled_parts_probmap[x_max_c[0]][x_max_c[1]]
        if score_or_multiple == 'score':
            while True:
                if y_min_c[0]-y_min_a == 0 or score_alter >= avg_pooled_parts_probmap[y_min_c[0]-y_min_a][y_min_c[1]]:
                    break
                if avg_pooled_parts_probmap[y_min_c[0]-y_min_a-1][y_min_c[1]] >= avg_pooled_parts_probmap[y_min_c[0]-y_min_a][y_min_c[1]]:
                    break
                y_min_a+=1
            while True:
                if y_max_c[0]+y_max_a == H-1 or score_alter >= avg_pooled_parts_probmap[y_max_c[0]+y_max_a][y_max_c[1]]:
                    break
                if avg_pooled_parts_probmap[y_max_c[0]+y_max_a+1][y_max_c[1]] >= avg_pooled_parts_probmap[y_max_c[0]+y_max_a][y_max_c[1]]:
                    break
                y_max_a+=1
            while True:
                if x_min_c[1]-x_min_a == 0 or score_alter >= multiple_alter*avg_pooled_parts_probmap[x_min_c[0]][x_min_c[1]-x_min_a]:
                    break
                if avg_pooled_parts_probmap[x_min_c[0]][x_min_c[1]-x_min_a-1] >= avg_pooled_parts_probmap[x_min_c[0]][x_min_c[1]-x_min_a]:
                    break
                x_min_a+=1
            while True:
                if x_max_c[1]+x_max_a == W-1 or score_alter >= avg_pooled_parts_probmap[x_max_c[0]][x_max_c[1]+x_max_a]:
                    break
                if avg_pooled_parts_probmap[x_max_c[0]][x_max_c[1]+x_max_a+1] >= avg_pooled_parts_probmap[x_max_c[0]][x_max_c[1]+x_max_a]:
                    break
                x_max_a+=1
        elif score_or_multiple == 'multiple':
            while True:
                if y_min_c[0]-y_min_a == 0 or y_min_v >= multiple_alter*avg_pooled_parts_probmap[y_min_c[0]-y_min_a][y_min_c[1]]:
                    break
                if avg_pooled_parts_probmap[y_min_c[0]-y_min_a-1][y_min_c[1]] >= avg_pooled_parts_probmap[y_min_c[0]-y_min_a][y_min_c[1]]:
                    break
                y_min_a+=1
            while True:
                if y_max_c[0]+y_max_a == H-1 or y_max_v >= multiple_alter*avg_pooled_parts_probmap[y_max_c[0]+y_max_a][y_max_c[1]]:
                    break
                if avg_pooled_parts_probmap[y_max_c[0]+y_max_a+1][y_max_c[1]] >= avg_pooled_parts_probmap[y_max_c[0]+y_max_a][y_max_c[1]]:
                    break
                y_max_a+=1
            while True:
                if x_min_c[1]-x_min_a == 0 or x_min_v >= multiple_alter*avg_pooled_parts_probmap[x_min_c[0]][x_min_c[1]-x_min_a]:
                    break
                if avg_pooled_parts_probmap[x_min_c[0]][x_min_c[1]-x_min_a-1] >= avg_pooled_parts_probmap[x_min_c[0]][x_min_c[1]-x_min_a]:
                    break
                x_min_a+=1
            while True:
                if x_max_c[1]+x_max_a == W-1 or x_max_v >= multiple_alter*avg_pooled_parts_probmap[x_max_c[0]][x_max_c[1]+x_max_a]:
                    break
                if avg_pooled_parts_probmap[x_max_c[0]][x_max_c[1]+x_max_a+1] >= avg_pooled_parts_probmap[x_max_c[0]][x_max_c[1]+x_max_a]:
                    break
                x_max_a+=1
        
        bbox_preds.append([max(0,x_min_c[1]-x_min_a),max(0,y_min_c[0]-y_min_a),min(W-1,x_max_c[1]+x_max_a),min(H-1,y_max_c[0]+y_max_a)])#xmin,ymin,xmax,ymax
    return bbox_preds

def predict(model, test_loader, evaluate_config, dev, file_dir=None, epoch='test'):
    model.eval()
    object_result = []
    parts_result = []
    divide_results_all = []

    pred_time = [0,0]

    with tqdm(test_loader) as tbar:
        i = 0
        for images, _, _, _, _, _ in tbar:
            tbar.set_description("testing...")
            images = images.float().to(dev)
            time_start = time.time()
            with torch.no_grad():
                testing_output = model(images, train=False)
            time_end = time.time()
            pred_time[0]+=(time_end-time_start)
            time_start = time.time()
            object_pred, parts_pred = testing_output
            divide_tags, divide_results, parts_points, moved_value = faster_up_hill_pred(object_pred,parts_pred,evaluate_config)
            bbox_preds = get_bbox_with_map(divide_results,images.shape[2],images.shape[3],parts_pred,evaluate_config)

            left_parts_points = []
            total_num = len(divide_tags)
            for j in range(len(divide_tags)):
                jj = total_num-1-j
                if len(divide_results[jj]) <= evaluate_config['min_parts_num'] or moved_value[jj] <= evaluate_config['filter_object_score']:
                    del divide_tags[jj]
                    del divide_results[jj]
                    del moved_value[jj]
                    del bbox_preds[jj]
                else:
                    left_parts_points += divide_results[jj]
            divide_results_all.append(divide_results)

            time_end = time.time()
            pred_time[1]+=(time_end-time_start)

            for bbox, score in zip(bbox_preds, moved_value):
                xmin, ymin, xmax, ymax = bbox
                result = {}
                result["image_id"] = str(i)
                result["category_id"] = 1
                result["bbox"] = [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
                result["score"] = float(score)
                object_result.append(result)
            parts_result.append(parts_points)

            if file_dir != None:
                if not os.path.exists(os.path.join(file_dir,'vis')):
                    os.makedirs(os.path.join(file_dir,'vis'))
                if 'img' in evaluate_config['vis_type']:
                    img = (images[0].permute(1,2,0).cpu().numpy()*255).astype(int).copy()
                    b,g,r = cv2.split(img)
                    img = cv2.merge((r,g,b))
                    for j in range(len(divide_tags)):
                        color = [random.randint(0, 255) for _ in range(3)]
                        for k in range(len(divide_results[j])):
                            cv2.circle(img, (int(divide_results[j][k][1]),int(divide_results[j][k][0])), 3, color, -1)
                        cv2.rectangle(img, (bbox_preds[j][0],bbox_preds[j][1]), (bbox_preds[j][2],bbox_preds[j][3]), color, 2)
                        cv2.circle(img, (int(divide_tags[j][1]),int(divide_tags[j][0])), 5, (0,0,255), -1)
                    cv2.imwrite(os.path.join(file_dir,'vis',str(i)+'.jpg'),img)
                if 'obj' in evaluate_config['vis_type']:
                    cv2.imwrite(os.path.join(file_dir,'vis',str(i)+'_object.jpg'),object_pred.permute(0, 2, 3, 1).cpu().numpy()[0]*255)
                if 'part' in evaluate_config['vis_type']:
                    cv2.imwrite(os.path.join(file_dir,'vis',str(i)+'_parts.jpg'),parts_pred.permute(0, 2, 3, 1).cpu().numpy()[0]*255)
            i+=1
        if file_dir != None and 'train' in evaluate_config['vis_type']:
            if not os.path.exists(os.path.join(file_dir,'vis')):
                os.makedirs(os.path.join(file_dir,'vis'))
            img = (images[0].permute(1,2,0).cpu().numpy()*255).astype(int).copy()
            b,g,r = cv2.split(img)
            img = cv2.merge((r,g,b))
            for j in range(len(divide_tags)):
                color = [random.randint(0, 255) for _ in range(3)]
                for k in range(len(divide_results[j])):
                    cv2.circle(img, (int(divide_results[j][k][1]),int(divide_results[j][k][0])), 3, color, -1)
                cv2.rectangle(img, (bbox_preds[j][0],bbox_preds[j][1]), (bbox_preds[j][2],bbox_preds[j][3]), color, 2)
                cv2.circle(img, (int(divide_tags[j][1]),int(divide_tags[j][0])), 5, (0,0,255), -1)
            cv2.imwrite(os.path.join(file_dir,'vis',str(epoch)+'.jpg'),img)
            cv2.imwrite(os.path.join(file_dir,'vis',str(epoch)+'_object.jpg'),object_pred.permute(0, 2, 3, 1).cpu().numpy()[0]*255)
            cv2.imwrite(os.path.join(file_dir,'vis',str(epoch)+'_parts.jpg'),parts_pred.permute(0, 2, 3, 1).cpu().numpy()[0]*255)
    
    return object_result, parts_result, divide_results_all, pred_time

def get_gt(test_loader, BC_dir):
    tbar = tqdm(test_loader)
    i = 0
    j = 0
    ground_truth = {}
    point_truth = []
    BC_truth = []
    categories_info = []
    images_info = []
    annotations_info = []

    category = {}
    category['supercategory'] = 'cluster'
    category['name'] = 'cluster'
    category['id'] = 1
    categories_info.append(category)

    for images, objectmap_true, partsmap_true, dotmap_true, bbox, ori_shape in tbar:
        image_info = {}

        image_info['file_name'] = str(i) + '.jpg'
        image_info['width'] = ori_shape[1].item()
        image_info['height'] = ori_shape[0].item()
        image_info['id'] = str(i)
        images_info.append(image_info)

        for one_bbox in bbox.numpy()[0]:
            annotation = {}
            iscrowd = 0

            xmin = one_bbox[0]
            ymin = one_bbox[1]
            width = one_bbox[2]-one_bbox[0]
            height = one_bbox[3]-one_bbox[1]

            annotation['area'] = int(width * height)
            annotation['category_id'] = int(1)
            annotation['image_id'] = str(i)
            annotation['iscrowd'] = iscrowd
            annotation['bbox'] = [int(xmin), int(ymin), int(width), int(height)]
            annotation['id'] = str(j)
            annotations_info.append(annotation)
            j += 1
        i+=1

        true_points = []
        nz = np.nonzero(dotmap_true.numpy()[0])
        for l in range(len(nz[0])):
            p = [(nz[1][l]),(nz[0][l])]
            true_points.append(p)
        point_truth.append(true_points)

    ground_truth['images'] = images_info
    ground_truth['categories'] = categories_info
    ground_truth['annotations'] = annotations_info

    BC_files = os.listdir(BC_dir)
    for name in BC_files:
        with open(os.path.join(BC_dir,name),'r',encoding='utf8')as fp:
            json_data = json.loads(json.load(fp))
        BC_truth.append(json_data)
    return ground_truth, point_truth, BC_truth

def load_checkpoint(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['state_dict'])
    return model

def MRD(gt,est):
    temp = 0
    for i in range(len(gt)):
        temp += abs(est[i]-gt[i])/gt[i]
    return temp/len(gt)

def ONE_FVU(gt,est):
    temp1 = 0
    temp2 = 0
    mean_gt = sum(gt)/len(gt)
    for i in range(len(gt)):
        temp1 += (gt[i]-est[i])*(gt[i]-est[i])
        temp2 += (gt[i]-mean_gt)*(gt[i]-mean_gt)
    
    return 1 - temp1/temp2

def eval_BC(test_loader, BC_preds, BC_gts, object_preds, object_gts):
    dataset_config_test = test_loader.dataset.dataset_config

    file_names_all = []
    file_names = []
    for name in test_loader.dataset.dataset_config['roots']:
        file_names_all.append(name.split('/')[-1])
    for ann in BC_gts:
        file_names.append(ann['name'])

    all_gt_count = []
    all_est_count = []
    for i in range(len(file_names)):
        gt_parts = []
        gt_objects = []
        est_parts = []
        est_objects = []

        match_id = file_names_all.index(file_names[i])
        h0,w0 = object_gts['images'][match_id]['height'],object_gts['images'][match_id]['width']
        h,w = int(dataset_config_test['img_size']*0.75),int(dataset_config_test['img_size'])
        factor = w/w0
        h1,w1 =  int(h0*factor),w
        alter = (h-h1)/2

        est_parts = BC_preds[match_id]
        for obj in BC_gts[i]['annotation']:
            gt_objects.append(obj['bbox'])
            gt_parts.append(obj['points'])
        for obj in object_preds:
            if obj['image_id'] == str(match_id):
                est_objects.append([int(obj['bbox'][0]/factor),
                                int((obj['bbox'][1]-alter)/factor),
                                int((obj['bbox'][0]+obj['bbox'][2])/factor),
                                int((obj['bbox'][1]+obj['bbox'][3]-alter)/factor)])

        iou_result = torchvision.ops.box_iou(torch.tensor(gt_objects),torch.tensor(est_objects)).tolist()

        num_match = 0
        for j in range(len(iou_result)):
            max_iou = max(iou_result[j])
            if max_iou >= 0.5:
                num_match+=1
                all_est_count.append(len(est_parts[iou_result[j].index(max_iou)]))
                all_gt_count.append(len(gt_parts[j]))

    mrd = MRD(all_gt_count,all_est_count)
    one_fvu = ONE_FVU(all_gt_count,all_est_count)
    return mrd, one_fvu, all_gt_count, all_est_count

def F_1(ap,ar):
    f_1 = 2*ap*ar/(ap+ar)
    return f_1

def count_object(preds,gts):
    mae = 0
    rmse = 0
    preds_num = [0]
    gts_num = [0]
    pred_bboxs = [[]]
    gt_bboxs = [[]]
    for pred in preds:
        if int(pred['image_id'])>=len(preds_num):
            preds_num.append(1)
            pred_bboxs.append([])
            bbox = pred['bbox']
            pred_bboxs[int(pred['image_id'])].append([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])
        else:
            preds_num[int(pred['image_id'])]+=1
            bbox = pred['bbox']
            pred_bboxs[int(pred['image_id'])].append([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])
    for gt in gts:
        if int(gt['image_id'])>=len(gts_num):
            gts_num.append(1)
            gt_bboxs.append([])
            bbox = gt['bbox']
            gt_bboxs[int(gt['image_id'])].append([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])
        else:
            gts_num[int(gt['image_id'])]+=1
            bbox = gt['bbox']
            gt_bboxs[int(gt['image_id'])].append([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])

    for i in range(len(preds_num)):
        et_count = preds_num[i]
        gt_count = gts_num[i]

        mae += abs(gt_count-et_count)
        rmse += ((gt_count-et_count)*(gt_count-et_count))

    mae = mae/len(preds_num)
    rmse = np.sqrt(rmse/(len(preds_num)))

    num_match = 0
    for i in range(len(gt_bboxs)):
        iou_result = torchvision.ops.box_iou(torch.tensor(gt_bboxs[i]),torch.tensor(pred_bboxs[i])).tolist()
        for j in range(len(iou_result)):
            max_iou = max(iou_result[j])
            if max_iou >= 0.5:
                num_match+=1

    return mae,rmse,num_match

if __name__ == '__main__':
    base_dir = sys.path[0]
    file_dir = os.path.join(base_dir,'runs','test')
    test_json = os.path.join(base_dir,'dataset/WGISD/test_WGISD.json')
    BC_dir = os.path.join(base_dir,'dataset/BC/test')
    weight_path = os.path.join(base_dir,'runs/train/1280/best_ap.pth.tar')

    model = BC_mtl().to(dev)
    model = load_checkpoint(model,weight_path)

    with open(test_json, 'r') as outfile:        
        test_list = json.load(outfile)

    dataset_config_test['base_dir'] = base_dir
    test_dataset = MTLDataset(test_list, dataset_config = dataset_config_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=dataset_config_test['batch_size'], 
                                                shuffle=False, 
                                                num_workers=dataset_config_test['num_workers'], 
                                                )
    
    object_preds, parts_preds, BC_preds, pred_time = predict(model, test_loader, evaluate_config, dev, file_dir)
    object_gts, parts_gts, BC_gts = get_gt(test_loader, BC_dir)

    gt_json_path = os.path.join(file_dir, 'instances_gt.json')
    dr_json_path = os.path.join(file_dir, 'instances_dr.json')

    with open(gt_json_path, "w") as f:
        json.dump(object_gts, f, indent=4)
    with open(dr_json_path, "w") as f:
        json.dump(object_preds, f, indent=4)

    print('object evaluation results:')
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(dr_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print(' * F_1 {f1:.3f} '.format(f1=F_1(coco_eval.stats[0],coco_eval.stats[8])))
    obj_mae,obj_rmse,match_num = count_object(object_preds,object_gts['annotations'])
    print(' * MAE {mae:.3f}--RMSE {rmse:.3f}--MATCH_NUM {match_num:d} '.format(mae=obj_mae,rmse=obj_rmse,match_num=match_num))
    print('parts evaluation results:')

    mae = 0
    rmse = 0
    gts = []
    ests = []
    for i in range(len(parts_preds)):
        et_count = len(parts_preds[i])
        gt_count = len(parts_gts[i])

        gts.append(gt_count)
        ests.append(et_count)

        mae += abs(gt_count-et_count)
        rmse += ((gt_count-et_count)*(gt_count-et_count))

    mae = mae/len(parts_preds)
    rmse = np.sqrt(rmse/(len(parts_preds)))

    mrd, one_fvu, all_gt_count, all_est_count = eval_BC(test_loader, BC_preds, BC_gts, object_preds, object_gts)

    print(gts)
    print(ests)
    print(' * MAE {mae:.3f}--RMSE {rmse:.3f} '.format(mae=mae,rmse=rmse))
    print(' * MRD {mrd:.3f}--ONE_FVU {one_fvu:.3f}--MATCH_NUM {match_num:d} '.format(mrd=mrd,one_fvu=one_fvu,match_num=len(all_est_count)))
    print('mean_pred_time: ',str(pred_time[0]/len(test_loader)),str(pred_time[1]/len(test_loader)))