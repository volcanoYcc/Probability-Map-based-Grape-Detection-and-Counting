import torch
import torch.optim as optim

import os
import sys

import json
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import random
from tqdm import tqdm
import time

from model_BC_mtl import BC_mtl
from data_dataset import MTLDataset
from test_vis import predict, get_gt, eval_BC
from config import dataset_config, dataset_config_test, train_config, evaluate_config, dev

def train_one_epoch(model, train_loader, epoch, optimizer, scheduler, outputfile, dev):
    model.train()
    loss = {'total_loss':[],'object_loss':[],'parts_loss':[]}

    with tqdm(train_loader) as tbar:
        for images, objectmap_true, partsmap_true, _, _, _ in tbar:
            tbar.set_description("epoch {}".format(epoch))

            # Set variables for training
            images = images.float().to(dev)
            objectmap_true = objectmap_true.float().to(dev)
            partsmap_true = partsmap_true.float().to(dev)

            # Zero the gradient
            optimizer.zero_grad()

            # Get model predictions, calculate loss
            training_output = model(images, train=True, ground_truth=[objectmap_true,partsmap_true])

            object_pred, parts_pred, object_loss, parts_loss, total_loss = training_output

            object_loss = object_loss.mean()
            parts_loss = parts_loss.mean()
            total_loss = total_loss.mean()

            loss['object_loss'].append(object_loss.item())
            loss['parts_loss'].append(parts_loss.item())
            loss['total_loss'].append(total_loss.item())

            total_loss.backward()
            optimizer.step()

            tbar.set_postfix(total="{:.4f}".format(np.mean(loss['total_loss'])),
                        object="{:.4f}".format(np.mean(loss['object_loss'])),
                        parts="{:.4f}".format(np.mean(loss['parts_loss'])),
                        lr="{:.4f}".format(optimizer.state_dict()['param_groups'][0]['lr']))

            del images, objectmap_true, partsmap_true

    text = 'epoch: ' + str(epoch) + ' tota_loss: ' + str(np.mean(loss['total_loss'])) + ' object_loss: ' + str(np.mean(loss['object_loss'])) + \
            ' parts_loss: ' + str(np.mean(loss['parts_loss'])) + ' lr: ' + str(optimizer.state_dict()['param_groups'][0]['lr'])
    print(text,file=outputfile)
    scheduler.step()

    return np.mean(loss['total_loss'])

def eval_one_epoch(model, test_loader, epoch, outputfile, dev, file_dir = None):
    model.eval()
    loss = {'total_loss':[],'object_loss':[],'parts_loss':[]}

    with tqdm(test_loader) as tbar:
        with torch.no_grad():
            for images, objectmap_true, partsmap_true, _, _, _ in tbar:
                tbar.set_description("evaluating...")

                # Set variables for testing
                images = images.float().to(dev)
                objectmap_true = objectmap_true.float().to(dev)
                partsmap_true = partsmap_true.float().to(dev)

                # Get model predictions, calculate loss
                testing_output = model(images, train=True, ground_truth=[objectmap_true,partsmap_true])

                object_pred, parts_pred, object_loss, parts_loss, total_loss = testing_output

                object_loss = object_loss.mean()
                parts_loss = parts_loss.mean()
                total_loss = total_loss.mean()

                loss['object_loss'].append(object_loss.item())
                loss['parts_loss'].append(parts_loss.item())
                loss['total_loss'].append(total_loss.item())

                tbar.set_postfix(total="{:.4f}".format(np.mean(loss['total_loss'])),
                            object="{:.4f}".format(np.mean(loss['object_loss'])),
                            parts="{:.4f}".format(np.mean(loss['parts_loss'])))

                del images, objectmap_true, partsmap_true
    if file_dir != None:
        if not os.path.exists(os.path.join(file_dir,'vis')):
            os.makedirs(os.path.join(file_dir,'vis'))
        cv2.imwrite(os.path.join(file_dir,'vis',str(epoch)+'_object.jpg'),object_pred.cpu().numpy()[0]*255)
        cv2.imwrite(os.path.join(file_dir,'vis',str(epoch)+'_parts.jpg'),parts_pred.permute(0, 2, 3, 1).cpu().numpy()[0]*255)

    text = 'evaluate: tota_loss: ' + str(np.mean(loss['total_loss'])) + ' object_loss: ' + str(np.mean(loss['object_loss'])) + \
            ' parts_loss: ' + str(np.mean(loss['parts_loss']))
    print(text,file=outputfile)

    return np.mean(loss['total_loss'])

def eval_one_epoch_ap(model, test_loader, epoch, outputfile, dev, evaluate_config, BC_dir, file_dir = None):
    model.eval()
    gt_json_path = os.path.join(file_dir, 'train_instances_gt.json')
    dr_json_path = os.path.join(file_dir, 'train_instances_dr.json')

    object_preds, parts_preds, BC_preds, _ = predict(model, test_loader, evaluate_config, dev, file_dir, epoch)
    object_gts, parts_gts, BC_gts = get_gt(test_loader, BC_dir)

    with open(gt_json_path, "w") as f:
        json.dump(object_gts, f, indent=4)
    with open(dr_json_path, "w") as f:
        json.dump(object_preds, f, indent=4)

    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(dr_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    precisions = coco_eval.eval['precision']
    for idx, catId in enumerate(coco_gt.getCatIds()):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]

        if precision.size:
            ap50_95 = np.mean(precision)
        else:
            ap50_95 = float('nan')

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

    text = 'evaluate_ap: ap IoU=0.50: ' + str(coco_eval.stats[1]) + ' ap IoU=0.50:0.95: ' + str(ap50_95) + \
            ' mae: ' + str(mae) + ' rmse: ' + str(rmse) + ' mrd: ' + str(mrd) + ' one_fvu: ' + str(one_fvu) + \
            ' matched_num: ' + str(len(all_est_count))
    print(text,file=outputfile)

    return float(coco_eval.stats[1]),mae,rmse,mrd

def save_checkpoint(model, info, name = 'test'):
    state = {
            'epoch': info['epoch'],
            'state_dict': model.state_dict(),
            }
    torch.save(state, os.path.join(info['base_root'], name+'.pth.tar'))

def load_checkpoint(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['state_dict'])
    epoch = state_dict['epoch']+1
    return model,epoch

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    seed_torch(train_config['seed'])

    base_dir = sys.path[0]
    train_json = os.path.join(base_dir,'dataset/WGISD/train_WGISD.json')
    test_json =  os.path.join(base_dir,'dataset/WGISD/train_WGISD.json')
    BC_dir = os.path.join(base_dir,'dataset/BC/val')
    load_weight =False

    with open(train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(test_json, 'r') as outfile:        
        test_list = json.load(outfile)

    dataset_config['base_dir'] = base_dir
    dataset_config_test['base_dir'] = base_dir
    train_dataset = MTLDataset(train_list, dataset_config = dataset_config)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=dataset_config['batch_size'], 
                                                shuffle=True, 
                                                num_workers=dataset_config['num_workers'], 
                                                )
    test_dataset = MTLDataset(test_list, dataset_config = dataset_config_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=dataset_config_test['batch_size'], 
                                                shuffle=False, 
                                                num_workers=dataset_config_test['num_workers'], 
                                                )

    model = BC_mtl().to(dev)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, train_config['lr'])
    if train_config['schedular'] == 'cosine':
        schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config['epoch'],eta_min=train_config['min_lr'])
    elif train_config['schedular'] == 'linear':
        lf = lambda x: (1 - x / train_config['epoch']) * (1.0 - train_config['lr']) + train_config['lr']  # linear
        schedular = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    if load_weight:
        weight_path = base_dir+'/checkpoint/testcheckpoint_last.pth.tar'
        model,e = load_checkpoint(model,weight_path)
        if train_config['start_epoch'] == None:
            train_config['start_epoch'] = e

    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    file_dir = os.path.join(base_dir,'runs','train',localtime)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    best_ap = 0
    best_mae = 1000
    best_mrd = 10
    best_epoch_ap = 0
    best_epoch_mae = 0
    best_epoch_mrd = 0

    for epoch in range(train_config['start_epoch'],train_config['epoch']):
        outputfile = open(os.path.join(file_dir,"log.txt"), 'a')
        train_loss = train_one_epoch(model, train_loader, epoch, optimizer, schedular, outputfile, dev)
        if epoch<=train_config['ap_epoch']:
            test_loss = eval_one_epoch(model, test_loader, epoch, outputfile, dev, file_dir)
            save_checkpoint(model,{'epoch':epoch,'base_root': file_dir},'last')
            print("=> loss: {:.4f}   test_loss: {:.4f}".format(train_loss, test_loss))
            outputfile.close()
        if epoch>train_config['ap_epoch']:
            ap,mae,rmse,mrd = eval_one_epoch_ap(model, test_loader, epoch, outputfile, dev, evaluate_config, BC_dir, file_dir)
            if ap > best_ap:
                best_ap = ap
                best_epoch_ap = epoch
                save_checkpoint(model,{'epoch':epoch,'base_root': file_dir},'best_ap')
            if mae < best_mae:
                best_mae = mae
                best_epoch_mae = epoch
                save_checkpoint(model,{'epoch':epoch,'base_root': file_dir},'best_mae')
            if mrd < best_mrd:
                best_mrd = mrd
                best_epoch_mrd = epoch
                save_checkpoint(model,{'epoch':epoch,'base_root': file_dir},'best_mrd')
            save_checkpoint(model,{'epoch':epoch,'base_root': file_dir},'last')
            print("=> loss: {:.4f}   bbox_ap: {:.4f}   mae: {:.4f}   rmse: {:.4f}   mrd: {:.4f}".format(train_loss, ap, mae, rmse, mrd))
            print('best_ap: ',best_ap,' best_epoch_ap: ',best_epoch_ap,
                ' best_mae: ',best_mae,' best_epoch_mae: ',best_epoch_mae,
                ' best_mrd: ',best_mrd,' best_epoch_mrd: ',best_epoch_mrd)
            outputfile.close()

    outputfile = open(os.path.join(file_dir,"log.txt"), 'a')
    final_text = 'best_ap: '+str(best_ap)+' best_epoch_ap: '+str(best_epoch_ap)+ \
                ' best_mae: '+str(best_mae)+' best_epoch_mae: '+str(best_epoch_mae)+ \
                ' best_mrd: '+str(best_mrd)+' best_epoch_mrd: '+str(best_epoch_mrd)
    print(final_text,file=outputfile)
    outputfile.close()
    