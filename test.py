#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import time

from libs.config.config import OPTION as opt
from libs.utils.Logger import TreeEvaluation as Evaluation, TimeRecord, LogTime, Tee, Loss_record
from libs.utils.Restore import get_save_dir,restore
from libs.dataset.YoutubeVOS import YTVOSDataset
from libs.dataset.transform import TestTransform
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from libs.utils.loss import *
from libs.utils.optimer import VIPMT_optimizer

from model.VIPMT import VIPMT
import torch.backends.cudnn as cudnn
import random

SNAPSHOT_DIR = opt.SNAPSHOT_DIR

def get_arguments():
    parser = argparse.ArgumentParser(description='FSVOS')
    parser.add_argument("--arch", type=str,default='VIPMT')
    parser.add_argument("--data_path", type=str,default=None)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--restore_epoch", type=int, default=0)
    parser.add_argument("--query_frame", type=int, default=5)
    parser.add_argument("--support_frame", type=int, default=5)
    parser.add_argument("--finetune_idx", type=int, default=1)

    parser.add_argument("--test", action='store_true')
    parser.add_argument("--test_best", default=True)
    parser.add_argument("--test_num", type=int, default=1)

    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--trainid", type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--fix_random_seed_test', type=bool, default=True)
    parser.add_argument('--test_seed', type=int, default=42)
    parser.add_argument("--initmodel_path", type=str, default=opt.initmodel_path)

    return parser.parse_args()


def get_mem(mask, pred, query_img, score, thresh=0.8):
    pred = pred.squeeze(0)
    mask = mask.squeeze(0).squeeze(1)
    score = score.squeeze(0)
    query_img_list = []
    pred_maps_list = []
    for i in range(mask.shape[0]):
        iou = score[i]
        if iou >= thresh:
            query_img_list.append(query_img[:, i, :, :, :].unsqueeze(1))
            pred_maps_list.append(pred[i, :, :].unsqueeze(0).unsqueeze(0).unsqueeze(2))

    if len(query_img_list) > 0:
        query_img_mem = torch.cat(query_img_list, dim=1)
        pred_maps_mem = torch.cat(pred_maps_list, dim=1)
    else:
        query_img_mem, pred_maps_mem = None, None

    return query_img_mem, pred_maps_mem


def test(args):
    # set manual seed
    if args.fix_random_seed_test and args.test_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.test_seed)
        np.random.seed(args.test_seed)
        torch.manual_seed(args.test_seed)
        torch.cuda.manual_seed_all(args.test_seed)
        random.seed(args.test_seed)

    # model & dataset
    model = VIPMT(args)
    model.eval()
    size = opt.test_size
    tsfm_test = TestTransform(size)
    finetune_idx = None
    test_dataset = YTVOSDataset(data_path=opt.data_path, train=False, query_frame=args.query_frame, support_frame=args.support_frame,
                                transforms=tsfm_test, set_index=args.group, finetune_idx=finetune_idx)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)
    test_list = test_dataset.get_class_list()
    model.cuda()

    print('test_group:', args.group, '  test_num:', len(test_dataloader))

    model = restore(args, model, test_best=True)
    print("Resume best model...")

    test_evaluations = Evaluation(class_list=test_list)

    for index, data in enumerate(test_dataloader):
        print(index)

        video_query_img, video_query_mask, support_img, support_mask, idx, vid = data
        support_img, support_mask = support_img.cuda(), support_mask.cuda()

        b, len_video, c, h, w = video_query_img.shape
        step_len = (len_video // args.query_frame)
        if len_video % args.query_frame != 0:
            step_len = step_len+1
        test_len = step_len

        query_img_list = []
        pred_maps_list = []
        video_pred_masks = []
        for i in range(test_len):
            if i == step_len - 1:
                query_img = video_query_img[:, i*args.query_frame:]
                query_mask = video_query_mask[:, i*args.query_frame:]
            else:
                query_img = video_query_img[:, i*args.query_frame:(i+1)*args.query_frame]
                query_mask = video_query_mask[:, i*args.query_frame:(i+1)*args.query_frame]
            query_img, query_mask,  idx \
                = query_img.cuda(), query_mask.cuda(),  idx.cuda()

            if len(query_img_list) != 0:
                query_hist = torch.cat(query_img_list, dim=1)
                query_mask_hist = torch.cat(pred_maps_list, dim=1)
                if query_hist.shape[1] < 5:
                    mem = query_hist
                    mem_mask = query_mask_hist
                else:
                    sap_idx = random.sample(range(query_hist.shape[1]), 5)
                    sap_idx.sort()
                    sap_idx = torch.tensor(sap_idx).cuda()
                    mem = query_hist.index_select(1, sap_idx)
                    mem_mask = query_mask_hist.index_select(1, sap_idx)
                support_img_input = torch.cat((support_img, mem), dim=1)
                support_mask_input = torch.cat((support_mask, mem_mask), dim=1)
            else:
                support_img_input = support_img
                support_mask_input = support_mask
            with torch.no_grad():
                pred_maps, score = model(query_img, query_mask, support_img_input, support_mask_input)
            query_img_mem, pred_maps_mem = get_mem(query_mask, pred_maps, query_img, score)
            if query_img_mem is not None:
                query_img_list.append(query_img_mem)
                pred_maps_list.append(pred_maps_mem)

            video_pred_masks.append(pred_maps)

            test_evaluations.update_evl(idx, query_mask.squeeze(2), pred_maps)

        video_pred_masks = torch.cat(video_pred_masks, dim=1)  # b t h w
        test_evaluations.update_vc7(idx, video_query_mask.squeeze(2), video_pred_masks)

    mean_f = np.mean(test_evaluations.f_score)
    str_mean_f = 'F: %.4f ' % (mean_f)
    mean_j = np.mean(test_evaluations.j_score)
    str_mean_j = 'J: %.4f ' % (mean_j)
    mean_vc7 = np.mean(test_evaluations.vc7)
    str_mean_vc7 = 'vc7: %.4f ' % (mean_vc7)

    f_list = ['%.4f' % n for n in test_evaluations.f_score]
    str_f_list = ' '.join(f_list)
    j_list = ['%.4f' % n for n in test_evaluations.j_score]
    str_j_list = ' '.join(j_list)
    vc7_list = ['%.4f' % n for n in test_evaluations.vc7]
    str_vc7_list = ' '.join(vc7_list)

    print(str_mean_f, str_f_list + '\n')
    print(str_mean_j, str_j_list + '\n')
    print(str_mean_vc7, str_vc7_list + '\n')

    return mean_f, mean_j, mean_vc7


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    args = get_arguments()
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    if not os.path.exists(get_save_dir(args)):
        os.makedirs(get_save_dir(args))
    args.snapshot_dir = get_save_dir(args)

    if args.test_best:
        logger = Tee(os.path.join(args.snapshot_dir, 'test_best_%d.txt' % args.test_num) , 'w')
    else:
        logger = Tee(os.path.join(args.snapshot_dir,'test_epoch_%d.txt' % args.restore_epoch),'w')

    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    print('Test Start')
    F, J, vc7 = test(args)
    print('F:', F)
    print('J:', J)
    print('vc7:', vc7)
