#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import time

from libs.config.config import OPTION as opt
from libs.utils.Logger import TreeEvaluation as Evaluation, TimeRecord, LogTime, Tee, Loss_record
from libs.utils.Restore import get_save_dir, save_model
from libs.dataset.YoutubeVOS import YTVOSDataset
from libs.dataset.transform import TrainTransform, TestTransform
from torch.utils.data import DataLoader
from libs.utils.optimer import VIPMT_optimizer
import numpy as np
from libs.utils.loss import *

import torch.backends.cudnn as cudnn
import random
from model.VIPMT import VIPMT
from collections import OrderedDict
from libs.dataset.BatchSampler import SameClassBatchSampler

SNAPSHOT_DIR = opt.SNAPSHOT_DIR


def get_arguments():
    parser = argparse.ArgumentParser(description='FSVOS')
    parser.add_argument("--arch", type=str, default='VIPMT')
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--novalid", action='store_true')
    parser.add_argument("--max_iters", type=int, default=3000)
    parser.add_argument("--step_iter", type=int, default=100)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--restore_epoch", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--save_epoch", type=int, default=10)
    parser.add_argument("--sample_per_class", type=int, default=100)
    parser.add_argument("--vsample_per_class", type=int, default=50)
    parser.add_argument("--query_frame", type=int, default=5)
    parser.add_argument("--support_frame", type=int, default=5)

    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--trainid", type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--fix_random_seed_train', type=bool, default=True)
    parser.add_argument('--train_seed', type=int, default=42)
    parser.add_argument('--fix_random_seed_val', type=bool, default=True)
    parser.add_argument('--valid_seed', type=int, default=42)

    parser.add_argument("--initmodel_path", type=str, default=opt.initmodel_path)
    parser.add_argument('--train_clip', type=int, default=3)

    return parser.parse_args()


def train(args):
    # set manual seed
    if args.fix_random_seed_train and args.train_seed is not None:
        print('set train seed:', args.train_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.train_seed)
        np.random.seed(args.train_seed)
        torch.manual_seed(args.train_seed)
        torch.cuda.manual_seed_all(args.train_seed)
        random.seed(args.train_seed)

    # build model
    print('Building Model')
    net = VIPMT(args)
    print('Total model params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    optimizer = VIPMT_optimizer(net)
    net = net.cuda()

    # dataset
    print('Preparing dataset')
    size = opt.input_size
    tsfm_train = TrainTransform(size)
    tsfm_val = TestTransform(size)
    query_frame = args.query_frame
    support_frame = args.support_frame
    traindataset = YTVOSDataset(data_path=opt.data_path, query_frame=query_frame, support_frame=support_frame,
                                sample_per_class=args.sample_per_class,
                                transforms=tsfm_train, set_index=args.group, clip=args.train_clip)
    validdataset = YTVOSDataset(valid=True, data_path=opt.data_path, query_frame=query_frame,
                                support_frame=support_frame, sample_per_class=args.vsample_per_class,
                                transforms=tsfm_val, set_index=args.group)
    train_list = traindataset.get_class_list()
    valid_list = validdataset.get_class_list()

    sampler_train = torch.utils.data.RandomSampler(traindataset)
    batch_sampler_train = SameClassBatchSampler(sampler_train, args.batch_size, drop_last=True)
    train_loader = DataLoader(traindataset, batch_sampler=batch_sampler_train, num_workers=args.num_workers)

    val_loader = DataLoader(validdataset, batch_size=1, shuffle=False, num_workers=2)
    train_iters = len(train_loader)
    val_iters = len(val_loader)
    print('training iters per epoch: ', train_iters)
    print('valid iters per epoch: ', val_iters)

    # set evaluation
    print('Setting losses')
    losses = Loss_record()
    train_evaluations = Evaluation(class_list=train_list)
    valid_evaluations = Evaluation(class_list=valid_list)

    # set epoch
    start_epoch = args.restore_epoch
    best_iou = 0

    max_step = int(train_iters / args.step_iter)
    train_time_record = TimeRecord(max_step, args.max_epoch)
    trained_iter = train_iters * start_epoch
    args.max_iters = train_iters * args.max_epoch

    # train
    print('Start training')
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.max_epoch):
        if args.fix_random_seed_train and args.train_seed is not None:
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.cuda.manual_seed(args.train_seed + epoch)
            np.random.seed(args.train_seed + epoch)
            torch.manual_seed(args.train_seed + epoch)
            torch.cuda.manual_seed_all(args.train_seed + epoch)
            random.seed(args.train_seed + epoch)
        print('==> Training epoch {:d}'.format(epoch))
        begin_time = time.time()
        is_best = False

        net.train()
        for iter, data in enumerate(train_loader):
            trained_iter += 1

            if trained_iter % 50 == 0:
                print('optimizer lr:', optimizer.state_dict()['param_groups'][0]['lr'])

            query_img, query_mask, support_img, support_mask, idx, query_vid, support_vid = data
            query_img, query_mask, support_img, support_mask, idx \
                = query_img.cuda(), query_mask.cuda(), support_img.cuda(), support_mask.cuda(), idx.cuda()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                support_img_input = support_img
                support_mask_input = support_mask
                pred_map_clip1, main_loss_clip1, aux_loss_clip1, multivid_contrast_loss1, loss_score1, qry_bcb_fts_clip1, supp_bcb_fts_clip1 = \
                    net(query_img[:, :5, :, :, :].contiguous(), query_mask[:, :5, :, :, :].contiguous(),
                        support_img_input, support_mask_input)

                support_img_input_clip2 = OrderedDict()
                for k, v in qry_bcb_fts_clip1.items():
                    support_img_input_clip2[k] = torch.cat((supp_bcb_fts_clip1[k], v), dim=1)
                support_mask_input = torch.cat((support_mask, query_mask[:, :5, :, :, :].contiguous()), dim=1)
                pred_map_clip2, main_loss_clip2, aux_loss_clip2, multivid_contrast_loss2, loss_score2, qry_bcb_fts_clip2, supp_bcb_fts_clip2 = \
                    net(query_img[:, 5:10, :, :, :].contiguous(), query_mask[:, 5:10, :, :, :].contiguous(),
                        support_img_input_clip2, support_mask_input, clip=2)

                query_mask_hist = query_mask[:, :10, :, :, :].contiguous()
                sap_idx = random.sample(range(10), 5)
                sap_idx.sort()
                # print(sap_idx)
                sap_idx = torch.tensor(sap_idx).cuda()

                support_img_input_clip3 = OrderedDict()
                for k, v in qry_bcb_fts_clip1.items():
                    mem = torch.cat((qry_bcb_fts_clip1[k], qry_bcb_fts_clip2[k]), dim=1)
                    mem = mem.index_select(1, sap_idx)
                    support_img_input_clip3[k] = torch.cat((supp_bcb_fts_clip1[k], mem), dim=1)
                mem_mask = query_mask_hist.index_select(1, sap_idx)
                support_mask_input = torch.cat((support_mask, mem_mask), dim=1)
                pred_map_clip3, main_loss_clip3, aux_loss_clip3, multivid_contrast_loss3, loss_score3, qry_bcb_fts_clip3, supp_bcb_fts_clip3 = \
                    net(query_img[:, 10:, :, :, :].contiguous(), query_mask[:, 10:, :, :, :].contiguous(),
                        support_img_input_clip3, support_mask_input, clip=3)

                pred_map = torch.cat((pred_map_clip1, pred_map_clip2, pred_map_clip3), dim=1)

                main_loss = main_loss_clip1 + main_loss_clip2 + main_loss_clip3
                aux_loss = aux_loss_clip1 + aux_loss_clip2 + aux_loss_clip3
                multivid_contrast_loss = multivid_contrast_loss1 + multivid_contrast_loss2 + multivid_contrast_loss3
                loss_score = loss_score1 + loss_score2 + loss_score3
                total_loss = main_loss + aux_loss + multivid_contrast_loss + loss_score

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            query_mask = query_mask.squeeze(2)   # [b, 15, h, w]
            losses.updateloss(total_loss, main_loss, aux_loss, multivid_contrast_loss)
            train_evaluations.update_evl(idx, query_mask, pred_map)

            if iter % args.step_iter == 0 and iter > 0:
                step_time, remain_time = train_time_record.gettime(epoch, begin_time)
                iou_str = train_evaluations.logiou(epoch, iter)
                loss_str = losses.getloss(epoch, iter)
                print(loss_str, ' | ', iou_str, ' | ', 'Step: %.4f s \t Remain: %.4f h' % (step_time, remain_time))
                begin_time = time.time()
            print('epoch:', epoch, 'iter:', iter, 'total_loss:%.4f' % total_loss.item(),
                  'main_loss:%.4f' % main_loss.item(), 'aux_loss:%.4f' % aux_loss.item(),
                  'multivid_contrast_loss:%.4f' % multivid_contrast_loss.item(), 'loss_score:%.4f' % loss_score.item())

            # break  # debug

        # validation
        if not args.novalid:
            # set manual seed
            if args.fix_random_seed_val and args.valid_seed is not None:
                cudnn.benchmark = False
                cudnn.deterministic = True
                torch.cuda.manual_seed(args.valid_seed)
                np.random.seed(args.valid_seed)
                torch.manual_seed(args.valid_seed)
                torch.cuda.manual_seed_all(args.valid_seed)
                random.seed(args.valid_seed)
            print('Current epoch: ', epoch, ', start eval')
            net.eval()
            valid_step = len(val_loader)
            valid_time = LogTime()
            valid_time.t1()
            with torch.no_grad():
                for step, data in enumerate(val_loader):
                    #print(step)
                    query_img, query_mask, support_img, support_mask, idx, query_vid, support_vid = data
                    query_img, query_mask, support_img, support_mask, idx \
                        = query_img.cuda(), query_mask.cuda(), support_img.cuda(), support_mask.cuda(), idx.cuda()

                    support_img_input = support_img
                    support_mask_input = support_mask
                    pred_map_clip1, score1 = net(query_img[:, :5, :, :, :].contiguous(), query_mask[:, :5, :, :, :].contiguous(),
                                                 support_img_input, support_mask_input)

                    support_img_input = torch.cat((support_img, query_img[:, :5, :, :, :].contiguous()), dim=1)
                    support_mask_input = torch.cat((support_mask, query_mask[:, :5, :, :, :].contiguous()), dim=1)
                    pred_map_clip2, score2 = net(query_img[:, 5:10, :, :, :].contiguous(), query_mask[:, 5:10, :, :, :].contiguous(),
                                                 support_img_input, support_mask_input)

                    query_mask_hist = query_mask[:, :10, :, :, :].contiguous()
                    sap_idx = random.sample(range(10), 5)
                    sap_idx.sort()
                    # print(sap_idx)
                    sap_idx = torch.tensor(sap_idx).cuda()
                    mem = query_img[:, :10, :, :, :].contiguous().index_select(1, sap_idx)
                    mem_mask = query_mask_hist.index_select(1, sap_idx)
                    support_img_input = torch.cat((support_img, mem), dim=1)
                    support_mask_input = torch.cat((support_mask, mem_mask), dim=1)
                    pred_map_clip3, score3 = net(query_img[:, 10:, :, :, :].contiguous(), query_mask[:, 10:, :, :, :].contiguous(),
                                                 support_img_input, support_mask_input)
                    pred_map = torch.cat([pred_map_clip1, pred_map_clip2, pred_map_clip3], dim=1)
                    query_mask = query_mask.squeeze(2)
                    valid_evaluations.update_evl(idx, query_mask, pred_map)
            mean_iou = np.mean(valid_evaluations.iou_list)
            valid_time.t2()
            if best_iou < mean_iou:
                is_best = True
                best_iou = mean_iou
            iou_list = ['%.4f' % n for n in valid_evaluations.iou_list]
            strings_iou_list = ' '.join(iou_list)
            print('valid ', valid_evaluations.logiou(epoch, valid_step), ' ', strings_iou_list, ' | ',
                  'valid_time: %.4f s' % valid_time.getalltime(), 'is_best', is_best,
                  'current iou:', mean_iou, 'best iou:', best_iou)

        save_model(args, epoch, net, optimizer, best_iou, is_best)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    args = get_arguments()
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    if not os.path.exists(get_save_dir(args)):
        os.makedirs(get_save_dir(args))
    args.snapshot_dir = get_save_dir(args)

    logger = Tee(os.path.join(args.snapshot_dir, 'train_log.txt'), 'w')

    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    train(args)
