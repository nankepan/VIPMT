#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import torch
import numpy as np


def restore(args, model, test_best=False):
    if test_best:
        savedir = args.snapshot_dir
        filename = 'model_best.pth.tar'
        snapshot = os.path.join(savedir, filename)
    else:
        savedir = args.snapshot_dir
        filename = 'epoch_100.pth.tar'
        snapshot = os.path.join(savedir, filename)

    assert os.path.exists(snapshot), "Snapshot file %s does not exist." % (snapshot)

    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    print('Loaded weights from %s' % (snapshot))

    return model


def get_save_dir(args):
    snapshot_dir = os.path.join(args.snapshot_dir, args.arch, 'id_%d_group_%d_of_%d'%(args.trainid, args.group, args.num_folds))
    return snapshot_dir


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savedir = args.snapshot_dir
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savepath = os.path.join(savedir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(savedir, 'model_best.pth.tar'))
    remove_extra_checkpoints(savedir)


def remove_extra_checkpoints(checkpoint_dir_path):
    filenames = sorted([model for model in os.listdir(checkpoint_dir_path) if model.endswith('.pth.tar')])
    max_num_checkpoints = 3
    num_files_to_remove = max(0, len(filenames) - max_num_checkpoints)
    for filename in filenames[:num_files_to_remove]:
        os.remove(os.path.join(checkpoint_dir_path, filename))


def save_model(args, epoch, model, optimizer, best_iou, is_best=False):
    if epoch % args.save_epoch == 0 or is_best:
        save_checkpoint(args,
                        {'epoch': epoch, 'best_iou': best_iou,
                         'state_dict': model.state_dict()},
                        is_best=is_best,
                        filename='epoch_%03d.pth.tar' % (epoch))

