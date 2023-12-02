#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pycocotools.ytvos import YTVOS
from torch.utils.data import Dataset
import os
import numpy as np
import random
from PIL import Image
import json
from tqdm import tqdm
#import cv2


class MiniVSPWDataset(Dataset):
    def __init__(self, data_path=None, train=True, valid=False,
                 set_index=1, finetune_idx=None,
                 support_frame=5, query_frame=1, sample_per_class=10,
                 transforms=None, another_transform=None, clip=1):
        self.train = train
        self.valid = valid
        self.set_index = set_index
        self.support_frame = support_frame
        self.query_frame = query_frame
        self.sample_per_class = sample_per_class
        self.transforms = transforms
        self.another_transform = another_transform
        self.clip = clip

        data_dir = os.path.join(data_path, 'VSPW_480p')
        self.img_dir = os.path.join(data_dir, 'data')
        self.ann_file_dir = os.path.join(data_dir, 'lists', 'nminivspw_pascal')

        self.data_list = self.load_filenames(self.img_dir)

        self.class_list = self.class_list_current

        if finetune_idx is not None:
            self.class_list = [self.class_list[finetune_idx]]

        self.video_ids = []

        for cls, vids in self.seqs_per_cls.items():
            vid_list = []
            for k, v in vids.items():
                vid_list.append(k)
            self.video_ids.append(vid_list)

        if not self.train:
            self.test_video_classes = []
            for i in range(len(self.class_list)):
                for j in range(len(self.video_ids[i])):
                    self.test_video_classes.append(i)

        if self.train:
            self.length = len(self.class_list) * sample_per_class
        else:
            self.length = len(self.test_video_classes)  # test

    def load_filenames(self, data_root):
        if self.train:
            data_list_file = 'train.txt'
            if self.valid:
                data_list_file = 'valid.txt'
        else:
            data_list_file = 'test.txt'
        data_list_path = data_list_file.replace('.txt', '_%d.txt' % (self.set_index-1))

        data_list_file = 'class_' + data_list_file
        clsfname = data_list_file.replace('.txt', '_%d.json' % (self.set_index-1))
        with open(os.path.join(self.ann_file_dir, clsfname), 'r') as f:
            classes = json.load(f)
        class_dic = {int(k): v for k, v in classes.items()}
        self.class_list_current = list(class_dic.keys())

        temp_data_list_path = ''
        temp_data_list_path += '.' + data_list_path.replace('txt', 'npy')
        temp_data_list_path = os.path.join(self.ann_file_dir, temp_data_list_path)

        if os.path.exists(temp_data_list_path):
            loaded_list = np.load(temp_data_list_path, allow_pickle=True).item()
            data_list = loaded_list['data_list']
            self.classes_per_seq = loaded_list['classes_per_seq']
            self.seqs_per_cls = loaded_list['seqs_per_cls']
        else:
            data_list = []
            self.classes_per_seq = {} # {seq: [cls1, cls2, ..], ..}
            self.seqs_per_cls = {cls: {} for cls in self.class_list_current} #{cls: [seq1, seq2, ..], ..}

            print("===> Processing Classes per Seq + Seqs per Cls")
            with open(os.path.join(self.ann_file_dir, data_list_path), 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    seq_name, fname = line.strip().split(' ')
                    mask = np.array(Image.open(os.path.join(data_root, fname)))

                    if seq_name not in self.classes_per_seq:
                        self.classes_per_seq[seq_name] = []

                    if seq_name not in data_list:
                        data_list.append(seq_name)

                    for cls in np.unique(mask):
                        if cls in self.class_list_current:
                            if cls not in self.classes_per_seq[seq_name]:
                                self.classes_per_seq[seq_name].append(cls)
                            if seq_name not in self.seqs_per_cls[cls]:
                                self.seqs_per_cls[cls][seq_name] = []
                            self.seqs_per_cls[cls][seq_name].append(fname)

            np.save(temp_data_list_path, {'data_list': data_list, 'seqs_per_cls': self.seqs_per_cls,
                                          'classes_per_seq': self.classes_per_seq})
        return data_list

    def get_GT_byclass(self, vid, class_id, frame_num=1, test=False, clip=1):
        seq_path = os.path.join(self.img_dir, vid, 'origin')
        frame_list = [os.path.join(seq_path, fname) for fname in sorted(os.listdir(seq_path))]
        frame_len = len(frame_list)

        if clip == 1:
            choice_frames = [random.sample(frame_list, 1)]
        else:
            metas = []
            for frame_id in range(5, frame_len, frame_num):  # 5 10 15 20 25 ...
                metas.append([frame_list[frame_id]])
            if len(metas) < clip:
                if len(metas) == 0:
                    metas.append([frame_list[0]])
                for i in range(clip-len(metas)):
                    metas.append(metas[-1])
            choice_frames = random.sample(metas, clip)
            choice_frames.sort()

        all_frames, all_masks = [], []
        for choice_frame in choice_frames:
            if test:
                frame_num = frame_len
            if frame_num > 1:
                if frame_num <= frame_len:
                    choice_idx = frame_list.index(choice_frame[0])
                    if choice_idx < frame_num:
                        begin_idx = 0
                        end_idx = frame_num
                    else:
                        begin_idx = choice_idx - frame_num + 1
                        end_idx = choice_idx + 1
                    choice_frame = [frame_list[n] for n in range(begin_idx, end_idx)]
                else:
                    choice_frame = []
                    for i in range(frame_num):
                        if i < frame_len:
                            choice_frame.append(frame_list[i])
                        else:
                            choice_frame.append(frame_list[frame_len - 1])
            frames = [np.array(Image.open(frame_dir)) for frame_dir in choice_frame]

            masks = []
            for image_path in choice_frame:
                mask = np.array(Image.open(image_path.replace('origin', 'mask').replace('jpg', 'png')))
                temp_mask = np.zeros_like(mask)
                temp_mask[mask == class_id] = 1
                masks.append(temp_mask)

            all_frames += frames
            all_masks += masks

        return all_frames, all_masks

    def __gettrainitem__(self, idx):
        list_id = idx // self.sample_per_class
        vid_set = self.video_ids[list_id]

        query_vid = random.sample(vid_set, 1)

        if len(vid_set) < self.support_frame:
            vid_set = vid_set * self.support_frame
        support_vid = random.sample(vid_set, self.support_frame)

        query_frames, query_masks = self.get_GT_byclass(query_vid[0], self.class_list[list_id], self.query_frame, clip=3)

        support_frames, support_masks = [], []
        for i in range(self.support_frame):
            one_frame, one_mask = self.get_GT_byclass(support_vid[i], self.class_list[list_id], 1)
            support_frames += one_frame
            support_masks += one_mask

        if self.transforms is not None:
            query_frames, query_masks = self.transforms(query_frames, query_masks)
            support_frames, support_masks = self.transforms(support_frames, support_masks)
        return query_frames, query_masks, support_frames, support_masks, self.class_list[list_id], query_vid, support_vid

    def __gettestitem__(self, idx):
        list_id = self.test_video_classes[idx]
        vid_set = self.video_ids[list_id]

        class_num = [len(self.video_ids[i]) for i in range(len(self.video_ids))]
        class_milestone = [sum(class_num[:i]) for i in range(len(class_num)+1)]
        current_id = idx - class_milestone[list_id]
        query_vid = vid_set[current_id]

        support_frames, support_masks = [], []
        support_vid = random.sample(vid_set, self.support_frame)
        while query_vid in support_vid:
            if len(vid_set) < (self.support_frame + 1):
                vid_set = vid_set * self.support_frame
            support_vid = random.sample(vid_set, self.support_frame)
        for i in range(self.support_frame):
            one_frame, one_mask = self.get_GT_byclass(support_vid[i], self.class_list[list_id], 1)
            support_frames += one_frame
            support_masks += one_mask

        query_frames, query_masks = self.get_GT_byclass(query_vid, self.class_list[list_id], test=True)

        if self.transforms is not None:
            query_frames, query_masks = self.transforms(query_frames, query_masks)
            if self.another_transform is not None:
                support_frames, support_masks = self.another_transform(support_frames, support_masks)
            else:
                support_frames, support_masks = self.transforms(support_frames, support_masks)
        vid_name = query_vid
        return query_frames, query_masks, support_frames, support_masks, self.class_list[list_id], vid_name

    def __getitem__(self, idx):
        if self.train:
            return self.__gettrainitem__(idx)
        else:
            return self.__gettestitem__(idx)

    def __len__(self):
        return self.length

    def get_class_list(self):
        return self.class_list
