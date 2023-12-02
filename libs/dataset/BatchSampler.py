import torch
import numpy as np
from torch.utils.data import Dataset, BatchSampler, DistributedSampler
import random


class SameClassBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)
        self.dataset = sampler.data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.class_list = list(range(30))

    def __iter__(self):
        class_choose = []
        batch = []
        for idx in self.sampler:
            class_idx = idx // self.dataset.sample_per_class  # idx // 100, idx:(0, 30*100-1)
            if class_idx not in class_choose:
                batch.append(idx)
                class_choose.append(class_idx)
            else:
                class_modify = class_idx
                while class_modify in class_choose:
                    class_modify = random.sample(self.class_list, 1)[0]
                idx_modify = class_modify * self.dataset.sample_per_class
                batch.append(idx_modify)
                class_choose.append(class_modify)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                class_choose = []

    def __len__(self):  # 30 * 100
        return len(self.sampler)
