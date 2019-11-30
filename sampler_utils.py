# -*- coding: utf-8 -*-

import random
import numpy as np
import torch.utils.data
from utils.basic_utils import *
from collections import OrderedDict
from torch._six import int_classes as _int_classes


class CustomRandomSampler(torch.utils.data.Sampler):

    def __init__(self, weights, num_samples, easy_ratio=0.5, drop_ratio = 0.05, replacement=True):
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.float)
        self.num_samples = num_samples
        self.replacement = replacement
        self.drop_ratio = drop_ratio
        self.easy_ratio = easy_ratio

    def __iter__(self):
        idx = torch.argsort(self.weights, descending=True).tolist()
        weight_list = self.weights.tolist()
        hard_num = len([i for i in weight_list if i > 0])
        easy_num = self.num_samples - hard_num
        drop_num = int(hard_num * self.drop_ratio)
        if (hard_num - drop_num + int(easy_num * self.easy_ratio)) % 2 != 0:
            hard_num += 1
        hard_list = idx[drop_num: hard_num]
        easy_list = idx[hard_num: self.num_samples]
        random.shuffle(easy_list)
        resample_list = hard_list + easy_list[0: int(easy_num * self.easy_ratio)]
        random.shuffle(resample_list)
        return iter(resample_list)

    def __len__(self):
        weight_list = self.weights.tolist()
        hard_num = len([i for i in weight_list if i > 0])
        easy_num = self.num_samples - hard_num
        drop_num = int(hard_num * self.drop_ratio)
        if (hard_num - drop_num + int(easy_num * self.easy_ratio)) % 2 != 0:
            hard_num += 1
        return (hard_num - drop_num + int(easy_num * self.easy_ratio))


class ClassBalanceSampler(torch.utils.data.Sampler):
    def __init__(self, input_id_to_name_map, input_class_dict, input_batch_size):

        self.class_sample_index_dict = {}
        self.class_sample_num_dict = {}
        self.class_id_num_order_dict = {}

        self.id_to_name_map = {}
        for cur_id in input_id_to_name_map:
            if isinstance(cur_id, int):
                cur_class_id = cur_id
            elif isinstance(cur_id, str):
                cur_class_id = int(cur_id)
            else:
                cur_class_id = cur_id

            self.id_to_name_map[cur_class_id] = input_id_to_name_map[cur_id]
            self.class_sample_index_dict[cur_class_id] = 0

        self.class_data_dict = input_class_dict
        assert self.class_data_dict is not None, 'input_class_dict is None!!!'
        self.init_class_count_dict()
        self.init_per_class_sample_num(input_batch_size)
        self.epoch_iteration = self.init_epoch_iterations()
        self.sample_length = self.epoch_iteration * input_batch_size

    def init_class_count_dict(self):
        class_num_dict = {}
        for cur_class_id in self.class_data_dict:
            class_num_dict[cur_class_id] = len(self.class_data_dict[cur_class_id])
            assert class_num_dict[cur_class_id] > 0, \
                '[init_class_count_dict]Class id:{}\tClass Name:{}\tClass num:{}'.format(
                cur_class_id, self.id_to_name_map[cur_class_id], class_num_dict[cur_class_id])

        self.class_id_num_order_dict = OrderedDict(sorted(class_num_dict.items(), key=lambda t: t[1]))

    def init_per_class_sample_num(self, input_batch_size):

        assert input_batch_size >= len(self.class_id_num_order_dict), \
            'Batch Size:{} should larger than class num:{}'.format(input_batch_size, len(self.class_id_num_order_dict))

        total_num = float(sum(self.class_id_num_order_dict.values()))
        class_rate_dict = {}
        for cur_id in self.class_id_num_order_dict:
            class_rate_dict[cur_id] = self.class_id_num_order_dict[cur_id] / total_num

        left_batch_space = input_batch_size
        cur_left_rate = 1.0
        for cur_id in self.class_id_num_order_dict:
            cur_class_sample_num = int(round(class_rate_dict[cur_id] / cur_left_rate * left_batch_space))
            self.class_sample_num_dict[cur_id] = max(1, cur_class_sample_num)
            left_batch_space -= self.class_sample_num_dict[cur_id]
            cur_left_rate -= class_rate_dict[cur_id]

        for cur_class_id in self.class_id_num_order_dict:
            print '[classBalanceSampler] Class Name:{0:20}, Id:{1:<4} Class Num:{2:<8} Batch Sample Num:{3:<3}'.format(
                self.id_to_name_map[cur_class_id], cur_class_id, self.class_id_num_order_dict[cur_class_id],
                self.class_sample_num_dict[cur_class_id])

        sample_batch_size = sum(self.class_sample_num_dict.values())
        assert sample_batch_size == input_batch_size, \
            'sample batch size:{} should equal to batch size:{}'.format(sample_batch_size, input_batch_size)

    def init_epoch_iterations(self):

        epoch_max_iteration = 0
        for cur_class_id in self.class_id_num_order_dict:
            cur_class_num = self.class_id_num_order_dict[cur_class_id]
            cur_class_batch_sample_num = self.class_sample_num_dict[cur_class_id]
            cur_iteration = cur_class_num / cur_class_batch_sample_num
            epoch_max_iteration = max(epoch_max_iteration, cur_iteration)

        return int(epoch_max_iteration)

    def __iter__(self):

        sample_index_list = []
        for cur_temp_batch_index in range(self.epoch_iteration):
            for cur_class_id in self.class_id_num_order_dict:
                cur_class_num = self.class_id_num_order_dict[cur_class_id]
                cur_class_sample_num = self.class_sample_num_dict[cur_class_id]
                cur_class_sample_index = self.class_sample_index_dict[cur_class_id]

                for cur_temp_index in range(cur_class_sample_num):
                    if cur_class_sample_index >= cur_class_num:
                        cur_class_sample_index = 0
                        random.shuffle(self.class_data_dict[cur_class_id])

                    cur_sample_index = self.class_data_dict[cur_class_id][cur_class_sample_index]
                    sample_index_list.append(cur_sample_index)
                    cur_class_sample_index += 1

                self.class_sample_index_dict[cur_class_id] = cur_class_sample_index

        return iter(sample_index_list)

    def __len__(self):
        return self.sample_length


if __name__ == '__main__':
    print 'toDo'
