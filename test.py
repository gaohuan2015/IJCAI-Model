import torch.multiprocessing as mp
# coding:utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import os
import math
import redis
import json
import multiprocessing as mp
import argparse
import time
import os
from tqdm import trange
import Model_modified

dtype_LongTensor_cpu = torch.LongTensor
dtype_LongTensor_cuda = torch.cuda.LongTensor
dtype_FloatTensor_cpu = torch.FloatTensor
dtype_FloatTensor_cuda = torch.cuda.FloatTensor


def test_frag(args, num_entity, num_relation, data_test_frag, data_all, rel_type, process_id):
    # load model parameters from file
    # print('Loading model...')
    model = Model_modified.Model(num_entity, num_relation, args)
    model.load_state_dict(torch.load(os.path.join(args.save_path, args.file)))
    model.cuda()

    rank_head_raw = 0
    rank_head_filter = 0
    rank_rel_raw = 0
    rank_rel_filter = 0
    rank_tail_raw = 0
    rank_tail_filter = 0
    hit10_head_raw = 0
    hit10_head_filter = 0
    hit1_rel_raw = 0
    hit1_rel_filter = 0
    hit10_rel_raw = 0
    hit10_rel_filter = 0
    hit10_tail_raw = 0
    hit10_tail_filter = 0

    hit10_head_1_1 = 0
    hit10_head_1_n = 0
    hit10_head_n_1 = 0
    hit10_head_n_n = 0
    hit10_tail_1_1 = 0
    hit10_tail_1_n = 0
    hit10_tail_n_1 = 0
    hit10_tail_n_n = 0
    # result_dict = {}
    for i in range(len(data_test_frag)):
        time_start = time.time()
        triple = data_test_frag[i]
        h = triple[0]
        r = triple[1]
        t = triple[2]
        score_dict = {}
        # iterate over all the entities to substitute current head
        start_index = 0
        while (start_index < num_entity - 1):
            if start_index + args.size_batch < num_entity - 1:
                end_index = start_index + args.size_batch
            else:
                end_index = num_entity - 1
            h_batch = Variable(torch.LongTensor(range(start_index, end_index))).type(dtype_LongTensor_cuda)
            r_batch = Variable(torch.LongTensor([r] * (end_index - start_index))).type(dtype_LongTensor_cuda)
            t_batch = Variable(torch.LongTensor([t] * (end_index - start_index))).type(dtype_LongTensor_cuda)
            score_list = model.score_function(h_batch, r_batch, t_batch)
            for (index, score) in zip(range(start_index, end_index), score_list):
                score_dict[index] = score.data[0]
            start_index += args.size_batch
        rank_raw = 0
        rank_filter = 0
        sorted_list = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
        # print('score: {}'.format(score_dict[h]))
        # print('rel_type: {}'.format(rel_type[r]))
        # print(sorted_list[0][1])
        # print((sorted_list[0][0], r, t) in data_all)
        # result_dict[','.join([str(s) for s in triple]) + ':0'] = sorted_list # [(pair[0], pair[1].data[0]) for pair in score_list]
        for pair in sorted_list:
            rank_raw += 1
            if (pair[0], r, t) not in data_all:
                rank_filter += 1
            if pair[0] == h:
                rank_filter += 1
                rank_head_raw += rank_raw
                rank_head_filter += rank_filter
                if rank_raw <= 10:
                    hit10_head_raw += 1
                if rank_filter <= 10:
                    hit10_head_filter += 1
                    if rel_type[r] == 0:
                        hit10_head_1_1 += 1
                    elif rel_type[r] == 1:
                        hit10_head_1_n += 1
                    elif rel_type[r] == 2:
                        hit10_head_n_1 += 1
                    elif rel_type[r] == 3:
                        hit10_head_n_n += 1
                break

        # iterate over all the entities to substitute current tail
        score_dict.clear()
        start_index = 0
        while (start_index < num_entity - 1):
            if start_index + args.size_batch < num_entity - 1:
                end_index = start_index + args.size_batch
            else:
                end_index = num_entity - 1
            h_batch = Variable(torch.LongTensor([h] * (end_index - start_index))).type(dtype_LongTensor_cuda)
            r_batch = Variable(torch.LongTensor([r] * (end_index - start_index))).type(dtype_LongTensor_cuda)
            t_batch = Variable(torch.LongTensor(range(start_index, end_index))).type(dtype_LongTensor_cuda)
            score_list = model.score_function(h_batch, r_batch, t_batch)
            for (index, score) in zip(range(start_index, end_index), score_list):
                score_dict[index] = score.data[0]
            start_index += args.size_batch
        rank_raw = 0
        rank_filter = 0
        sorted_list = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
        # result_dict[','.join([str(s) for s in triple]) + ':1'] = sorted_list # [(pair[0], pair[1].data[0]) for pair in score_list]
        for pair in sorted_list:
            rank_raw += 1
            if (h, r, pair[0]) not in data_all:
                rank_filter += 1
            if pair[0] == t:
                rank_filter += 1
                rank_tail_raw += rank_raw
                rank_tail_filter += rank_filter
                if rank_raw <= 10:
                    hit10_tail_raw += 1
                if rank_filter <= 10:
                    hit10_tail_filter += 1
                    if rel_type[r] == 0:
                        hit10_tail_1_1 += 1
                    elif rel_type[r] == 1:
                        hit10_tail_1_n += 1
                    elif rel_type[r] == 2:
                        hit10_tail_n_1 += 1
                    elif rel_type[r] == 3:
                        hit10_tail_n_n += 1
                break

        # predicting relation
        score_dict.clear()
        start_index = 1  # ignore the NULL relation
        while (start_index < num_relation):
            if start_index + args.size_batch < num_relation:
                end_index = start_index + args.size_batch
            else:
                end_index = num_relation
            h_batch = Variable(torch.LongTensor([h] * (end_index - start_index))).type(dtype_LongTensor_cuda)
            r_batch = Variable(torch.LongTensor(range(start_index, end_index))).type(dtype_LongTensor_cuda)
            t_batch = Variable(torch.LongTensor([t] * (end_index - start_index))).type(dtype_LongTensor_cuda)
            score_list = model.score_function(h_batch, r_batch, t_batch)
            for (index, score) in zip(range(start_index, end_index), score_list):
                score_dict[index] = score.data[0]
            start_index += args.size_batch
        rank_raw = 0
        rank_filter = 0
        sorted_list = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
        # result_dict[','.join([str(s) for s in triple]) + ':2'] = sorted_list # [(pair[0], pair[1].data[0]) for pair in score_list]
        for pair in sorted_list:
            rank_raw += 1
            if (h, pair[0], t) not in data_all:
                rank_filter += 1
            if pair[0] == r:
                rank_filter += 1
                rank_rel_raw += rank_raw
                rank_rel_filter += rank_filter
                if rank_raw == 1:
                    hit1_rel_raw += 1
                if rank_filter == 1:
                    hit1_rel_filter += 1
                if rank_raw <= 10:
                    hit10_rel_raw += 1
                if rank_filter <= 10:
                    hit10_rel_filter += 1
                break

        time_end = time.time()
        print('process_id: {}\tprogress: {}/{}\ttime used: {}s'.format(process_id, i + 1, len(data_test_frag), time_end-time_start))
        # with open('result.txt', 'a') as f:
        #     f.write(json.dumps(result_dict))
        #     f.write('\n')

    return [rank_head_raw, rank_head_filter, rank_tail_raw, rank_tail_filter, hit10_head_raw, hit10_head_filter,
            hit10_tail_raw, hit10_tail_filter, hit10_head_1_1, hit10_head_1_n,
            hit10_head_n_1, hit10_head_n_n, hit10_tail_1_1, hit10_tail_1_n, hit10_tail_n_1, hit10_tail_n_n,
            rank_rel_raw, rank_rel_filter, hit1_rel_raw, hit1_rel_filter, hit10_rel_raw, hit10_rel_filter]


def main():
    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument('--data-path', type=str, default='data/fb15k')
    parser.add_argument('--save-path', type=str, default='model_save/fb15k')
    parser.add_argument('--dimension', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--num-epoch', type=int, default=1000)
    parser.add_argument('--size-batch', type=int, default=1000)
    parser.add_argument('--num-process', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--init', type=bool, default=False)  # True - init embeddings using pre-trained embeddings
    parser.add_argument('--file', type=str, default=None)  # saved model file
    parser.add_argument('--size-frag', type=int, default=None)  # size of a fragment for test
    parser.add_argument('--p', type=int, default=1)  # the norm of score

    args = parser.parse_args()

    with open(os.path.join(args.data_path, 'encode/entity_id.txt')) as f:
        entity2id = {line.split('\t')[0].strip(): line.split('\t')[1].strip() for line in f.readlines()}
    with open(os.path.join(args.data_path, 'encode/relation_id.txt')) as f:
        relation2id = {line.split('\t')[0].strip(): line.split('\t')[1].strip() for line in f.readlines()}
    id2entity = {value: key for key, value in entity2id.items()}
    id2relation = {value: key for key, value in relation2id.items()}
    num_entity = len(entity2id)
    num_relation = len(relation2id)

    print('num of entity:', num_entity)
    print('num of relation:', num_relation)

    data_train = set()
    data_valid = set()
    data_test = set()
    data_all = set()

    rt_h = np.zeros((num_relation, num_entity), dtype=int)
    rh_t = np.zeros((num_relation, num_entity), dtype=int)
    hpt = np.zeros(num_relation, dtype=float)
    tph = np.zeros(num_relation, dtype=float)

    with open(os.path.join(args.data_path, 'encode/train_encode.txt')) as f:
        for line in f.readlines():
            t = tuple(int(i) for i in line.strip().split('\t'))
            data_train.add(t)
            rt_h[t[1]][t[2]] += 1
            rh_t[t[1]][t[0]] += 1

    with open(os.path.join(args.data_path, 'encode/valid_encode.txt')) as f:
        for line in f.readlines():
            t = tuple(int(i) for i in line.strip().split('\t'))
            data_valid.add(t)
            rt_h[t[1]][t[2]] += 1
            rh_t[t[1]][t[0]] += 1

    with open(os.path.join(args.data_path, 'encode/test_encode.txt')) as f:
        for line in f.readlines():
            t = tuple(int(i) for i in line.strip().split('\t'))
            data_test.add(t)
            rt_h[t[1]][t[2]] += 1
            rh_t[t[1]][t[0]] += 1
    data_all = data_train | data_valid | data_test
    hpt = np.delete(rt_h.sum(axis=1), 0) / np.delete(np.count_nonzero(rt_h, axis=1), 0)
    tph = np.delete(rh_t.sum(axis=1), 0) / np.delete(np.count_nonzero(rh_t, axis=1), 0)

    rel_type = np.zeros(num_relation - 1, dtype=int)

    for i in range(len(rel_type)):
        if hpt[i] < 1.5:
            if tph[i] < 1.5:
                rel_type[i] = 0
            else:
                rel_type[i] = 1
        else:
            if tph[i] < 1.5:
                rel_type[i] = 2
            else:
                rel_type[i] = 3
    rel_type = np.insert(rel_type, 0, -1, axis=0)  # add a NULL type for NULL relation

    # num_rel_1_1 = (rel_type == 0).sum()
    # num_rel_1_n = (rel_type == 1).sum()
    # num_rel_n_1 = (rel_type == 2).sum()
    # num_rel_n_n = (rel_type == 3).sum()
    num_rel_1_1 = 0
    num_rel_1_n = 0
    num_rel_n_1 = 0
    num_rel_n_n = 0
    for triple in data_test:
        if rel_type[triple[1]] == 0:
            num_rel_1_1 += 1
        elif rel_type[triple[1]] == 1:
            num_rel_1_n += 1
        elif rel_type[triple[1]] == 2:
            num_rel_n_1 += 1
        elif rel_type[triple[1]] == 3:
            num_rel_n_n += 1

    num_test = len(data_test)
    print(num_test)
    data_test_list = list(data_test)
    result = []
    start_index = 0
    p = mp.Pool(processes=args.num_process)
    process_id = 0
    while start_index < num_test:
        if start_index + args.size_frag < num_test:
            end_index = start_index + args.size_frag
        else:
            end_index = num_test
        result.append(p.apply_async(test_frag, args=(
            args, num_entity, num_relation, data_test_list[start_index:end_index], data_all, rel_type, process_id)))
        start_index += args.size_frag
        process_id += 1
    p.close()
    p.join()
    result_array = []
    for i in result:
        result_array.append(i.get())
    result = np.array(result_array)
    result_sum = np.sum(result, axis=0)
    rank_head_raw = result_sum[0]
    rank_head_filter = result_sum[1]
    rank_tail_raw = result_sum[2]
    rank_tail_filter = result_sum[3]
    hit10_head_raw = result_sum[4]
    hit10_head_filter = result_sum[5]
    hit10_tail_raw = result_sum[6]
    hit10_tail_filter = result_sum[7]
    hit10_head_1_1 = result_sum[8]
    hit10_head_1_n = result_sum[9]
    hit10_head_n_1 = result_sum[10]
    hit10_head_n_n = result_sum[11]
    hit10_tail_1_1 = result_sum[12]
    hit10_tail_1_n = result_sum[13]
    hit10_tail_n_1 = result_sum[14]
    hit10_tail_n_n = result_sum[15]
    rank_rel_raw = result_sum[16]
    rank_rel_filter = result_sum[17]
    hit1_rel_raw = result_sum[18]
    hit1_rel_filter = result_sum[19]
    hit10_rel_raw = result_sum[20]
    hit10_rel_filter = result_sum[21]

    rank_raw = (rank_head_raw + rank_tail_raw) / num_test / 2
    rank_filter = (rank_head_filter + rank_tail_filter) / num_test / 2
    hit10_raw = (hit10_head_raw + hit10_tail_raw) / num_test / 2
    hit10_filter = (hit10_head_filter + hit10_tail_filter) / num_test / 2

    print('MR(Raw): {}\tMR(Filter): {}\tHit@10(Raw): {}\tHit@10(Filter): {}'.format(rank_raw, rank_filter, hit10_raw,
                                                                                    hit10_filter))
    print('Predicting Head - 1-1: {}\t1-N: {}\tN-1: {}\tN-N: {}'.format(hit10_head_1_1 / num_rel_1_1,
                                                                        hit10_head_1_n / num_rel_1_n,
                                                                        hit10_head_n_1 / num_rel_n_1,
                                                                        hit10_head_n_n / num_rel_n_n))
    print('Predicting tail - 1-1: {}\t1-N: {}\tN-1: {}\tN-N: {}'.format(hit10_tail_1_1 / num_rel_1_1,
                                                                        hit10_tail_1_n / num_rel_1_n,
                                                                        hit10_tail_n_1 / num_rel_n_1,
                                                                        hit10_tail_n_n / num_rel_n_n))
    print(
        'Predicting relation: MR(Raw): {}\tMR(Filter): {}\tHit@1(Raw): {}\tHit@1(Filter): {}\t Hit@10(Raw): {}\tHit@10(Filter): {}'.format(
            rank_rel_raw/num_test, rank_rel_filter/num_test, hit1_rel_raw/num_test, hit1_rel_filter/num_test, hit10_rel_raw/num_test, hit10_rel_filter/num_test))


if __name__ == '__main__':
    main()
