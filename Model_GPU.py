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
import torch.multiprocessing as mp
import argparse
import time
from tqdm import tqdm
from tqdm import trange

dtype_LongTensor_cpu = torch.LongTensor
dtype_LongTensor_cuda = torch.cuda.LongTensor
dtype_FloatTensor_cpu = torch.FloatTensor
dtype_FloatTensor_cuda = torch.cuda.FloatTensor

size_path_context = 10
size_neighbor_context = 10
max_len_path = 3
id_null_relation = 0
id_null_entity = 14951

redis_host = '223.3.77.122'
redis_port = 6379
path_context_db = 4
neighbor_context_db = 6
redis_pool_path = redis.ConnectionPool(host=redis_host, port=redis_port, db=path_context_db)
redis_pool_neighbor = redis.ConnectionPool(host=redis_host, port=redis_port, db=neighbor_context_db)

vec_path_entity = 'vec/entity2vec.txt'
vec_path_relation = 'vec/entity2vec.txt'

def get_path_context(h, t):
    '''
    Get the path context of a pair of entity h and t
    :param h: a variable containing the id of head entity
    :param t: a variable containing the id of tail entity
    :return: a list of relation path
    '''
    r = redis.Redis(connection_pool=redis_pool_path) # redis.StrictRedis(host=redis_host, port=redis_port, db=path_context_db)
    value = r.get(','.join([str(h.data[0]), str(t.data[0])]))
    if value is not None:
        pathList = json.loads(value.decode())
    else:
        pathList = []
    return pathList

def get_path_context_batch(h_batch, t_batch):
    '''
    Get the path context of a batch, and pad them as a tensor
    :param h_batch:
    :param t_batch:
    :return:
    '''
    path_idx_batch = []
    for (h, t) in zip(h_batch, t_batch):
        path_context = get_path_context(h, t)
        for i in range(len(path_context)):
            while len(path_context[i]) < max_len_path:
                path_context[i].append(id_null_relation)
        while len(path_context) < size_path_context:
            padding_path = []
            while len(padding_path) < max_len_path:
                padding_path.append(id_null_relation)
            path_context.append(padding_path)
        path_idx_batch.append(path_context)
    return path_idx_batch # size_batch * size_path_context * length_path

def get_neighbor_context(e):
    '''
    Get the neighbor context of an entity e
    :param e: a variable containing the id of input entity
    :return: a list of (relation, entity) list
    '''
    r = redis.Redis(connection_pool=redis_pool_neighbor) # redis.StrictRedis(host=redis_host, port=redis_port, db=neighbor_context_db)
    value = r.get(str(e.data[0]))
    if value is not None:
        neighborList = json.loads(value.decode())
    else:
        neighborList = []
    return neighborList

def get_neighbor_context_batch(e_batch):
    neighbor_idx_batch = []
    for e in e_batch:
        neighbor_context = get_neighbor_context(e)
        while len(neighbor_context) < size_neighbor_context:
            padding_neighbor = [id_null_relation, id_null_entity]
            neighbor_context.append(padding_neighbor)
        neighbor_idx_batch.append(neighbor_context)
    return neighbor_idx_batch # size_batch * size_neighbor_context * 2

class TripleData(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            data = [[int(s) for s in line.split('\t')] for line in f.readlines()]
        self.data = data
        self.data_set = set(tuple(i) for i in data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class TripleLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle):
        super(TripleLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=True)


class Model(nn.Module):
    def __init__(self, num_entity, num_relation, args):
        super(Model, self).__init__()
        self.dim_embedding = args.dimension
        self.embed_entity = nn.Embedding(num_entity, self.dim_embedding)
        self.embed_relation = nn.Embedding(num_relation, self.dim_embedding)

        if args.init:
            with open(os.path.join(args.data_path, vec_path_entity)) as file_entity_vec:
                entity_vec = [[float(n) for n in line.strip().split('\t')] for line in file_entity_vec.readlines()]
                entity_vec.append([0] * self.dim_embedding)
                self.embed_entity.weight.data = torch.FloatTensor(entity_vec)
            with open(os.path.join(args.data_path, vec_path_relation)) as file_relation_vec:
                relation_vec = [[float(n) for n in line.strip().split('\t')] for line in file_relation_vec.readlines()]
                relation_vec.insert(0, [0] * self.dim_embedding)
                self.embed_relation.weight.data = torch.FloatTensor(relation_vec)
        else:
            initrange = 6 / math.sqrt(self.dim_embedding)
            self.embed_entity.weight.data.uniform_(-initrange, initrange)
            self.embed_relation.weight.data.uniform_(-initrange, initrange)
            self.embed_relation.weight.data = F.normalize(self.embed_relation.weight.data)

    def forward(self, h_batch, r_batch, t_batch, h_neg_batch, r_neg_batch, t_neg_batch):
        '''
        :param h_batch: variable containing tensor of head entity
        :param r_batch: variable containing tensor of relation
        :param t_batch: variable containing tensor of tail entity
        :return:
        '''
        embed_h_batch = self.embed_entity(h_batch)  # size_batch * dim
        embed_r_batch = self.embed_relation(r_batch)  # size_batch * dim
        embed_t_batch = self.embed_entity(t_batch)  # size_batch * dim
        embed_h_neg_batch = self.embed_entity(h_neg_batch)  # size_batch * dim
        embed_r_neg_batch = self.embed_entity(r_neg_batch)  # size_batch * dim
        embed_t_neg_batch = self.embed_entity(t_neg_batch)  # size_batch * dim

        # get neighbor context
        neighbor_context = torch.LongTensor(get_neighbor_context_batch(h_batch)).type(dtype_LongTensor_cuda)
        embed_neighbor_context_r = self.embed_relation(Variable(neighbor_context[:,:,0])) # size_batch * size_neighbor_context * dim
        embed_neighbor_context_t = self.embed_entity(Variable(neighbor_context[:,:,1])) # size_batch * size_neighbor_context * dim
        neighbor_tmp = embed_neighbor_context_r - embed_neighbor_context_t # size_batch * size_neighbor_context * dim
        a = torch.norm(- neighbor_tmp + embed_r_batch[:,np.newaxis,:] - embed_t_batch[:,np.newaxis,:], p=2, dim=2, keepdim=False) # size_batch * size_neighbor_context
        alpha = F.softmin(a, dim=1) # size_batch * size_neighbor_context
        g_n_pos = torch.sum(alpha * torch.norm(embed_h_batch[:,np.newaxis,:] + neighbor_tmp, p=2, dim=2, keepdim=False)) # size_batch
        g_n_neg = torch.sum(alpha * torch.norm(embed_h_neg_batch[:,np.newaxis,:] + neighbor_tmp, p=2, dim=2, keepdim=False)) # size_batch

        # get path context
        path_context = torch.LongTensor(get_path_context_batch(h_batch, t_batch)).type(dtype_LongTensor_cuda)
        rel_sign = torch.sign(path_context.type(dtype_FloatTensor_cuda))  # size_batch * size_path_context * length_path
        embed_path_list = []
        for i in range(len(path_context)): # because the indices of embedding should be <= 2
            embed_path_list.append(self.embed_relation(Variable(torch.abs(path_context[i]))))
        embed_path_context = torch.cat([torch.unsqueeze(embed, 0) for embed in embed_path_list], 0) # size_batch * size_path_context * length_path * dim
        embed_path = torch.sum(Variable(rel_sign[:,:,:,np.newaxis], requires_grad=True) * embed_path_context, dim=2, keepdim=False) # size_batch * size_path_context * dim
        b = torch.norm(embed_h_batch[:,np.newaxis,:] + embed_path - embed_t_batch[:,np.newaxis,:], p=2, dim=2, keepdim=False) # size_batch * size_path_context
        beta = F.softmin(b, dim=1) # size_batch * size_path_context
        path_tmp = embed_h_batch[:,np.newaxis,:] + embed_path # size_batch * size_path_context * dim
        g_p_pos = torch.sum(beta * (torch.norm(path_tmp - embed_t_batch[:,np.newaxis,:], p=2, dim=2, keepdim=False) + torch.norm(embed_path - embed_r_batch[:,np.newaxis,:], p=2, dim=2, keepdim=False)), dim=1, keepdim=False)
        g_p_neg = torch.sum(beta * (torch.norm(path_tmp - embed_t_neg_batch[:,np.newaxis,:], p=2, dim=2, keepdim=False) + torch.norm(embed_path - embed_r_neg_batch[:,np.newaxis,:], p=2, dim=2, keepdim=False)), dim=1, keepdim=False)

        loss = torch.sum(- F.logsigmoid(g_n_pos) - F.logsigmoid(-g_n_neg) - F.logsigmoid(g_p_pos) - F.logsigmoid(-g_p_neg))# / len(h_batch)
        return loss

    def score_function(self, h, r, t, h_prime, r_prime, t_prime):
        return F.logsigmoid(self.g_n(h_prime, h, r, t)) + F.logsigmoid(self.g_p(r_prime, t_prime, h, r, t))

    def save_embedding(self, path):
        torch.save(self.state_dict(), path)

# def test():
    # load embeddings
    # model = Model()
    # model.load_state_dict('')

    # load test data

    # test the performance using the embedding

def train(args, model, num_entity):
    model.train()
    print('Start training...')
    # torch.manual_seed(rank)
    train_data = TripleData(os.path.join(args.data_path, 'encode/train_encode.txt'))
    train_loader = TripleLoader(train_data, batch_size=args.size_batch, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.num_epoch+1):
        time_start = time.time()
        loss_total = 0
        # p = mp.Pool(processes=args.num_process)
        for batch_idx, data_batch in enumerate(train_loader, start=1):
            loss = train_batch(epoch, args, model, train_data, train_loader, optimizer, num_entity, batch_idx, data_batch)
            loss_total += loss
            # loss.append(p.apply_async(train_batch, args=(epoch, args, model, train_data, train_loader, optimizer, num_entity, batch_idx, data_batch)))
        # test_epoch(model, test_loader)
        # p.close()
        # p.join()
        time_end = time.time()
        print('Epoch: {}\tLoss: {:.2f}\tTime used: {:.2f} min'.format(epoch, loss_total, (time_end-time_start)/60))
        model.save_embedding(os.path.join(args.save_path,
                                          'embed_' + str(args.dimension) + '_' + str(args.learning_rate) + '_' + str(
                                              args.size_batch) + '_' + str(epoch)))

# def train_epoch(epoch, args, model, data, data_loader, optimizer, num_entity, time_start, p):
#     for batch_idx, data_batch in enumerate(data_loader, start=1):
#         p.apply_async(train_batch, args=(epoch, args, model, data, data_loader, optimizer, num_entity, time_start, batch_idx, data_batch, p))


def train_batch(epoch, args, model, data, data_loader, optimizer, num_entity, batch_idx, data_batch):
    pid = os.getpid()
    time_start = time.time()
    # print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]'.format(
    #     pid, epoch, batch_idx, math.ceil(len(data) / args.size_batch),
    #     100. * batch_idx / math.ceil(len(data) / args.size_batch)))
    model.embed_entity.weight.data = F.normalize(model.embed_entity.weight.data)

    h_batch, r_batch, t_batch = data_batch
    h_batch = list(h_batch)
    r_batch = list(r_batch)
    t_batch = list(t_batch)

    # generate negative samples
    h_n_batch = []
    r_n_batch = []
    t_n_batch = []
    for i in range(len(h_batch)):
        r_n_batch.append(r_batch[i])
        flag = random.randint(0, 1)
        if flag == 0:  # substitute head entity
            h = random.randint(0, num_entity - 1)
            while (h, r_batch[i], t_batch[i]) in data.data_set:
                h = random.randint(0, num_entity - 1)
            h_n_batch.append(h)
            t_n_batch.append(t_batch[i])
        else:  # substitute tail entity
            t = random.randint(0, num_entity - 1)
            while (h_batch[i], r_batch[i], t) in data.data_set:
                t = random.randint(0, num_entity - 1)
            h_n_batch.append(h_batch[i])
            t_n_batch.append(t)

    model.zero_grad()

    h_batch = Variable(torch.LongTensor(h_batch)).type(dtype_LongTensor_cuda)
    r_batch = Variable(torch.LongTensor(r_batch)).type(dtype_LongTensor_cuda)
    t_batch = Variable(torch.LongTensor(t_batch)).type(dtype_LongTensor_cuda)
    h_n_batch = Variable(torch.LongTensor(h_n_batch)).type(dtype_LongTensor_cuda)
    r_n_batch = Variable(torch.LongTensor(r_n_batch)).type(dtype_LongTensor_cuda)
    t_n_batch = Variable(torch.LongTensor(t_n_batch)).type(dtype_LongTensor_cuda)

    time1 = time.time()
    loss = model.forward(h_batch, r_batch, t_batch, h_n_batch, r_n_batch, t_n_batch)
    time2 = time.time()
    # print('Time of forward: {:.8f} min'.format((time2-time1)/60))
    loss.backward()
    time3 = time.time()
    # print('Time of backward: {:.8f} min'.format((time3 - time2) / 60))
    optimizer.step()
    time_end = time.time()
    # print('{}\tEpoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}\tTime used: {:.2f} min'.format(
    #     pid, epoch, batch_idx, math.ceil(len(data)/args.size_batch),
    #                 100. * batch_idx / math.ceil(len(data)/args.size_batch), loss.data[0], (time_end-time_start)/60))
    return loss.data[0]

def main():
    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument('--data-path', type=str, default='data/fb15k')
    parser.add_argument('--save-path', type=str, default='model_save')
    parser.add_argument('--dimension', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-epoch', type=int, default=1000)
    parser.add_argument('--size-batch', type=int, default=1000)
    parser.add_argument('--num-process', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--init', type=bool, default=False) # True - init embeddings using pre-trained embeddings

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # config = Config(data_path='data/fb15k/', save_path='model_save/', dimension=50, learning_rate=0.001, num_epoch=1000, size_batch=16)
    with open(os.path.join(args.data_path, 'encode/entity_id.txt')) as f:
        entity2id = {line.split('\t')[0].strip() : line.split('\t')[1].strip() for line in f.readlines()}
    with open(os.path.join(args.data_path, 'encode/relation_id.txt')) as f:
        relation2id = {line.split('\t')[0].strip() : line.split('\t')[1].strip() for line in f.readlines()}
    id2entity = {value : key for key, value in entity2id.items()}
    id2relation = {value : key for key, value in relation2id.items()}
    num_entity = len(entity2id)
    num_relation = len(relation2id)

    print('num of entity:', num_entity)
    print('num of relation:', num_relation)

    # load traning data
    # print('Loading data...')
    # train_data = TripleData(os.path.join(args.data_path, 'encode/train_encode.txt'))
    # train_loader = TripleLoader(train_data, args.size_batch)

    model = Model(num_entity, num_relation, args)
    # model.share_memory()
    model.cuda()

    train(args, model, num_entity)

if __name__ == '__main__':
    main()
    # test()