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
    def __init__(self, num_entity, num_relation, dim_embedding):
        super(Model, self).__init__()
        self.embed_entity = nn.Embedding(num_entity, dim_embedding)
        self.embed_relation = nn.Embedding(num_relation, dim_embedding)
        self.dim_embedding = dim_embedding
        initrange = 6 / math.sqrt(dim_embedding)
        self.embed_entity.weight.data.uniform_(-initrange, initrange)
        self.embed_relation.weight.data.uniform_(-initrange, initrange)
        self.embed_relation.weight.data = F.normalize(self.embed_relation.weight.data)

    def forward(self, h_batch, r_batch, t_batch, h_neg_batch, r_neg_batch, t_neg_batch):
        '''
        :param h_batch: tensor of head entity
        :param r_batch: tensor of relation
        :param t_batch: tensor of tail entity
        :return:
        '''
        g_n_pos = []
        g_n_neg = []
        g_p_pos = []
        g_p_neg = []
        idx = 0
        for (h, r, t, h_neg, r_neg, t_neg) in zip(h_batch, r_batch, t_batch, h_neg_batch, r_neg_batch, t_neg_batch):
            idx += 1
            # print(idx, '/', len(h_batch))
            g_n_pos.append(self.g_n(h, h, r, t))
            g_n_neg.append(self.g_n(h_neg, h, r, t))
            g_p_pos.append(self.g_p(r, t, h, r, t))
            g_p_neg.append(self.g_p(r_neg, t_neg, h, r, t))
        g_n_pos = torch.cat(g_n_pos)
        g_n_neg = torch.cat(g_n_neg)
        g_p_pos = torch.cat(g_p_pos)
        g_p_neg = torch.cat(g_p_neg)
        loss = torch.sum(- F.logsigmoid(g_n_pos) - F.logsigmoid(g_n_neg) - F.logsigmoid(g_p_pos) - F.logsigmoid(g_p_neg))
        return loss

    def g_n(self, h_prime, h, r, t):
        embed_h = self.embed_entity(h)  # 1 * dim
        embed_r = self.embed_relation(r)  # 1 * dim
        embed_t = self.embed_entity(t)  # 1 * dim
        embed_h_prime = self.embed_entity(h_prime)  # 1 * dim

        neighbor_context = np.array(getNeighborContext(h, 2))
        size_neighbor_context = len(neighbor_context)

        if size_neighbor_context == 0:  # if the entity has no neighbor
            g_n =  Variable(torch.FloatTensor([-100]), requires_grad=False)
        else:
            # remove (r, t) from neighbor context
            for i in range(len(neighbor_context)):
                if np.array_equal(neighbor_context[i], [r, t]):
                    neighbor_context = np.delete(neighbor_context, i, 0)

            neighbor_context_r = neighbor_context[:, 0]
            neighbor_context_t = neighbor_context[:, 1]

            embed_nc_r = self.embed_relation(
                Variable(torch.LongTensor(neighbor_context_r)))  # size_neighbor_context * dim
            embed_nc_t = self.embed_entity(
                Variable(torch.LongTensor(neighbor_context_t)))  # size_neighbor_context * dim

            a = torch.norm(embed_nc_t - embed_nc_r + embed_r - embed_t, p=2, dim=1,
                           keepdim=True)  # size_neighbor_context * 1
            alpha = F.softmax(-a, dim=0)  # size_neighbor_context * 1 # softmin(a)
            g_n = - torch.sum(alpha * torch.norm(embed_h_prime + embed_nc_r - embed_nc_t, p=2, dim=1,
                                                          keepdim=True))
        return g_n

    def g_p(self, r_prime, t_prime, h, r, t):
        embed_h = self.embed_entity(h)  # 1 * dim
        embed_r = self.embed_relation(r)  # 1 * dim
        embed_t = self.embed_entity(t)  # 1 * dim
        embed_r_prime = self.embed_relation(r_prime)  # 1 * dim
        embed_t_prime = self.embed_entity(t_prime)  # 1 * dim

        path_context = np.array(getPathContext(h, t, 0))
        size_path_context = len(path_context)

        if size_path_context == 0:  # if the entity pair has no path
            g_p = Variable(torch.FloatTensor([-100]), requires_grad=False)
        else:
            p = []  # size_path_context * dim_embedding
            for i in range(size_path_context):
                p_i = path_context[i]
                embed_path = self.embed_relation(Variable(torch.LongTensor(np.abs(p_i))))  # len(p_i) * dim_embedding
                rel_sign = Variable(torch.sign(torch.FloatTensor(p_i)).resize_(1, len(p_i)),
                                    requires_grad=False)  # 1 * len(p_i)
                p.append(rel_sign.matmul(embed_path))  # torch.sum(embed_path * rel_sign, 1, True) # 1 * dim_embedding
            embed_p = torch.cat(p, 0)  # size_path_context * dim_embedding
            b = torch.norm(embed_h + embed_p - embed_t, p=2, dim=1, keepdim=True)  # size_path_context * 1
            beta = F.softmax(-b, dim=0)  # size_path_context * 1
            g_p = - torch.sum(beta * (
            torch.norm(embed_h + embed_p - embed_t_prime, p=2, dim=1, keepdim=True) + torch.norm(embed_p - embed_r_prime,
                                                                                               p=2, dim=1,
                                                                                               keepdim=True)))
        return g_p

    def score_function(self, h, r, t, h_prime, r_prime, t_prime):
        return F.logsigmoid(self.g_n(h_prime, h, r, t)) + F.logsigmoid(self.g_p(r_prime, t_prime, h, r, t))

    def save_embedding(self, path):
        torch.save(self.state_dict(), path)

def getPathContext(h, t, dbNum):
    '''
    Get the path context of a pair of entity h and t
    :param h: a variable containing the id of head entity
    :param t: a variable containing the id of tail entity
    :return: a list of relation path
    '''
    r = redis.StrictRedis(host='223.3.78.231', port=6380, db=dbNum)
    pathList = []
    key_2 = str(h.data[0]) + ',' + str(t.data[0]) + ':' + str(2)
    key_3 = str(h.data[0]) + ',' + str(t.data[0]) + ':' + str(3)
    value_2 = r.get(key_2)
    value_3 = r.get(key_3)
    if value_2 is not None:
        path_json_2 = json.loads(value_2.decode())
        for key in path_json_2.keys():
            pathList.append(json.loads(key))
    if value_3 is not None:
        path_json_3 = json.loads(value_3.decode())
        for key in path_json_3.keys():
            pathList.append(json.loads(key))
    return pathList

def getNeighborContext(e, dbNum):
    '''
    Get the neighbor context of an entity e
    :param e: a variable containing the id of input entity
    :return: a list of (relation, entity) list
    '''
    r = redis.StrictRedis(host='223.3.78.231', port=6380, db=dbNum)
    neighborList = []
    # print('e', e.data[0])
    for neighbor in r.smembers(e.data[0]):
        neighbor = [int(n) for n in neighbor.decode().split(',')]
        if neighbor[0] > 0:
            neighborList.append(neighbor) # only use the out edge from entity h
    return neighborList

def test():
    # load embeddings
    model = Model()
    model.load_state_dict('')

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
        loss = []
        p = mp.Pool(processes=args.num_process)
        for batch_idx, data_batch in enumerate(train_loader, start=1):
            # train_batch(epoch, args, model, train_data, train_loader, optimizer, num_entity, batch_idx, data_batch)
            loss.append(p.apply_async(train_batch, args=(epoch, args, model, train_data, train_loader, optimizer, num_entity, batch_idx, data_batch)))
        # test_epoch(model, test_loader)
        p.close()
        p.join()
        time_end = time.time()
        loss_total = sum(loss)
        print('Epoch: {}\tLoss: {:.2f}\tTime used: {:.2f}'.format(epoch, loss_total, (time_end-time_start)/60))
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

    h_batch = Variable(torch.LongTensor(h_batch))
    r_batch = Variable(torch.LongTensor(r_batch))
    t_batch = Variable(torch.LongTensor(t_batch))
    h_n_batch = Variable(torch.LongTensor(h_n_batch))
    r_n_batch = Variable(torch.LongTensor(r_n_batch))
    t_n_batch = Variable(torch.LongTensor(t_n_batch))
    
    loss = model.forward(h_batch, r_batch, t_batch, h_n_batch, r_n_batch, t_n_batch)
    loss.backward()
    optimizer.step()
    time_end = time.time()
    print('{}\tEpoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}\tTime used: {:.2f} min'.format(
        pid, epoch, batch_idx, math.ceil(len(data)/args.size_batch),
                    100. * batch_idx / math.ceil(len(data)/args.size_batch), loss.data[0], (time_end-time_start)/60))
    return loss.data[0]

def main():
    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument('--data-path', type=str, default='data/fb15k/')
    parser.add_argument('--save-path', type=str, default='model_save/')
    parser.add_argument('--dimension', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-epoch', type=int, default=1000)
    parser.add_argument('--size-batch', type=int, default=1024)
    parser.add_argument('--num-process', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)

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

    # load traning data
    # print('Loading data...')
    # train_data = TripleData(os.path.join(args.data_path, 'encode/train_encode.txt'))
    # train_loader = TripleLoader(train_data, args.size_batch)

    model = Model(num_entity, num_relation, args.dimension)
    model.share_memory()

    train(args, model, num_entity)


if __name__ == '__main__':
    main()
    # test()