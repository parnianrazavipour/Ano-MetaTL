import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict, Counter
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample(user_train, usernum, itemnum, maxlen):
    if random.random() < 0.5:
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        if len(user_train[user]) < maxlen:
            nxt_idx = len(user_train[user]) - 1
        else:
            nxt_idx = np.random.randint(maxlen, len(user_train[user]))

        nxt = user_train[user][nxt_idx]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][min(0, nxt_idx - 1 - maxlen): nxt_idx - 1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        curr_rel = user
        support_triples, support_negative_triples, query_triples, negative_triples = [], [], [], []
        for idx in range(maxlen - 1):
            support_triples.append([seq[idx], curr_rel, pos[idx]])
            support_negative_triples.append([seq[idx], curr_rel, neg[idx]])
        query_triples.append([seq[-1], curr_rel, pos[-1]])
        negative_triples.append([seq[-1], curr_rel, neg[-1]])

        return support_triples, support_negative_triples, query_triples, negative_triples, curr_rel

    else:
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        list_idx = random.sample([i for i in range(len(user_train[user]))], maxlen + 1)
        list_item = [user_train[user][i] for i in sorted(list_idx)]

        nxt = list_item[-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(list_item[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        curr_rel = user
        support_triples, support_negative_triples, query_triples, negative_triples = [], [], [], []
        for idx in range(maxlen - 1):
            support_triples.append([seq[idx], curr_rel, pos[idx]])
            support_negative_triples.append([seq[idx], curr_rel, neg[idx]])
        query_triples.append([seq[-1], curr_rel, pos[-1]])
        negative_triples.append([seq[-1], curr_rel, neg[-1]])

        return support_triples, support_negative_triples, query_triples, negative_triples, curr_rel


def sample_function_mixed(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(user_train, usernum, itemnum, maxlen))
        support, support_negative, query, negative, curr_rel = zip(*one_batch)
        result_queue.put(([support, support_negative, query, negative], curr_rel))


def sample_ano(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED, ctr=0.05):
    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            support_triples, support_negative_triples, \
                query_triples, negative_triples, curr_rel = sample(user_train, usernum, itemnum, maxlen)
            support = [*np.array(support_triples)[:, 0], np.array(support_triples)[-1, -1]]
            query_pos = [*support[1:], query_triples[0][2]]
            query_neg = [*support[1:], negative_triples[0][2]]
            support_neg = [*np.array(support_negative_triples)[:, 2]]
            support_label = 0
            if np.random.random() < ctr:
                support[-1] = support_neg[-1]
                support_label = 1
            one_batch.append([curr_rel, support, support_label, query_pos, query_neg])
        curr_rel, support, support_label, query_pos, query_neg = zip(*one_batch)
        result_queue.put([curr_rel, support, support_label, query_pos, query_neg])


class WarpSampler(object):
    def __init__(self, user, usernum, itemnum, sampler, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sampler, args=(user,
                                              usernum,
                                              itemnum,
                                              batch_size,
                                              maxlen,
                                              self.result_queue,
                                              np.random.randint(2e9)
                                              )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
