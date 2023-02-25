import argparse
import random

import numpy as np
import torch
from torch import nn

from collections import OrderedDict

from ad_maml.eval_dataset import load_val_test_dataset, SeriesTestDataset
from ad_maml.model import AnomalyDetector

from ad_maml.train import train_model
from sampler import WarpSampler, sample_ano
from utils import data_load


def get_params():
    args = argparse.ArgumentParser()

    args.add_argument("-data", "--dataset", default="beauty", type=str)
    args.add_argument("--seed", default=None, type=int)
    args.add_argument("--K", default=5, type=int)
    args.add_argument("--embed_dim", default=100, type=int)
    args.add_argument("--batch_size", default=1024, type=int)
    args.add_argument("--adapt_lr", default=0.01, type=float)
    args.add_argument("--meta_lr", default=0.005, type=float)
    args.add_argument("--epoch", default=100000, type=int)
    args.add_argument("--eval_epoch", default=1000, type=int)
    args.add_argument("--dropout_p", default=0.5, type=float)
    args.add_argument("--device", default="cpu", type=str)
    args.add_argument("--embeder_path", default=None, type=str)
    args.add_argument("--eval_path", default="data/beauty/beauty_test_new_user.csv", type=str)
    args.add_argument("--model_save_path", default="models/maml_ad.pt", type=str)

    args = args.parse_args()

    return args


if __name__ == '__main__':
    args = get_params()

    if args.seed is not None:
        SEED = args.seed
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)

    user_train, usernum_train, max_item, user_input_test, user_test, user_input_valid, user_valid = data_load(
        args.dataset, args.K)
    train_sampler = WarpSampler(user_train, usernum_train, max_item, sample_ano, batch_size=args.batch_size,
                                maxlen=args.K,
                                n_workers=3)
    val_data, test_data = load_val_test_dataset(args.K, args.eval_path)
    val_dataset = SeriesTestDataset(val_data, max_item)
    test_dataset = SeriesTestDataset(test_data, max_item)

    device = args.device
    model = AnomalyDetector(args.K, input_embed_size=args.embed_dim, dropout_p=args.dropout_p)
    model.to(device)

    embedder = nn.Embedding(max_item + 1, args.embed_dim)
    if args.embeder_path is not None:
        embeder_data = torch.load(args.embeder_path)
        embedder.load_state_dict(OrderedDict([('weight', embeder_data['embedding.embedding.weight'])]))
        print('Used pretrained data')
    else:
        nn.init.xavier_uniform_(embedder.weight)
    embedder.to(device)

    train_model(model, embedder, train_sampler, val_dataset, test_dataset, args)
    train_sampler.close()
