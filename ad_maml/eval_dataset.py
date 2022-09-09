import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SeriesTestDataset(Dataset):
    def __init__(self, data, max_item):
        self.data_dict = data
        self.max_item = max_item
        self.users = list(self.data_dict.keys())

    def get_random_item(self):
        return np.random.randint(1, self.max_item + 1)

    def get_ano_item(self, item_set):
        item = self.get_random_item()
        while item in item_set:
            item = self.get_random_item()
        return item

    def __getitem__(self, idx):
        user = self.users[idx]
        user_data = self.data_dict[user]

        support = user_data[:-1]
        query_norm = user_data[1:]
        query_ano = query_norm.copy()
        query_ano[-1] = self.get_ano_item(set(user_data))
        return [user, support, 0, query_norm, query_ano]

    def __len__(self):
        return len(self.users)


def load_val_test_dataset(k, data_path, val_split=0.3):
    df = pd.read_csv(data_path, sep='\t', header=None).drop(2, axis=1)
    df = df.groupby(0).agg(list)
    val_data = {}
    test_data = {}
    for user in df.index:
        data = df.loc[user, 1]
        if len(data) < k + 1:
            continue
        if np.random.random() < val_split:
            val_data[user] = data[: k + 1]
        else:
            test_data[user] = data[: k + 1]
    return val_data, test_data
