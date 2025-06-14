# -*- coding: utf-8 -*- 

import numpy as np
import json, os, pickle
import pandas as pd
import torch
from .utils import get_folds
from torch.utils.data import TensorDataset, DataLoader, Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Dataset(object):

    def __init__(self, data, concept_map,
                 num_students, num_questions, num_concepts):

        self._raw_data = data
        self._concept_map = concept_map
        self._data = {}
        for sid, qid, label in data:
            self._data.setdefault(int(sid), {})
            self._data[sid].setdefault(int(qid), {})
            self._data[sid][qid] = label

        self.n_students = num_students
        self.n_questions = num_questions
        self.n_concepts = num_concepts

    @property
    def num_students(self):
        return self.n_students

    @property
    def num_questions(self):
        return self.n_questions

    @property
    def num_concepts(self):
        return self.n_concepts

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def data(self):
        return self._data

    @property
    def concept_map(self):
        return self._concept_map


class CDDataset(_Dataset, Dataset):
    def __init__(self, data, concept_map, num_stus, num_ques, num_cpts):
        super().__init__(data, concept_map, num_stus, num_ques, num_cpts)

    def __getitem__(self, item):
        stu_id, que_id, score = self._raw_data[item]
        stu_id = int(stu_id)
        que_id = int(que_id)
        concepts = np.array([0.] * self.n_concepts)
        concepts[self._concept_map[str(que_id)]] = 1.
        return stu_id, que_id, score, concepts

    def __len__(self):
        return len(self._raw_data)


class MyDataset(object):

    def __init__(self, train_data, test_data, cpt_map, num_stus, num_ques, num_cpts):

        train_data = np.array(train_data)
        test_data = np.array(test_data)

        self.train_data = CDDataset(train_data, cpt_map, num_stus, num_ques, num_cpts)  # 训练集
        self.test_data = CDDataset(test_data, cpt_map, num_stus, num_ques, num_cpts)  # 测试集

    def get_dataloader(self, batch_size):
        train_data_loader = DataLoader(self.train_data,
                                       batch_size=batch_size,
                                       shuffle=True)
        test_data_loader = DataLoader(self.test_data,
                                      batch_size=batch_size,
                                      shuffle=True)

        return train_data_loader, test_data_loader


def dataset4train(dataset_name, data_config, batch_size):

    data_config = data_config[dataset_name]
    folds = data_config["folds"]  # k-fold

    inter_matrix_path = os.path.join(data_config["dpath"], data_config[f"inter_matrix"])
    data_path = os.path.join(data_config["dpath"], data_config[f"all_data"])
    cpt_path = os.path.join(data_config["dpath"], data_config[f"que_cpt_dict"])

    inter_matrix = np.load(inter_matrix_path)["interaction_matrix"]
    all_data = pd.read_csv(data_path)
    cpt_map = json.load(open(cpt_path, 'r'))
    train_list, test_list = get_folds(all_data, folds)

    train_loader_list = []
    test_loader_list = []
    inter_matrix_list = []

    train_data_list = train_list

    for i in range(folds):
        Data = MyDataset(
            train_list[i],
            test_list[i],
            cpt_map,
            data_config["num_stu"],
            data_config["num_que"],
            data_config["num_cpt"]
        )

        train_loader, test_loader = Data.get_dataloader(batch_size)
        train_loader_list.append(train_loader)
        test_loader_list.append(test_loader)

        stus = test_list[i][:, 0].astype(int)
        exes = test_list[i][:, 1].astype(int)
        train_im = inter_matrix.copy()
        train_im[stus, exes] = -1
        inter_matrix_list.append(torch.tensor(train_im, dtype=torch.float32))

    return train_loader_list, test_loader_list, train_data_list, inter_matrix_list
