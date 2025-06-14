# -*- coding: utf-8 -*- 

import os, datetime, json
import random as python_random
from torch import nn
from torch.nn import Module, Embedding, Linear, Dropout, MaxPool1d, Sequential, ReLU
import copy, math, logging
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

device = "cpu" if not torch.cuda.is_available() else "cuda"


def set_seed(seed):
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e_s:
        print("Set seed failed, details are ", e_s)
        pass

    np.random.seed(seed)
    python_random.seed(seed)

    # cuda env
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def get_now_time():
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    return dt_string


def debug_print(text, log=False):

    print("="*20+f"{get_now_time()}: {text}"+"="*20)
    if log:
        logging.info("=" * 20 + f"{get_now_time()}: {text}" + "=" * 20)


def get_folds(data, fold_num):

    folds = [[] for _ in range(fold_num)]

    grouped = data.groupby('stu_id')

    for _, group in grouped:
        kf = KFold(n_splits=fold_num, shuffle=True, random_state=42)
        for fold_idx, (_, test_index) in enumerate(kf.split(group)):
            folds[fold_idx].append(group.iloc[test_index])

    folds = [pd.concat(fold).reset_index(drop=True) for fold in folds]

    train_list = []
    test_list = []

    for i in range(fold_num):
        test_list.append(np.array(folds[i]))
        train_list.append(np.array(pd.concat(folds[:i] + folds[i + 1:]).reset_index(drop=True)))

    return train_list, test_list

