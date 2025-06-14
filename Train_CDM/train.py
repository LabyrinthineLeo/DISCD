# -*- coding: utf-8 -*-

import os, sys

# sys.path.append("..")
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import argparse, logging
import json

import torch
from torch.optim import SGD, Adam

from Utils.utils import set_seed, debug_print, plot_training
from Utils.dataset import dataset4train
from Utils.init_model import init_model
from train_model import train_model
import datetime
import numpy as np

device = "cpu" if not torch.cuda.is_available() else "cuda"


def main(params):
    debug_print(text="load parameters.")
    if params['seed_true']:
        set_seed(params["seed"])
    model_name, dataset_name, fold, save_dir = params["model_name"], params["dataset_name"], params["folds"], params["save_dir"]
    gamma, lambda_1, lambda_2 = params['gamma'], params['lambda_1'], params['lambda_2']

    debug_print(text="load config infos.")

    train_config = {
        "batch_size": params["batch_size"],
        "num_epochs": params["epoch"],
        "optimizer": params["opt"],
        "lr_sch": params["lr_sch"]
    }

    batch_size, num_epochs, optimizer = train_config["batch_size"], train_config["num_epochs"], train_config['optimizer']

    config_path = os.path.abspath("../Configs/data_info.json")
    with open(config_path) as fin:
        data_config = json.load(fin)

    debug_print(text="print data infos")
    print(f"dataset_name:{dataset_name}, model_name:{model_name}")
    for i, j in data_config.items():
        print(f"{i}: {j}")
    print(f"fold:{fold}, batch_size:{batch_size}")

    debug_print(text="init_dataset")
    train_loader_list, test_loader_list, train_data_list, train_inter_matrix_list = dataset4train(dataset_name, data_config, batch_size)

    debug_print(text="create save dir")

    time_suffix = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    path_suffix = '-'.join([dataset_name, str(num_epochs), str(params['lr']), str(params['lambda_1']), str(params['lambda_2']),str(params['gamma'])])
    result_path = os.path.join(save_dir, model_name, path_suffix + f"_{time_suffix}")
    print("result_path:", result_path)
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    print("result_path:", result_path)
    print("log file:", os.path.join(result_path, 'log.txt'))

    try:
        logging.basicConfig(
            filename=os.path.join(result_path, 'log.txt'),
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            format='[%(asctime)s %(levelname)s] %(message)s',
        )

        print("Logging setup complete. Log file is %s", os.path.join(result_path, 'logs.txt'))
        logging.info("Logging setup complete. Log file is %s", os.path.join(result_path, 'logs.txt'))
    except Exception as e:
        print(f"Error setting up logging: {e}")

    debug_print("print params", True)
    for i, j in params.items():
        print(f"{i}: {j}")
        logging.info(f"{i}: {j}")
    print(f"{device}: {device}")
    logging.info(f"{device}: {device}")

    debug_print("training model", True)
    print(f"Start training model: {model_name}, \nsave_dir: {result_path}, \ndataset_name: {dataset_name}")
    logging.info(f"Start training model: {model_name}, \nsave_dir: {result_path}, \ndataset_name: {dataset_name}")
    print(f"train_config: {train_config}")
    logging.info(f"train_config: {train_config}")

    lr = params["lr"]

    debug_print("init model", True)

    print(f"model_name:{model_name}")
    logging.info(f"model_name:{model_name}")

    model = init_model(model_name, data_config[dataset_name], params, train_inter_matrix_list[0])

    if optimizer == "sgd":
        opt = SGD(model.parameters(), lr, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), lr)

    debug_print("train model", True)
    start_time = datetime.datetime.now()

    training_infos_list = []
    test_acc_list, test_auc_list, test_rmse_list, test_mae_list = [], [], [], []

    for i, (train_loader, test_loader) in enumerate(zip(train_loader_list, test_loader_list)):
        debug_print("training fold:{}".format(i))

        train_data = train_data_list[i]
        test_acc, test_auc, test_rmse, test_mae, training_infos = train_model(model, i, train_loader, test_loader, train_data, num_epochs, opt, result_path, (lambda_1, lambda_2), 50)

        debug_print("New Fold", True)
        model = init_model(model_name, data_config[dataset_name], params, train_inter_matrix_list[(i+1)%fold])

        if optimizer == "sgd":
            opt = SGD(model.parameters(), lr, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), lr)

        test_acc_list.append(test_acc)
        test_auc_list.append(test_auc)
        test_rmse_list.append(test_rmse)
        test_mae_list.append(test_mae)
        training_infos_list.append(training_infos)

    debug_print("acc list", True)
    print(["{:.5}".format(i) for i in test_acc_list])
    logging.info(["{:.5}".format(i) for i in test_acc_list])

    debug_print("auc list", True)
    print(["{:.5}".format(i) for i in test_auc_list])
    logging.info(["{:.5}".format(i) for i in test_auc_list])

    debug_print("rmse list", True)
    print(["{:.5}".format(i) for i in test_rmse_list])
    logging.info(["{:.5}".format(i) for i in test_rmse_list])

    debug_print("mae list", True)
    print(["{:.5}".format(i) for i in test_mae_list])
    logging.info(["{:.5}".format(i) for i in test_mae_list])

    debug_print("the mean and std of acc/auc/rmse/mae")

    print("acc mean:{:.5}, acc std:{:.5}".format(np.mean(test_acc_list), np.std(test_acc_list)))
    logging.info("acc mean:{:.5}, acc std:{:.5}".format(np.mean(test_acc_list), np.std(test_acc_list)))

    print("auc mean:{:.5}, auc std:{:.5}".format(np.mean(test_auc_list), np.std(test_auc_list)))
    logging.info("auc mean:{:.5}, auc std:{:.5}".format(np.mean(test_auc_list), np.std(test_auc_list)))

    print("rmse mean:{:.5}, rmse std:{:.5}".format(np.mean(test_rmse_list), np.std(test_rmse_list)))
    logging.info("rmse mean:{:.5}, rmse std:{:.5}".format(np.mean(test_rmse_list), np.std(test_rmse_list)))

    print("mae mean:{:.5}, mae std:{:.5}".format(np.mean(test_mae_list), np.std(test_mae_list)))
    logging.info("mae mean:{:.5}, mae std:{:.5}".format(np.mean(test_mae_list), np.std(test_mae_list)))

    print(f"start:{start_time.strftime('%Y-%m-%d-%H-%M-%S')}")
    logging.info(f"start:{start_time.strftime('%Y-%m-%d-%H-%M-%S')}")
    print(f"end:{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    logging.info(f"end:{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    end_time = datetime.datetime.now().timestamp()
    print(f"cost time:{(end_time - start_time.timestamp()) // 60} min")
    logging.info(f"cost time:{(end_time - start_time.timestamp()) // 60} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="nips-edu")
    parser.add_argument("--model_name", type=str, default="discd")
    parser.add_argument("--save_dir", type=str, default="../Results")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--seed_true", type=int, default=1)

    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--n_classes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--lambda_1", type=float, default=1e-5)
    parser.add_argument("--lambda_2", type=float, default=0.)

    parser.add_argument("--num_dim", type=int, default=256)

    args = parser.parse_args()

    params = vars(args)
    main(params)