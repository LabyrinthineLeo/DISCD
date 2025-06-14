# -*- coding: utf-8 -*-

import os, sys
import torch
from torch.nn import BCELoss
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy, mse_loss
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error
import logging
import torch.nn.utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_forward(model, data, params, idx):
    stu_inputs, que_inputs, labels, concepts = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
    labels = labels.float()
    concepts = concepts.float()

    pred_loss_fun = BCELoss()

    if model.model_name == 'discd':
        output, kl_loss, ib_loss = model(stu_inputs, que_inputs, concepts)
        loss = pred_loss_fun(output, labels) + params[0]*kl_loss + params[1]*ib_loss
        if idx >= 1:
            output_ = output

    return loss


def evaluate(model, test_data, params):
    model_name = model.model_name
    loss_fun = BCELoss()
    acc_list, auc_list, rmse_list, mae_list, loss_list = [], [], [], [], []

    with torch.no_grad():
        model.eval()
        for stu_inputs, que_inputs, labels, concepts in test_data:
            stu_inputs, que_inputs, labels, concepts = stu_inputs.to(device), que_inputs.to(device), labels.to(device), concepts.to(device)
            labels = labels.float()
            concepts = concepts.float()

            if model.model_name == 'discd':
                output, kl_loss, ib_loss = model(stu_inputs, que_inputs, concepts, False)
                loss = loss_fun(output, labels) + params[0]*kl_loss + params[1]*ib_loss

            loss_list.append(loss.detach().cpu().numpy())

            pred = output.detach().cpu().numpy()
            pred = np.where(pred >= 0.5, 1, 0)
            labels = labels.detach().cpu().numpy()

            acc = accuracy_score(labels, pred)

            unique_values = np.unique(labels)
            if len(unique_values) == 1:
                auc = 1.0 if unique_values[0] == 1 else 0.0
            else:
                auc = roc_auc_score(labels, pred)

            rmse = np.sqrt(mean_squared_error(labels, pred))
            mae = mean_absolute_error(labels, pred)

            acc_list.append(acc)
            auc_list.append(auc)
            rmse_list.append(rmse)
            mae_list.append(mae)

        model.train()

        acc_final = np.mean(acc_list)
        auc_final = np.mean(auc_list)
        rmse_final = np.mean(rmse_list)
        mae_final = np.mean(mae_list)
        loss_final = np.mean(loss_list)

    return acc_final, auc_final, rmse_final, mae_final, loss_final


def train_model(model, fold_idx, train_loader, test_loader, train_data, num_epochs, opt, result_path, lambdas, break_epoch=5):
    max_auc, max_acc = 0, 0
    min_rmse, min_mae = np.inf, np.inf
    best_epoch, best_step = 0, 0

    train_loss_list, test_loss_list = [], []
    train_acc_list, train_auc_list, train_rmse_list, train_mae_list = [], [], [], []
    test_acc_list, test_auc_list, test_rmse_list, test_mae_list = [], [], [], []

    train_step = 0
    save_model = True

    for epoch in range(1, num_epochs + 1):
        loss_mean = []

        for step, data in enumerate(train_loader):
            train_step += 1
            model.train()

            loss = model_forward(model, data, lambdas, step)

            opt.zero_grad()
            loss.backward()

            if model.model_name == "dcd":
                utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            opt.step()

            loss_mean.append(loss.detach().cpu().numpy())

        loss_mean = np.mean(loss_mean)
        train_loss_list.append(loss_mean)

        train_acc, train_auc, train_rmse, train_mae, _ = evaluate(model, train_loader, lambdas)
        train_acc_list.append(train_acc)
        train_auc_list.append(train_auc)
        train_rmse_list.append(train_rmse)
        train_mae_list.append(train_mae)
        test_acc, test_auc, test_rmse, test_mae, test_loss = evaluate(model, test_loader, lambdas)
        test_acc_list.append(test_acc)
        test_auc_list.append(test_auc)
        test_rmse_list.append(test_rmse)
        test_mae_list.append(test_mae)
        test_loss_list.append(test_loss)

        if test_auc >= max_auc and test_acc >= max_acc:

            if save_model:
                print("-" * 20+f"Epoch {epoch}, Updating Best Model!"+"-" * 20)
                logging.info("-" * 20+f"Epoch {epoch}, Updating Best Model!"+"-" * 20)
                torch.save(model.state_dict(), os.path.join(result_path, model.model_name + f"_best_fold{fold_idx}.ckpt"))
            max_auc = test_auc
            max_acc = test_acc
            min_rmse = test_rmse
            min_mae = test_mae
            best_epoch = epoch

        print("=" * 20 + f"Epoch {epoch}" + "=" * 20)
        logging.info("=" * 20 + f"Epoch {epoch}" + "=" * 20)
        print(f"Epoch: {epoch}, train_acc: {train_acc:.4}, train_auc: {train_auc:.4}, train_rmse: {train_rmse:.4}, train_mae: {train_mae:.4},"
              f"test_acc: {test_acc:.4}, test_auc: {test_auc:.4}, test_rmse: {test_rmse:.4}, test_mae: {test_mae:.4},"
              f"best epoch: {best_epoch}, best acc: {max_acc:.4}, best auc: {max_auc:.4},"
              f"best rmse: {min_rmse:.4}, best mae: {min_mae:.4},"
              f"train loss: {loss_mean:.5}, test loss: {test_loss:.5}.")
        logging.info(
            f"Epoch: {epoch}, train_acc: {train_acc:.4}, train_auc: {train_auc:.4}, train_rmse: {train_rmse:.4}, train_mae: {train_mae:.4},"
            f"test_acc: {test_acc:.4}, test_auc: {test_auc:.4}, test_rmse: {test_rmse:.4}, test_mae: {test_mae:.4},"
            f"best epoch: {best_epoch}, best acc: {max_acc:.4}, best auc: {max_auc:.4},"
            f"best rmse: {min_rmse:.4}, best mae: {min_mae:.4},"
            f"train loss: {loss_mean:.5}, test loss: {test_loss:.5}.")

        if epoch - best_epoch >= break_epoch:
            break

    cpt_path = os.path.join(result_path, model.model_name + f"_best_fold{fold_idx}.ckpt")
    cpt = torch.load(cpt_path)
    model.load_state_dict(cpt)
    test_acc, test_auc, test_rmse, test_mae, test_loss = evaluate(model, test_loader, lambdas)
    print(f"Final Testing, test_acc: {test_acc:.4}, test_auc: {test_auc:.4} "
          f"test_rmse: {test_rmse:.4}, test_mae: {test_mae:.4}.")
    logging.info(f"Final Testing, test_acc: {test_acc:.4}, test_auc: {test_auc:.4} "
          f"test_rmse: {test_rmse:.4}, test_mae: {test_mae:.4}.")

    training_info = [[train_acc_list, train_auc_list, train_rmse_list, train_mae_list, train_loss_list],
                     [test_acc_list, test_auc_list, test_rmse_list, test_mae_list, test_loss_list]]
    return test_acc, test_auc, test_rmse, test_mae, training_info