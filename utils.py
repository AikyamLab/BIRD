from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torchmetrics

def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(
        sum(pred[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred[idx_s1_y1]) / sum(idx_s1_y1)
    )
    return parity.item(), equality.item()


def deo(pred, labels, sens):
    num_classes = len(np.unique(labels))
    num_sens_classes = len(np.unique(sens))

    parity_matrix, equality_matrix = [], []

    for i in range(num_classes):
        parities, equalities = [], []
        for j in range(num_sens_classes):
            idx_s = (sens == j).bool()
            idx_y = (labels == i).bool()
            idx_s_y = np.bitwise_and(idx_s, idx_y).bool()
            idx_not_s_y = np.bitwise_and(~idx_s, idx_y).bool()
            idx_not_s = ~idx_s

            parity = abs(
                (sum(pred[idx_s]) / sum(idx_s))
                - (sum(pred[idx_not_s]) / sum(idx_not_s))
            )
            equality = abs(
                (sum(pred[idx_s_y]) / sum(idx_s_y))
                - (sum(pred[idx_not_s_y]) / sum(idx_not_s_y))
            )

            # print(i, j, equality, parity)

            if not math.isnan(parity):
                parities.append(parity.item())
            if not math.isnan(equality):
                equalities.append(equality.item())

        if len(parities) > 0:
            parity_matrix.append(max(parities))
        if len(equalities) > 0:
            equality_matrix.append(max(equalities))

    if len(parity_matrix) == 0:
        parity_matrix = [0]

    if len(equality_matrix) == 0:
        equality_matrix = [0]

    return (
        np.mean(parity_matrix),
        max(parity_matrix),
        np.mean(equality_matrix),
        max(equality_matrix),
    )


def get_metrics(config, outputs, labels, prot_labels, get_acc_metrics=False):
    # calculates and returns deo_mean, deo_max, deo_poslabel, parity_mean, parity_max, parity_poslabel
    ## last batch might contain less number of samples
    if isinstance(labels, list):
        task = torch.cat([task for task in labels], dim=0)
    else:
        task = labels
    if isinstance(outputs, list):
        outputs = torch.cat([o for o in outputs], dim=0)

    assert task.shape[0] == outputs.shape[0]

    if isinstance(task, torch.Tensor):
        task = task.cpu()
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu()

    # accuracy
    _, predicted = torch.max(outputs.data, 1)
    total = task.size(0)
    correct = (predicted == task).sum().item()
    predicted = predicted.cpu().detach()

    # F1 and AUROC
    auroc_fn = torchmetrics.AUROC(task="multiclass", num_classes=config.num_task_classes)
    auroc = auroc_fn(outputs, task)
    
    if config.num_task_classes > 2:
        f1_fn = torchmetrics.F1Score(task="multiclass", num_classes=config.num_task_classes) # need to add binary explicitly
        f1 = f1_fn(outputs, task)
    else:
        f1_fn = torchmetrics.F1Score(task="binary", num_classes=config.num_task_classes) # need to add binary explicitly
        f1 = f1_fn(predicted, task)
    
    
    log_dict = {
        "acc": correct / total,
        "auroc": auroc.item() if isinstance(auroc, torch.Tensor) else auroc,
        "f1": f1.item() if isinstance(f1, torch.Tensor) else f1,
    }

    if get_acc_metrics:
        return log_dict

    if isinstance(prot_labels[0], list):
        sens = torch.cat([r for r in prot_labels[0]], dim=0)
    else:
        sens = prot_labels[0]


    parity_sens_mean, parity_sens_max, equality_sens_mean, equality_sens_max = deo(
        predicted, task, sens
    )

    # log wandb
    if config.dataset == "utk":
        fairness_metrics = {
            "parity/race_mean": parity_sens_mean,
            "parity/race_max": parity_sens_max,
            "equality/race_mean": equality_sens_mean,
            "equality/race_max": equality_sens_max,
        }
    
    elif config.dataset == "celeba":
        fairness_metrics = {
            "parity/gender_mean": parity_sens_mean,
            "parity/gender_max": parity_sens_max,
            "equality/gender_mean": equality_sens_mean,
            "equality/gender_max": equality_sens_max,
        }
    elif config.dataset == "cifar10s":
        fairness_metrics = {
            "parity/color_mean": parity_sens_mean,
            "parity/color_max": parity_sens_max,
            "equality/color_mean": equality_sens_mean,
            "equality/color_max": equality_sens_max,
        }

    log_dict.update(fairness_metrics)

    return log_dict


class EarlyStopping:
    def __init__(self, patience=5, delta=0.005, mode="min"):
        self.patience = patience
        self.counter = 0
        self.delta = delta
        self.best_score = None
        self.mode = mode

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score

        condition = False
        if self.mode == "min":
            condition = score > (self.best_score - self.delta)
        elif self.mode == "max":
            condition = score < (self.best_score + self.delta)

        if condition:  # if current loss is greater than best loss-0.005
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0

        return False


if __name__ == "__main__":
    loss = NST()
    x1 = torch.randn(2, 1024, 64, 64)
    x2 = torch.randn(2, 768, 64, 64)

    print(loss(x1, x2))
