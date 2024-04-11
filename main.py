from parser_ import get_args
import numpy as np
import random

# utils
config = get_args()

import os
from pathlib import Path
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from train_models import train, distill, test_final
from models import prepare_models
from models_cifar10s import prepare_models as prepare_models_cifar

from mmd import mfd, adv_fn
from metabird import Bird

import json

import pprint
from dataset import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


torch.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)

torch.cuda.manual_seed_all(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# experiment tracking
import wandb

Path(os.path.join(config.root, str(config.seed), config.log_dir)).mkdir(
    exist_ok=True, parents=True
)

wandb.init(
    dir=os.path.join(config.root, str(config.seed), config.log_dir),
    name=os.path.join(config.root, str(config.seed), config.log_dir),
)


def main(config):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)

    log_dir = os.path.join(config.root, str(config.seed), config.log_dir)
    Path(log_dir).mkdir(exist_ok=True, parents=True)

    # save config
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        print(config, file=f)

    # get dataloaders
    train_dataloader = get_dataloader(config, mode="train")
    test_dataloader = get_dataloader(config, mode="test")
    if config.meta2:
        quiz_dataloader = get_dataloader(config, mode="quiz")
        print(f"Quiz dataloader: {len(quiz_dataloader.dataset)}")
    else:
        quiz_dataloader = test_dataloader

    if config.datasetv2:
        held_dataloader = get_dataloader(config, mode="test")
    else:
        held_dataloader = get_dataloader(config, mode="held_out")
        
    print(f"Train dataloader: {len(train_dataloader.dataset)}")
    print(f"Test dataloader: {len(test_dataloader.dataset)}")
    print(f"Held dataloader: {len(held_dataloader.dataset)}")

    if (
        config.distill
        or config.AT
        or config.AT_only
        or config.fitnet_s1 or config.fitnet_s2
        or config.loss in ["adv", "mse", "kl", "mmd"]
        or config.meta2
        or config.fwd_loss
    ):
        try:
            assert config.pretrained_teacher != ""
        except:
            print(config.pretrained_teacher)
            raise ()

        if config.dataset == "cifar10s" and config.student in ["resnet18", "resnet34", "shufflenetv2"]:
            student = prepare_models_cifar(config, student=True)
            teacher = prepare_models_cifar(config, student=False)
        else:
            student = prepare_models(config, student=True)
            teacher = prepare_models(config, student=False)

        # student = nn.DataParallel(student)
        student.to(device)

        # load teacher weights
        state_dict = torch.load(config.pretrained_teacher, map_location=device)[
            "checkpoint"
        ]
        teacher = load_model(teacher, config, state_dict)
        print(f"Loaded Teacher Model from {config.pretrained_teacher}")
        
        # two stage fitnet or overfit
        if config.fitnet_s2:
            assert config.continue_student != ""
            student = load_model(student, config, torch.load(config.continue_student, map_location=device)["checkpoint"])

        if config.test != "":
            if config.meta2:
                Bird(
                    config,
                    student,
                    teacher,
                    train_dataloader,
                    held_dataloader,
                    test_dataloader,
                    quiz_dataloader,
                ).test()

            else:
                test_final(student, test_dataloader, config, config.test, mode="latest")
                test_final(student, test_dataloader, config, config.test, mode="best")

                
        elif config.meta2:
            Bird(
                config,
                student,
                teacher,
                train_dataloader,
                held_dataloader,
                test_dataloader,
                quiz_dataloader,
            )()

        elif config.loss in ["mse", "kl", "mmd"]:
            mfd(
                config,
                student,
                teacher,
                train_dataloader,
                quiz_dataloader,
                held_dataloader,
                test_dataloader,
            )

        elif config.loss == "adv":
            adv_fn(
                config,
                student,
                teacher,
                train_dataloader,
                quiz_dataloader,
                held_dataloader,
                test_dataloader,
            )

        else:
            distill(
                teacher,
                student,
                train_dataloader,
                held_dataloader,
                test_dataloader,
                config,
            )

    else:
        # vanilla training
        config.student = config.base
        config.teacher = config.base

        if config.dataset == "cifar10s" and config.base in ["resnet18", "resnet34", "shufflenetv2"]:
            model = prepare_models_cifar(config)
        else:
            model = prepare_models(config)
            
        # continue
        if config.continue_student != "":
            print(f"Continuing from ckpt {config.continue_student}")
            model = load_model(model, config, torch.load(config.continue_student, map_location=device)["checkpoint"])
        
        model.to(device)

        if config.test != "":
            test_final(model, test_dataloader, config, config.test, mode="latest")
        else:
            train(model, train_dataloader, held_dataloader, test_dataloader, config)


def load_model(teacher, config, state_dict):
    try:
        teacher.load_state_dict(state_dict, strict=False)
    except:
        corrected_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                corrected_state_dict[k[len("module.") :]] = v
            else:
                corrected_state_dict[k] = v

        teacher.load_state_dict(corrected_state_dict, strict=False)

    teacher.to(device)

    return teacher


if __name__ == "__main__":
    main(config)