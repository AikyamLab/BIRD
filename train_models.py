import os
import json
import torch
import torchvision
import numpy as np
import torch.optim as optim
import torch.nn as nn

from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import pprint
from utils import get_metrics
from losses import loss_fn_kd, AttentionTransfer, Hint, FairWithoutDemo
import wandb
import clip
from transformers import FlavaModel

import transformers
transformers.utils.move_cache()

pp = pprint.PrettyPrinter(indent=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_dataloader, val_dataloader, test_dataloader, config):

    log_dir = os.path.join(config.root, str(config.seed), config.log_dir)

    # make misc directories
    Path(os.path.join(log_dir, "checkpoints")).mkdir(exist_ok=True, parents=True)
    
    if config.dataset == "cifar10s":
        # optimizer, scheduler
        optimizer = optim.SGD(
            model.parameters(), lr=config.lr, momentum=0.90, weight_decay=5e-4
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, [50, 80], gamma=0.1
            # optimizer, [80, 120], gamma=0.1
        ) # from takd paper
    else:
        # optimizer, scheduler
        optimizer = optim.Adam(
            model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9
        )
        
        if config.student not in ["clip", "flava", "clip50"]:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.5
            ) # use scheduler for utk, training celeba only for 10 epochs


    if config.vlm == "clip":
        clip_model, _ = clip.load("ViT-B/32", jit=False, device=device)
        clip_model.float()
    elif config.vlm == "clip50":
        clip_model, _ = clip.load("RN50", jit=False, device=device)
        clip_model.float()
    elif config.vlm == "flava":
        flava_model = FlavaModel.from_pretrained("facebook/flava-full").to(device)
    else:
        clip_model = None

    # loss
    criterion = nn.CrossEntropyLoss()

    store_path = os.path.join(log_dir, "checkpoints")
    Path(store_path).mkdir(exist_ok=True, parents=True)

    global_steps = 0
    val_steps = 0
    # best_loss = float("inf")
    best_auroc = float("-inf")
    best_fair = float("-inf")

    pp = pprint.PrettyPrinter(indent=4)

    wandb.watch(model, log="all")

    train_metrics, val_metrics = {}, {}

    for epoch in range(config.epochs):
        # train run
        model.train()
        print(f"Epoch: {epoch}/{config.epochs}")

        for batch_idx, batch in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            image, task, sens, _ = batch

            image, task = image.to(device), task.to(device)            

            if config.base.startswith("clip"):
                with torch.no_grad():
                    image_embed = clip_model.encode_image(image)
                outputs, _, _, _ = model(image_embed)
                                
            elif config.base == "flava":
                with torch.no_grad():
                    image_embed = flava_model.get_image_features(image)
                outputs, _, _, _ = model(image_embed)
            else:
                outputs, _, _, _ = model(image)
                        
            # loss
            loss = criterion(outputs, task)
            loss.backward()
            optimizer.step()

            train_loss = loss.item()

            new_metrics = get_metrics(
                config=config,
                outputs=outputs,
                labels=task,
                prot_labels=[sens],
                get_acc_metrics=True,
            )
            for k, v in new_metrics.items():
                train_metrics[k] = ((train_metrics.get(k, 0) * global_steps) + v) / (
                    global_steps + 1
                )

            train_metrics["loss"] = (
                (train_metrics.get("loss", 0) * global_steps) + train_loss
            ) / (global_steps + 1)

            wandb.log({"train": train_metrics})

            global_steps += 1

        # val run
        all_dict = {"outputs": [], "task": [], "sens": [], "loss": []}
        with torch.no_grad():
            model.eval()

            for batch_idx, batch in tqdm(enumerate(val_dataloader)):
                image, task, sens, _ = batch
                image, task = image.to(device), task.to(device)

                if config.base.startswith("clip"):
                    with torch.no_grad():
                        image_embed = clip_model.encode_image(image)
                    outputs, _, _, _ = model(image_embed)
                elif config.base == "flava":
                    with torch.no_grad():
                        image_embed = flava_model.get_image_features(image)
                    outputs, _, _, _ = model(image_embed)
                else:
                    outputs, _, _, _ = model(image)

                # loss
                loss = criterion(outputs, task)
                val_loss = loss.item()

                all_dict["outputs"].append(outputs)
                all_dict["task"].append(task)
                all_dict["sens"].append(sens)
                all_dict["loss"].append(val_loss)
                val_steps += 1

        loss = sum(all_dict["loss"]) / len(all_dict["loss"])

        val_metrics = get_metrics(
            config=config,
            outputs=torch.cat(all_dict["outputs"], dim=0),
            labels=torch.cat(all_dict["task"], dim=0),
            prot_labels=[
                torch.cat(all_dict["sens"], dim=0),
            ],
        )
        val_metrics["loss"] = loss

        wandb.log({"val": val_metrics})
        log_dict = val_metrics

        average_auroc = log_dict["auroc"]

        if config.dataset == "cifar10s":
            scheduler.step()
        elif config.student not in ["clip", "flava", "clip50"]:
            scheduler.step()

        # save best model (min val loss)
        save_dict = {
            "checkpoint": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }

        for k, v in log_dict.items():
            save_dict[k] = v

        if (average_auroc) > best_auroc:
            torch.save(save_dict, os.path.join(store_path, "best.pth"))
            best_auroc = average_auroc

        if (epoch + 1) % 50 == 0:
            torch.save(save_dict, os.path.join(store_path, f"{epoch+1}.pth"))

        # save latest model
        torch.save(save_dict, os.path.join(store_path, "latest.pth"))


    ## Test and log final metrics
    test_final(model, test_dataloader, config, store_path, mode="best")
    test_final(model, test_dataloader, config, store_path, mode="latest")


def distill(
    teacher, student, train_dataloader, val_dataloader, test_dataloader, config
):
    log_dir = os.path.join(config.root, str(config.seed), config.log_dir)

    if config.dataset == "cifar10s":
        # optimizer, scheduler,
        optimizer = optim.SGD(
            student.parameters(), lr=config.lr, weight_decay=5e-4, momentum=0.90
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, [50, 80], gamma=0.1
        )

    else:
        # optimizer, scheduler
        optimizer = optim.Adam(
            student.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9
        )
        # v2
        if config.student not in ["clip", "flava"]:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.5
            ) # only for utk
    
    if config.vlm == "clip":
        clip_model, _ = clip.load("ViT-B/32", jit=False, device=device)
        clip_model.float()
    elif config.vlm == "clip50":
        clip_model, _ = clip.load("RN50", jit=False, device=device)
        clip_model.float()
    elif config.vlm == "flava":
        flava_model = FlavaModel.from_pretrained("facebook/flava-full").to(device)
    else:
        clip_model = None
        flava_model = None

    store_path = os.path.join(log_dir, "checkpoints")
    Path(store_path).mkdir(exist_ok=True, parents=True)
    teacher.eval()

    if config.AT or config.AT_only:
        criterion = AttentionTransfer(
            depth=config.depth, mobilenet_student=config.student.startswith("mobile")
        )
    elif config.fitnet_s1:
        criterion = Hint()

    # early_stopping = EarlyStopping(patience=5, delta=0.001, mode="min")

    global_steps, val_steps = 0, 0
    # best_loss = float("inf")
    best_auroc = float("-inf")
    best_fair = float("-inf")

    wandb.watch(student, log="all")

    train_metrics, val_metrics = {}, {}

    for epoch in range(config.epochs):
        # train run
        student.train()
        for batch_idx, batch in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            image, task, sens, _ = batch
            image, task = image.to(device), task.to(device)
            
            if config.vlm.startswith("clip"):
                if config.student.startswith("clip") and config.teacher.startswith("clip"):
                    with torch.no_grad():
                        image_embed = clip_model.encode_image(image)
                        teacher_outputs, teacher_feat, h_t, _ = teacher(image_embed)
                    outputs, student_feat, h_s, _ = student(image_embed) 
                
                elif config.teacher.startswith("clip"):
                    with torch.no_grad():
                        image_embed = clip_model.encode_image(image)
                        teacher_outputs, teacher_feat, h_t, _ = teacher(image_embed)
                    outputs, student_feat, h_s, _ = student(image)
                
                elif config.student.startswith("clip"):
                    with torch.no_grad():
                        image_embed = clip_model.encode_image(image)
                        teacher_outputs, teacher_feat, h_t, _ = teacher(image)
                    outputs, student_feat, h_s, _ = student(image_embed) 
                    
            elif config.vlm=="flava":
                if config.student == "flava" and config.teacher == "flava":
                    with torch.no_grad():
                        image_embed = flava_model.get_image_features(image)
                        teacher_outputs, teacher_feat, h_t, _ = teacher(image_embed)
                    outputs, student_feat, h_s, _ = student(image_embed)
                
                elif config.teacher == "flava":
                    with torch.no_grad():
                        image_embed = flava_model.get_image_features(image)
                        teacher_outputs, teacher_feat, h_t, _ = teacher(image_embed)
                    outputs, student_feat, h_s, _ = student(image)
                
                elif config.student == "flava":
                    with torch.no_grad():
                        image_embed = flava_model.get_image_features(image)
                        teacher_outputs, teacher_feat, h_t, _ = teacher(image)
                    outputs, student_feat, h_s, _ = student(image_embed)
            
            else:
                outputs, student_feat, h_s, _ = student(image)
                with torch.no_grad():
                    teacher_outputs, teacher_feat, h_t, _ = teacher(image)
                
            # loss
            if config.AT or config.AT_only:
                loss_attn = criterion(student_feat, teacher_feat)

                if config.AT_only:
                    loss = nn.CrossEntropyLoss()(outputs, task) + (config.beta * loss_attn)
                else:
                    loss_kd, loss_ce = loss_fn_kd(outputs, task, teacher_outputs, config)
                    loss = (config.beta * loss_attn) + loss_kd

            elif config.fitnet_s1:
                loss_attn = criterion(h_s, h_t.detach())
                # fitnet hparam from paper: fixed
                # loss = nn.CrossEntropyLoss()(outputs, task) + (0.1 * loss_attn) ## incorrect implementation
                loss = loss_attn
            
            elif config.fitnet_s2:
                loss, loss_ce = loss_fn_kd(outputs, task, teacher_outputs, config)

            elif config.fwd_loss:
                loss = FairWithoutDemo(config)(outputs, teacher_outputs)
            
            else:
                loss, loss_ce = loss_fn_kd(outputs, task, teacher_outputs, config)

            loss.backward()

            optimizer.step()

            train_loss = loss.item()

            # do not calculate bias metrics running mean
            new_metrics = get_metrics(
                config=config,
                outputs=outputs,
                labels=task,
                prot_labels=[sens],
                get_acc_metrics=True,
            )
            for k, v in new_metrics.items():
                train_metrics[k] = ((train_metrics.get(k, 0) * global_steps) + v) / (
                    global_steps + 1
                )

            train_metrics["loss"] = (
                (train_metrics.get("loss", 0) * global_steps) + train_loss
            ) / (global_steps + 1)

            wandb.log({"train": train_metrics})

            global_steps += 1

        # val run
        with torch.no_grad():
            student.eval()
            all_dict = {
                "outputs": [],
                "task": [],
                "sens": [],
                "loss": [],
            }
            for batch_idx, batch in tqdm(enumerate(val_dataloader)):
                image, task, sens, _ = batch
                image, task = image.to(device), task.to(device)
                
                if config.vlm.startswith("clip"):
                    if config.student.startswith("clip") and config.teacher.startswith("clip"):
                        image_embed = clip_model.encode_image(image)
                        teacher_outputs, teacher_feat, h_t, _ = teacher(image_embed)
                        outputs, student_feat, h_s, _ = student(image_embed) 
                    
                    elif config.teacher.startswith("clip"):
                        image_embed = clip_model.encode_image(image)
                        teacher_outputs, teacher_feat, h_t, _ = teacher(image_embed)
                        outputs, student_feat, h_s, _ = student(image)
                    
                    elif config.student.startswith("clip"):
                        image_embed = clip_model.encode_image(image)
                        teacher_outputs, teacher_feat, h_t, _ = teacher(image)
                        outputs, student_feat, h_s, _ = student(image_embed) 
                        
                elif config.vlm=="flava":
                    if config.student == "flava" and config.teacher == "flava":
                        image_embed = flava_model.get_image_features(image)
                        teacher_outputs, teacher_feat, h_t, _ = teacher(image_embed)
                        outputs, student_feat, h_s, _ = student(image_embed)
                    
                    elif config.teacher == "flava":
                        image_embed = flava_model.get_image_features(image)
                        teacher_outputs, teacher_feat, h_t, _ = teacher(image_embed)
                        outputs, student_feat, h_s, _ = student(image)
                    
                    elif config.student == "flava":
                        image_embed = flava_model.get_image_features(image)
                        teacher_outputs, teacher_feat, h_t, _ = teacher(image)
                        outputs, student_feat, h_s, _ = student(image_embed)
                
                else:
                    outputs, student_feat, h_s, _ = student(image)
                    teacher_outputs, teacher_feat, h_t, _ = teacher(image)
                    
                        
                # loss
                if config.AT or config.AT_only:
                    if config.AT_only:
                        loss_ce = nn.CrossEntropyLoss()(outputs, task)
                        loss = loss_ce * (1.0 - config.alpha) + (
                            config.alpha * loss_attn
                        )

                    else:
                        loss_attn = criterion(student_feat, teacher_feat)
                        loss_kd, loss_ce = loss_fn_kd(
                            outputs, task, teacher_outputs, config
                        )
                        loss = (config.beta * loss_attn) + loss_kd

                elif config.fitnet_s1:
                    loss_attn = criterion(h_s, h_t.detach())
                    loss = loss_attn
                    
                elif config.fitnet_s2:
                    loss, loss_ce = loss_fn_kd(outputs, task, teacher_outputs, config)
                
                elif config.fwd_loss:
                    loss = FairWithoutDemo(config)(outputs, teacher_outputs)    
                
                else:
                    loss, loss_ce = loss_fn_kd(outputs, task, teacher_outputs, config)

                val_loss = loss.item()

                val_steps += 1

                all_dict["outputs"].append(outputs)
                all_dict["task"].append(task)
                all_dict["sens"].append(sens)
                all_dict["loss"].append(val_loss)

        loss = sum(all_dict["loss"]) / len(all_dict["loss"])
        val_metrics = get_metrics(
            config=config,
            outputs=torch.cat(all_dict["outputs"], dim=0),
            labels=torch.cat(all_dict["task"], dim=0),
            prot_labels=[
                torch.cat(all_dict["sens"], dim=0),
            ],
        )
        val_metrics["loss"] = loss

        wandb.log({"val": val_metrics})
        log_dict = val_metrics

        average_auroc = log_dict["auroc"]
        
        if config.dataset == "cifar10s":
            scheduler.step()
        elif config.student not in ["clip", "flava"]:
            scheduler.step()

        # save best student (min val loss)
        save_dict = {
            "checkpoint": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        for k, v in log_dict.items():
            save_dict[k] = v

        if average_auroc > best_auroc:
            torch.save(save_dict, os.path.join(store_path, "best.pth"))
            best_auroc = average_auroc

        if (epoch + 1) % 50 == 0:
            torch.save(save_dict, os.path.join(store_path, f"{epoch+1}.pth"))

        # save latest model
        torch.save(save_dict, os.path.join(store_path, "latest.pth"))


    print("Testing")
    ## Test and log final metrics
    test_final(student, test_dataloader, config, store_path, mode="best")
    test_final(student, test_dataloader, config, store_path, mode="latest")



def test_final(model, test_dataloader, config, store_path, mode="best"):
    # load best held out model
    state_dict = torch.load(os.path.join(store_path, f"{mode}.pth"))["checkpoint"]
    criterion = nn.CrossEntropyLoss()
    
    if config.vlm == "clip":
        clip_model, _ = clip.load("ViT-B/32", jit=False, device=device)
        clip_model.float()
        clip_model.eval()
    elif config.vlm == "clip50":
        clip_model, _ = clip.load("RN50", jit=False, device=device)
        clip_model.float()
        clip_model.eval()
    elif config.vlm == "flava":
        flava_model = FlavaModel.from_pretrained("facebook/flava-full").to(device)
        flava_model.eval()
    else:
        clip_model = None
        flava_model = None

    try:
        model.load_state_dict(state_dict)
    except:
        corrected_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                corrected_state_dict[k[len("module.") :]] = v
            else:
                corrected_state_dict[k] = v

        model.load_state_dict(corrected_state_dict)
    
    model.to(device)

    with torch.no_grad():
        model.eval()

        # test run
        losses = []
        all_dict = {"outputs": [], "task": [], "sens": [], "loss": []}

        for batch_idx, batch in tqdm(enumerate(test_dataloader)):
            image, task, sens, _ = batch
            image, task = image.to(device), task.to(device)

            if config.student.startswith("clip"):
                image_embed = clip_model.encode_image(image)
                outputs, _, _, _ = model(image_embed)
            elif config.student == "flava":
                image_embed = flava_model.get_image_features(image)
                outputs, _, _, _ = model(image_embed)
            else:
                outputs, _, _, _ = model(image)

            # loss - ce only
            loss = criterion(outputs, task)

            all_dict["outputs"].append(outputs)
            all_dict["task"].append(task)
            all_dict["sens"].append(sens)
            all_dict["loss"].append(loss.item())

    ######## DO NOT TAKE RUNNING MEAN OF BIAS METRICS ########
    loss = sum(all_dict["loss"]) / len(all_dict["loss"])
    metrics = get_metrics(
        config=config,
        outputs=torch.cat(all_dict["outputs"], dim=0),
        labels=torch.cat(all_dict["task"], dim=0),
        prot_labels=[
            torch.cat(all_dict["sens"], dim=0),
        ],
    )
    metrics["loss"] = loss

    wandb.log({f"test_{mode}": metrics})

    json_path = "/".join(store_path.split("/")[:-1])

    try:
        with open(os.path.join(json_path, f"results_{mode}.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        pp.pprint(metrics)

    except:
        pp.pprint(metrics)
