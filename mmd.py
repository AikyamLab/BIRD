import os
import torch
import torch.optim as optim
import torch.nn as nn

from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import pprint
from utils import get_metrics
import wandb
from train_models import test_final
from losses import (
    loss_fn_kd,
    AttentionTransfer,
    Hint,
    Losses,
)
from transformers import FlavaModel

pp = pprint.PrettyPrinter(indent=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import clip

def mfd(
    config,
    student,
    teacher,
    train_dataloader,
    quiz_dataloader,
    val_dataloader,
    test_dataloader,
):
    ############ MISC ################
    log_dir = os.path.join(config.root, str(config.seed), config.log_dir)
    

    if config.mmdv2:
        if config.dataset == "cifar10s":
            student_optimizer = optim.SGD(
                student.parameters(), lr=config.lr, weight_decay=5e-4, momentum=0.90
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                student_optimizer, [50, 80], gamma=0.1
            )
            
            # mmdv2_adam --> worse
            # student_optimizer = optim.Adam(
            #     student.parameters(), lr=config.lr, weight_decay=5e-4
            # )
            # scheduler = optim.lr_scheduler.MultiStepLR(
            #     student_optimizer, [50], gamma=0.1
            # )
            
        else:
            student_optimizer = optim.Adam(
                student.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9
            )
            if config.dataset == "celeba":
                scheduler = optim.lr_scheduler.StepLR(
                    student_optimizer, step_size=5, gamma=0.10, verbose=True
                )
            else:
                if config.student not in ["clip", "flava", "clip50"]:
                    scheduler = optim.lr_scheduler.MultiStepLR(
                        student_optimizer, [20, 40], gamma=0.50, verbose=True
                    )
                else:
                    scheduler = optim.lr_scheduler.StepLR(
                        student_optimizer, step_size=5, gamma=0.10, verbose=True
                    )

    store_path = os.path.join(log_dir, "checkpoints")
    Path(store_path).mkdir(exist_ok=True, parents=True)
    teacher.eval()
    teacher.requires_grad_(False)

    if config.AT:
        ofkd_loss = AttentionTransfer(
            mobilenet_student=config.student.startswith("mobile"), depth=config.depth
        )

    criterion = Losses(config)
    
    if config.vlm == "clip":
        clip_model, _ = clip.load("ViT-B/32", jit=False, device=device)
        clip_model.float()
    elif config.vlm == "clip50":
        clip_model, _ = clip.load("RN50", jit=False, device=device)
        clip_model.float()
    elif config.vlm == "flava":
        flava_model = FlavaModel.from_pretrained("facebook/flava-full").to(device)

    # early_stopping = EarlyStopping(patience=5, delta=0.001, mode="min")

    global_steps, val_steps, meta_steps = 0, 0, 0
    # best_loss = float("inf")
    best_auroc = float("-inf")
    best_fair = float("-inf")

    wandb.watch(student, log="all")

    train_metrics, val_metrics, meta_metrics = {}, {}, {}

    for epoch in range(config.epochs):
        print(f"EPOCH: {epoch}/{config.epochs}")
        student.train()

        for batch_idx, batch in tqdm(enumerate(train_dataloader)):

            student_optimizer.zero_grad()
            image, task, sens, _ = batch
            image, task = image.to(device), task.to(device)
        
            if config.vlm.startswith("clip"):
                if config.teacher.startswith("clip") and config.student.startswith("clip"):
                    with torch.no_grad():
                        image = clip_model.encode_image(image)
                        teacher_outputs, teacher_feat, _, _ = teacher(image)
                        
                    student_outputs, student_feat, _, _ = student(image)
                    
                elif config.teacher.startswith("clip"):
                    with torch.no_grad():
                        teacher_outputs, teacher_feat, _, _ = teacher(clip_model.encode_image(image))
                    student_outputs, student_feat, _, _ = student(image)
                
                elif config.student.startswith("clip"):
                    with torch.no_grad():
                        teacher_outputs, teacher_feat, _, _ = teacher(image)
                    student_outputs, student_feat, _, _ = student(clip_model.encode_image(image))                    
            
            elif config.vlm == "flava":
                if config.teacher=="flava" and config.student=="flava":
                    image = flava_model.get_image_features(image)
                    with torch.no_grad():
                        teacher_outputs, teacher_feat, _, _ = teacher(image)
                    student_outputs, student_feat, _, _ = student(image)
                    
                elif config.teacher=="flava":
                    with torch.no_grad():
                        teacher_outputs, teacher_feat, _, _ = teacher(flava_model.get_image_features(image))
                    student_outputs, student_feat, _, _ = student(image)
                    
                elif config.student=="flava":
                    with torch.no_grad():
                        teacher_outputs, teacher_feat, _, _ = teacher(image)
                    student_outputs, student_feat, _, _ = student(flava_model.get_image_features(image))
                
            else:
                with torch.no_grad():
                    teacher_outputs, teacher_feat, _, _ = teacher(image)
                student_outputs, student_feat, _, _ = student(image)        
                
            bias_loss = criterion(
                student_feat[-1], teacher_feat[-1].detach(), [task, sens]
            )

            # MFD + X
            if config.fitnet_s2:
                loss, _ = loss_fn_kd(student_outputs, task, teacher_outputs.detach(), config)
            
            elif config.AT:
                loss_kd, _ = loss_fn_kd(student_outputs, task, teacher_outputs.detach(), config)
                loss = (config.beta * ofkd_loss(student_feat, teacher_feat)) + loss_kd
                
            else:
                _, loss = loss_fn_kd(student_outputs, task, teacher_outputs.detach(), config)
                
            
            loss = bias_loss + loss

            loss.backward()
            student_optimizer.step()

            # log student performance
            new_metrics = get_metrics(
                config=config,
                outputs=student_outputs,
                labels=task,
                prot_labels=[sens],
                get_acc_metrics=True,
            )
            for k, v in new_metrics.items():
                train_metrics[k] = ((train_metrics.get(k, 0) * global_steps) + v) / (
                    global_steps + 1
                )

            train_metrics["loss"] = (
                (train_metrics.get("loss", 0) * global_steps)
                + (loss.item() - bias_loss.item())
            ) / (global_steps + 1)

            train_metrics["bias_loss"] = (
                (train_metrics.get("bias_loss", 0) * global_steps) + bias_loss.item()
            ) / (global_steps + 1)

            global_steps += 1

            wandb.log({"train": train_metrics})
            
            ## free memory ?
            del loss, bias_loss
            del student_outputs, teacher_outputs
            del student_feat, teacher_feat
            del image, task, sens
                        

        ##################################### VALIDATION LOOP #####################################
        student.eval()
        all_dict = {
            "outputs": [],
            "task": [],
            "sens": [],
            "loss": [],
            "bias_loss": [],
        }
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                image, task, sens, _ = batch
                image, task = image.to(device), task.to(device)                    
                
                if config.vlm.startswith("clip"):
                    if config.teacher.startswith("clip") and config.student.startswith("clip"):
                        teacher_outputs, teacher_feat, _, _ = teacher(clip_model.encode_image(image))
                        student_outputs, student_feat, _, _ = student(clip_model.encode_image(image))
                        
                    elif config.teacher.startswith("clip"):
                        teacher_outputs, teacher_feat, _, _ = teacher(clip_model.encode_image(image))
                        student_outputs, student_feat, _, _ = student(image)
                    
                    elif config.student.startswith("clip"):
                        teacher_outputs, teacher_feat, _, _ = teacher(image)
                        student_outputs, student_feat, _, _ = student(clip_model.encode_image(image))
                
                elif config.vlm == "flava":
                    if config.teacher=="flava" and config.student=="flava":
                        teacher_outputs, teacher_feat, _, _ = teacher(flava_model.get_image_features(image))
                        student_outputs, student_feat, _, _ = student(flava_model.get_image_features(image))
                        
                    elif config.teacher=="flava":
                        teacher_outputs, teacher_feat, _, _ = teacher(flava_model.get_image_features(image))
                        student_outputs, student_feat, _, _ = student(image)
                        
                    elif config.student=="flava":
                        teacher_outputs, teacher_feat, _, _ = teacher(image)
                        student_outputs, student_feat, _, _ = student(flava_model.get_image_features(image))
                    
                else:
                    teacher_outputs, teacher_feat, _, _ = teacher(image)
                    student_outputs, student_feat, _, _ = student(image)
                        
                        
                _, loss = loss_fn_kd(
                    student_outputs, task, teacher_outputs.detach(), config
                )

                val_loss = loss.item()

                bias_loss = criterion(
                    student_feat[-1], teacher_feat[-1].detach(), [task, sens]
                )
                bias_loss = bias_loss.item()

                all_dict["loss"].append(val_loss)
                all_dict["bias_loss"].append(bias_loss)
                all_dict["outputs"].append(student_outputs.detach())
                all_dict["task"].append(task.detach())
                all_dict["sens"].append(sens)

                val_steps += 1

            loss = sum(all_dict["loss"]) / len(all_dict["loss"])
            bias_loss = sum(all_dict["bias_loss"]) / len(all_dict["bias_loss"])

            val_metrics = get_metrics(
                config=config,
                outputs=torch.cat(all_dict["outputs"], dim=0),
                labels=torch.cat(all_dict["task"], dim=0),
                prot_labels=[
                    torch.cat(all_dict["sens"], dim=0),
                ],
            )
            val_metrics["loss"] = loss
            val_metrics["bias_loss"] = bias_loss

            wandb.log({"val": val_metrics})

            # save best student (max auroc)
            save_dict = {
                "checkpoint": student.state_dict(),
                "student_optimizer": student_optimizer.state_dict(),
                "epoch": epoch,
            }
            save_dict.update(val_metrics)

            # save best model
            if val_metrics["auroc"] > best_auroc:
                torch.save(save_dict, os.path.join(store_path, "best.pth"))
                best_auroc = val_metrics["auroc"]

            if (epoch + 1) % 50 == 0:
                torch.save(save_dict, os.path.join(store_path, f"{epoch+1}.pth"))

            # save latest model
            torch.save(save_dict, os.path.join(store_path, "latest.pth"))

            # scheduler.step(val_metrics["loss"])
            
            if config.mmdv2 is True and config.student not in ["clip", "clip50", "flava"]:
                scheduler.step()

            # if early_stopping(save_dict["loss"]) is True and epoch > 10:
            #     break  # stop training

    print("Testing...")
    ## Test and log final metrics
    test_final(student, test_dataloader, config, store_path, mode="best")
    test_final(student, test_dataloader, config, store_path, mode="latest")


# Adversarial Debiasing
# https://github.com/princetonvisualai/DomainBiasMitigation/blob/master/models/celeba_gradproj_adv.py
def adv_fn(
    config,
    student,
    teacher,
    train_dataloader,
    quiz_dataloader,
    val_dataloader,
    test_dataloader,
):
    print("Starting adversarial training")
    ############ MISC ################
    log_dir = os.path.join(config.root, str(config.seed), config.log_dir)
    
    base_parameters = [param for name, param in student.named_parameters() if "adv_classifiers" not in name]
    adv_parameters = [param for name, param in student.named_parameters() if "adv_classifiers" in name]

    base_optimizer = optim.Adam(base_parameters, lr=config.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=5e-4)
    adv_optimizer = optim.Adam(adv_parameters, lr=config.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=5e-4)
    
    if config.dataset == "celeba": # 10 epochs training
        step_size = [5]
        gamma = 0.1
    elif config.dataset == "utk": # 50 epochs training
        step_size = [5] if config.student in ["clip", "flava", "clip50"] else [20, 40]
        gamma = 0.1 if config.student in ["clip", "flava", "clip50"] else 0.5
    elif config.dataset == "cifar10s":
        base_optimizer = optim.SGD(base_parameters, lr=config.lr, momentum=0.90, weight_decay=5e-4)
        adv_optimizer = optim.SGD(adv_parameters, lr=config.lr, momentum=0.90, weight_decay=5e-4)
        step_size=[50, 80]
        gamma = 0.1
    
    base_scheduler = optim.lr_scheduler.MultiStepLR(base_optimizer, step_size, gamma=gamma, verbose=True)
    
    # if config.fitnet:
    #     hint_loss = Hint()

    store_path = os.path.join(log_dir, "checkpoints")
    Path(store_path).mkdir(exist_ok=True, parents=True)
    teacher.eval()
    teacher.requires_grad_(False)

    global_steps, val_steps = 0, 0
    best_auroc = float("-inf")
    best_fair = float("-inf")
    
    training_ratio = 1 if config.epochs < 20 else 3

    wandb.watch(student, log="all")
    
    if config.vlm == "clip":
        clip_model, _ = clip.load("ViT-B/32", jit=False, device=device)
        clip_model.float()
    elif config.vlm == "clip50":
        clip_model, _ = clip.load("RN50", jit=False, device=device)
        clip_model.float()
    elif config.vlm == "flava":
        flava_model = FlavaModel.from_pretrained("facebook/flava-full").to(device)

    train_metrics, val_metrics = {}, {}

    for epoch in range(config.epochs):
        print(f"EPOCH: {epoch}/{config.epochs}")
        student.train()

        for batch_idx, batch in tqdm(enumerate(train_dataloader)):
            base_optimizer.zero_grad()
            adv_optimizer.zero_grad()
            image, task, sens, _ = batch
            image, task, sens = (
                image.to(device),
                task.to(device),
                sens.to(device),
            )
            
            if config.vlm.startswith("clip"):
                with torch.no_grad():
                    embeds = clip_model.encode_image(image)
                if config.student.startswith("clip") and config.teacher.startswith("clip"):
                    student_outputs, student_feat, z_s, adv_output = student(embeds)
                    with torch.no_grad():
                        teacher_outputs, teacher_feat, z_t, _ = teacher(embeds)
                
                elif config.student.startswith("clip"):
                    student_outputs, student_feat, z_s, adv_output = student(embeds)
                    with torch.no_grad():
                        teacher_outputs, teacher_feat, z_t, _ = teacher(image)
                
                elif config.teacher.startswith("clip"):
                    student_outputs, student_feat, z_s, adv_output = student(image)
                    with torch.no_grad():
                        teacher_outputs, teacher_feat, z_t, _ = teacher(embeds)
            
            elif config.vlm.startswith("flava"):
                with torch.no_grad():
                    embeds = flava_model.get_image_features(image)
                if config.student.startswith("flava") and config.teacher.startswith("flava"):
                    student_outputs, student_feat, z_s, adv_output = student(embeds)
                    with torch.no_grad():
                        teacher_outputs, teacher_feat, z_t, _ = teacher(embeds)
                
                elif config.student.startswith("flava"):
                    student_outputs, student_feat, z_s, adv_output = student(embeds)
                    with torch.no_grad():
                        teacher_outputs, teacher_feat, z_t, _ = teacher(image)
                
                elif config.teacher.startswith("flava"):
                    student_outputs, student_feat, z_s, adv_output = student(image)
                    with torch.no_grad():
                        teacher_outputs, teacher_feat, z_t, _ = teacher(embeds)
            else:
                student_outputs, student_feat, z_s, adv_output = student(image)
                with torch.no_grad():
                    teacher_outputs, teacher_feat, z_t, _ = teacher(image)
                
            # task loss
            loss_kd, loss_ce = loss_fn_kd(
                student_outputs, task, teacher_outputs.detach(), config
            )

            # distillation loss (just use loss kd for experiments)
            if config.AT:
                loss_attn = ofkd_loss(student_feat, teacher_feat)
                loss = loss_kd + (config.beta * loss_attn)
            # elif config.fitnet:
            #     loss_attn = hint_loss(z_s, z_t.detach())
                # loss = (loss_attn * 0.1) + loss_ce  # from fitnet paper
            elif config.distill:
                loss = loss_kd
            else:
                loss = loss_ce

            # domain loss
            loss_sens = F.cross_entropy(adv_output[1], sens)
            

            # Update the main network
            if epoch % training_ratio == 0:
                class_params = [param for name, param in student.named_parameters() if "adv_classifiers" not in name]
                grad_from_class = torch.autograd.grad(loss, class_params, retain_graph=True, allow_unused=True)
                class_params = [param for name, param in student.named_parameters() if "adv_classifiers" not in name]
                grad_from_domain = torch.autograd.grad(loss_sens, class_params, retain_graph=True, allow_unused=True)
    
                for param, class_grad, domain_grad in zip(class_params, grad_from_class, grad_from_domain):
                    if (class_grad is not None) and (domain_grad is not None): 
                        # Gradient projection
                        if domain_grad.norm() > 1e-5:
                            param.grad = class_grad - config.lambda_*domain_grad - \
                                    ((class_grad*domain_grad).sum()/domain_grad.norm()) \
                                    * (domain_grad/domain_grad.norm()) 
                        else:
                            param.grad = class_grad - config.lambda_*domain_grad 
                
                base_optimizer.step()
            
            # Update the domain classifier
            domain_params = [param for name, param in student.named_parameters() if "adv_classifiers" in name]
            domain_grad = torch.autograd.grad(loss_sens, domain_params, retain_graph=True, allow_unused=True)
            for param, grad in zip(domain_params, domain_grad):
                param.grad = grad
                
            adv_optimizer.step()

            
            # log student performance
            new_metrics = get_metrics(
                config=config,
                outputs=student_outputs,
                labels=task,
                prot_labels=[sens],
                get_acc_metrics=True,
            )
            for k, v in new_metrics.items():
                train_metrics[k] = ((train_metrics.get(k, 0) * global_steps) + v) / (
                    global_steps + 1
                )

            train_metrics["loss_task"] = (
                (train_metrics.get("loss_task", 0) * global_steps)
                + loss.item()
            ) / (global_steps + 1)

            train_metrics["loss_sens"] = (
                (train_metrics.get("loss_sens", 0) * global_steps) + loss_sens.item()
            ) / (global_steps + 1)
                

            global_steps += 1

            wandb.log({"train": train_metrics})

        # if config.student not in ["clip", "clip50", "flava"]:
        if epoch%training_ratio==0:
            base_scheduler.step()
            
        # no scheduler for adv
                    
        
        ##################################### VALIDATION LOOP #####################################
        student.eval()
        all_dict = {
            "outputs": [],
            "task": [],
            "sens": [],
            "loss_task": [],
            "loss_sens": [],
        }
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                image, task, sens, _ = batch
                image, task, sens = (
                    image.to(device),
                    task.to(device),
                    sens.to(device),
                )
                    
                if config.vlm.startswith("clip"):
                    embeds = clip_model.encode_image(image)
                    if config.student.startswith("clip") and config.teacher.startswith("clip"):
                        student_outputs, student_feat, z_s, adv_output = student(embeds)
                        teacher_outputs, teacher_feat, z_t, _ = teacher(embeds)
                    
                    elif config.student.startswith("clip"):
                        student_outputs, student_feat, z_s, adv_output = student(embeds)
                        teacher_outputs, teacher_feat, z_t, _ = teacher(image)
                    
                    elif config.teacher.startswith("clip"):
                        student_outputs, student_feat, z_s, adv_output = student(image)
                        teacher_outputs, teacher_feat, z_t, _ = teacher(embeds)
                
                elif config.vlm.startswith("flava"):
                    embeds = flava_model.get_image_features(image)
                    if config.student.startswith("flava") and config.teacher.startswith("flava"):
                        student_outputs, student_feat, z_s, adv_output = student(embeds)
                        teacher_outputs, teacher_feat, z_t, _ = teacher(embeds)
                    
                    elif config.student.startswith("flava"):
                        student_outputs, student_feat, z_s, adv_output = student(embeds)
                        teacher_outputs, teacher_feat, z_t, _ = teacher(image)
                    
                    elif config.teacher.startswith("flava"):
                        student_outputs, student_feat, z_s, adv_output = student(image)
                        teacher_outputs, teacher_feat, z_t, _ = teacher(embeds)
                else:
                    student_outputs, student_feat, z_s, adv_output = student(image)
                    teacher_outputs, teacher_feat, z_t, _ = teacher(image)
                

                loss_kd, loss_ce = loss_fn_kd(
                    student_outputs, task, teacher_outputs.detach(), config
                )

                if config.AT:
                    loss_attn = ofkd_loss(student_feat, teacher_feat)
                    loss = loss_kd + (config.beta * loss_attn)
                # elif config.fitnet:
                #     loss_attn = hint_loss(z_s, z_t.detach())
                    # loss = (loss_attn * 0.1) + loss_ce  # from fitnet paper
                elif config.distill:
                    loss = loss_kd
                else:
                    loss = loss_ce

                val_loss = loss.item()
                loss_sens = F.cross_entropy(adv_output[1], sens)

                all_dict["loss_task"].append(val_loss)
                all_dict["outputs"].append(student_outputs)
                all_dict["task"].append(task.cpu())
                all_dict["sens"].append(sens.cpu())
                all_dict["loss_sens"].append(loss_sens.item())

                val_steps += 1

            val_metrics = get_metrics(
                config=config,
                outputs=torch.cat(all_dict["outputs"], dim=0),
                labels=torch.cat(all_dict["task"], dim=0),
                prot_labels=[
                    torch.cat(all_dict["sens"], dim=0),
                ],
            )
            
            val_metrics["loss"] = sum(all_dict["loss_task"]) / len(all_dict["loss_task"])
            val_metrics["loss_sens"] = sum(all_dict["loss_sens"]) / len(all_dict["loss_sens"])

            wandb.log({"val": val_metrics})

            # save best student (max auroc)
            save_dict = {
                "checkpoint": student.state_dict(),
                "base_optimizer": base_optimizer.state_dict(),
                "adv_optimizer": adv_optimizer.state_dict(),
                "epoch": epoch,
            }
            save_dict.update(val_metrics)

            # save best model
            if val_metrics["auroc"] > best_auroc:
                torch.save(save_dict, os.path.join(store_path, "best.pth"))
                best_auroc = val_metrics["auroc"]

            # save latest model
            torch.save(save_dict, os.path.join(store_path, "latest.pth"))

            if (epoch + 1) % 50 == 0:
                torch.save(save_dict, os.path.join(store_path, f"{epoch+1}.pth"))


    print("Testing...")
    ## Test and log final metrics
    test_final(student, test_dataloader, config, store_path, mode="best")
    test_final(student, test_dataloader, config, store_path, mode="latest")

