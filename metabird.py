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
import wandb
from train_models import test_final
from collections import OrderedDict
from copy import deepcopy
from losses import Losses, AttentionTransfer, loss_fn_kd, equalized_odds_loss
import higher
from channel_prune import ChannelPruningv2
import clip
import time
from transformers import FlavaModel

from flopth import flopth

pp = pprint.PrettyPrinter(indent=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetaLearner(nn.Module):
    def __init__(self, config):
        super(MetaLearner, self).__init__()
        self.config = config

        if config.vlm in ["clip", "flava", "clip50"]:
            size = 512
        elif config.student == "cnn":
            size = 1024
        else:
            with open("feat_shapes200.json", "r") as f:
                size = json.load(f)[config.teacher][-1][1]
        
        self.phi = ChannelPruningv2(config, size)
        self.phi.requires_grad_(True)

    def transform(self, teacher_feat):
        tf = self.phi(teacher_feat)
        return tf

    def forward(self, teacher_feat):
        self.transform(teacher_feat)


class Bird:
    def __init__(
        self,
        config,
        student,
        teacher,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        quiz_dataloader,
    ):
        super(Bird, self).__init__()
        config.loss = "kl"  ## tune
        self.config = config
        self.student = student
        self.teacher = teacher
        self.meta_learner = MetaLearner(config)
        self.log_dir = os.path.join(config.root, str(config.seed), config.log_dir)
        self.store_path = os.path.join(self.log_dir, "checkpoints")
        Path(self.store_path).mkdir(exist_ok=True, parents=True)

        self.teacher.eval()
        self.teacher.requires_grad_(False)
        
        self.student.requires_grad_(True)
        
        self.meta_learner.requires_grad_(True)
        
        if config.meta_mmd:
            config.loss = "mmd"
            self.mmd_criterion = Losses(config)
        
        if config.AT:
            self.at_loss = AttentionTransfer(
            depth=config.depth, mobilenet_student=config.student.startswith("mobile")
        )
        
        # adam doesnt work
        self.optimizer = optim.SGD([p for p in self.student.parameters() if p.requires_grad is True], lr=config.lr, momentum=0.9, weight_decay=5e-4)
        
        if config.meta_adv:
            self.optimzer = optim.SGD([p for n, p in self.student.named_parameters() if p.requires_grad is True and "adv_classifiers" not in n], lr=config.lr, momentum=0.9, weight_decay=5e-4)
            self.adv_optimizer = optim.SGD([p for n, p in self.student.named_parameters() if p.requires_grad is True and "adv_classifiers" in n], lr=config.lr, momentum=0.9, weight_decay=5e-4)
        
        self.meta_optimizer = optim.AdamW(
            self.meta_learner.parameters(),
            lr=5e-2 if config.dataset == "cifar10s" else config.lr,
            weight_decay=5e-2,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        if config.dataset == "cifar10s":
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, [50, 80], gamma=0.1
            ) 
        else:
            if self.config.student == "flava":
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=5, gamma=0.5
                )
            else:
                step_size = [5] if config.dataset == "celeba" else [10]
                
                # inter architecture ablation
                if config.student == "resnet18" and config.teacher == "clip" and config.dataset == "celeba":
                    step_size = [4, 8]
                    
                self.scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer, step_size, gamma=0.1
                )
        
        self.global_steps, self.val_steps = 0, 0
        self.best_auroc = float("-inf")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.meta_learner.to(device)

        # datasets
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.quiz_dataloader = quiz_dataloader
        self.val_dataloader = val_dataloader
        
        if self.config.vlm == "clip":
            self.clip_model, _ = clip.load("ViT-B/32", jit=False, device=device)
            self.clip_model.float()
        elif self.config.vlm == "clip50":
            self.clip_model, _ = clip.load("RN50", jit=False, device=device)
            self.clip_model.float()
        elif self.config.vlm == "flava":
            self.flava_model = FlavaModel.from_pretrained("facebook/flava-full").to(device)
        else:
            self.clip_model = None
            self.flava_model = None

        wandb.watch(self.student, log="all")
        wandb.watch(self.meta_learner, log="all")

        self.train_metrics, self.val_metrics = {}, {}
        
        if self.config.get_complexity:
            flops_student, params_student = flopth(self.student, in_size=(3, 200, 200))
            flops_teacher, params_teacher = flopth(self.teacher, in_size=(3, 200, 200))
            
            print("FLOPS:", flops_student, flops_teacher)
            print("Params:", params_student, params_teacher)
            
            import ipdb;ipdb.set_trace()
            exit()
            

    def __call__(self):
        for epoch in tqdm(range(self.config.epochs)):
            self.train(epoch)
            self.val(epoch)

            if not self.config.student.startswith("clip"):
                self.scheduler.step()  # steplr
        
        print("[INFO] Testing")
        test_final(self.student, self.test_dataloader, self.config, self.store_path, mode="best")
        test_final(self.student, self.test_dataloader, self.config, self.store_path, mode="latest")

    def test(self):
        test_final(self.student, self.test_dataloader, self.config, self.config.test, mode="best")
        test_final(self.student, self.test_dataloader, self.config, self.config.test, mode="latest")

    def train(self, epoch):
        self.student.train()
        self.meta_optimizer.zero_grad()
 
        for batch_idx, batch in tqdm(enumerate(self.train_dataloader)):   
            self.meta_learner.eval()

            if epoch >= self.config.later:
                self.meta_learner.train()
                ## v5 1e-3 --> v6 1e-2
                if self.config.inner_lr > 0:
                    inner_opt = optim.SGD(self.student.parameters(), lr=self.config.inner_lr, momentum=0.9, weight_decay=5e-4)
                else:
                    inner_opt = self.optimizer # repro

                # fmodel = higher.patch.monkeypatch(self.student, copy_initial_weights=False, track_higher_grads=True)
                # diffopt = higher.optim.get_diff_optim(inner_opt, self.student.parameters(), fmodel=fmodel)
                        
                with higher.innerloop_ctx(
                    self.student, inner_opt, track_higher_grads=True, copy_initial_weights=False
                ) as (fmodel, diffopt):
                    
                    image, task, sens, _ = self.to_device(batch)
                    
                    if self.config.student == self.config.teacher:
                        image = self.get_features(image, s=False)
                        teacher_outputs, teacher_feat, z_t, _ = self.teacher(image)
                        student_outputs, student_feat, z_s, adv_outputs = fmodel(image)
                    else:
                        teacher_outputs, teacher_feat, z_t, _ = self.teacher(self.get_features(image, s=False))
                        student_outputs, student_feat, z_s, adv_outputs = fmodel(self.get_features(image, s=True))
                    
                    # do kd with transformed teacher feat
                    loss_ce = F.cross_entropy(student_outputs, task)
                    loss_kd = nn.MSELoss()(z_s, self.meta_learner.transform(z_t))
                    loss = ((1-self.config.alpha) * loss_ce) + (self.config.alpha * loss_kd)
                    diffopt.step(loss)

                    # wandb.log({"train_inner": loss.item()}, commit=False)
                    
                    for meta_batch_idx, meta_batch in enumerate(self.quiz_dataloader):
                        meta_image, meta_task, meta_sens, _ = self.to_device(meta_batch)
                            
                        student_outputs, _, _, _ = fmodel(self.get_features(meta_image, s=True))
                        
                        bias_feedback = equalized_odds_loss(student_outputs, meta_task, meta_sens, self.config)

                        # wandb.log(
                        #     {"train_outer": {"loss_bias": bias_feedback.item()}},
                        #     commit=False,
                        # )

                        # use just a single batch from the quiz dataset per iteration (memory constraints)
                        break

                    # calculate gradients of bias_feedback wrt meta_learner parameters
                    phi_grads = torch.autograd.grad(
                        bias_feedback, self.meta_learner.parameters()
                    )

                    for p, g in zip(self.meta_learner.parameters(), phi_grads):
                        assert g is not None
                        p.grad = g
                        

                # update meta parameters using bias loss
                self.meta_optimizer.step()
            
                if not self.config.sigmoid:
                    self.meta_learner.phi.param.data = torch.clamp(
                        self.meta_learner.phi.param.data, 0, 1
                    )
            
            # update student
            self.optimizer.zero_grad()
            self.meta_learner.eval()

            image, task, sens, _ = self.to_device(batch)
            
            if self.config.student == self.config.teacher:
                image = self.get_features(image, s=False)
                teacher_outputs, teacher_feat, z_t, _ = self.teacher(image) 
                student_outputs, student_feat, z_s, adv_output = self.student(image)
            else:
                teacher_outputs, teacher_feat, z_t, _ = self.teacher(self.get_features(image, s=False)) 
                student_outputs, student_feat, z_s, adv_output = self.student(self.get_features(image, s=True))

            loss_ce = F.cross_entropy(student_outputs, task)
            
            if self.config.eq_odds_ab:
                loss_kd = nn.MSELoss()(z_s, z_t)
                
            else:
                loss_kd = nn.MSELoss()(z_s, self.meta_learner.transform(z_t))
            
            if self.config.dataset == "utk":
                if self.config.student in ["flava"]:
                    loss = (0.60*loss_ce) + (0.40*loss_kd) # bird4
                elif self.config.student in ["clip50"]:
                    loss = (0.70*loss_ce) + (0.30*loss_kd)
                elif self.config.teacher in ["clip50", "clip"] and self.config.student == "resnet18":
                    loss = (0.70*loss_ce) + (0.30*loss_kd) # bird 3
                else:
                    loss = (0.1 * loss_ce) + (0.9 * loss_kd)
            else:
                # default 0.1, 0.9
                loss = ((1-self.config.alpha)*loss_ce) + (self.config.alpha*loss_kd)
            
            if self.config.meta_mmd: # mmd
                loss_mmd = loss_ce + self.mmd_criterion(student_feat[-1], teacher_feat[-1].detach(), [task, sens])
                loss = loss + loss_mmd
                
            else: # ours
                if self.config.student == "shufflenetv2":
                    if self.config.dataset == "utk":
                        loss = loss + ((0.80*loss_ce) + equalized_odds_loss(student_outputs, task, sens, self.config)*0.20)
                    elif self.config.dataset == "celeba":
                        loss = loss + (equalized_odds_loss(student_outputs, task, sens, self.config)*0.05)
                    else:
                        raise(NotImplementedError)

                elif self.config.student == "resnet18":
                    if self.config.teacher in ["clip", "clip50", "flava"]:
                        loss = loss + (equalized_odds_loss(student_outputs, task, sens, self.config)*0.10)
                    else:
                        loss = loss + (equalized_odds_loss(student_outputs, task, sens, self.config)*0.20) 
                elif self.config.student == "resnet34":
                    loss = loss + (equalized_odds_loss(student_outputs, task, sens, self.config)*0.10)
                elif self.config.student in ["clip"]:
                    loss = loss + (equalized_odds_loss(student_outputs, task, sens, self.config)*0.20)
                elif self.config.student in ["flava", "clip50"]:
                    loss = loss + (equalized_odds_loss(student_outputs, task, sens, self.config)*0.10)
                
            # augment with AT loss
            if self.config.AT: # eq_odds + AT
                loss_attn = self.at_loss(student_feat, teacher_feat)
                loss = loss + (loss_attn*self.config.beta)
            
            if self.config.fitnet_s2:
                loss_fit = loss_fn_kd(student_outputs, task, teacher_outputs, self.config)

            if not self.config.meta_adv:
                grads = torch.autograd.grad(
                    loss, [p for p in self.student.parameters() if p.requires_grad is True], allow_unused=True
                )

                for p, g in zip(self.student.parameters(), grads):
                    p.grad = g

                self.optimizer.step()

            ## logging
            new_metrics = get_metrics(
                config=self.config,
                outputs=student_outputs,
                labels=task,  # task
                prot_labels=[sens],  # sensitive
                get_acc_metrics=True,
            )

            for k, v in new_metrics.items():
                self.train_metrics[k] = (
                    (self.train_metrics.get(k, 0) * self.global_steps) + v
                ) / (self.global_steps + 1)

            self.train_metrics["loss"] = (
                (self.train_metrics.get("loss", 0) * self.global_steps) + loss.item()
            ) / (self.global_steps + 1)

            self.global_steps += 1
            wandb.log({"train": self.train_metrics})

    def to_device(self, batch):
        return [i.to(self.device) for i in batch]

    def get_features(self, image, s=True):
        with torch.no_grad():
            check = self.config.student if s else self.config.teacher
            
            if check.startswith("clip"):
                image = self.clip_model.encode_image(image)
            elif check == "flava":
                image = self.flava_model.get_image_features(image)
        
        return image

    def val(self, epoch):
        # simply validate student model
        self.student.eval()

        all_dict = {
            "outputs": [],
            "task": [],
            "sens": [],
            "loss_task": [],
            "loss_sens": [],
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                image, task, sens, _ = self.to_device(batch)
                                    
                student_outputs, student_feat, z_s, adv_outputs = self.student(self.get_features(image, s=True))

                loss_ce = F.cross_entropy(student_outputs, task).item()
                loss_sens = F.cross_entropy(adv_outputs[1], sens).item()

                all_dict["loss_task"].append(loss_ce)
                all_dict["loss_sens"].append(loss_sens)
                all_dict["outputs"].append(student_outputs.detach().cpu())
                all_dict["task"].append(task.detach().cpu())
                all_dict["sens"].append(sens.detach().cpu())

                self.val_steps += 1

        loss_task = sum(all_dict["loss_task"]) / len(all_dict["loss_task"])
        loss_sens = sum(all_dict["loss_sens"]) / len(all_dict["loss_sens"])

        val_metrics = get_metrics(
            config=self.config,
            outputs=torch.cat(all_dict["outputs"], dim=0),
            labels=torch.cat(all_dict["task"], dim=0),
            prot_labels=[
                torch.cat(all_dict["sens"], dim=0),
            ],
        )
        val_metrics["loss_task"] = loss_task
        val_metrics["loss_sens"] = loss_sens

        wandb.log({"val": val_metrics})

        # save best student (max auroc)
        save_dict = {
            "checkpoint": self.student.state_dict(),
            "phi": self.meta_learner.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
        }
        save_dict.update(val_metrics)
        
        # save best model
        if val_metrics["auroc"] > self.best_auroc:
            torch.save(save_dict, os.path.join(self.store_path, "best.pth"))
            self.best_auroc = val_metrics["auroc"]

        if (epoch + 1) % 50 == 0:
            torch.save(save_dict, os.path.join(self.store_path, f"{epoch+1}.pth"))

        # save latest model
        torch.save(save_dict, os.path.join(self.store_path, "latest.pth"))

