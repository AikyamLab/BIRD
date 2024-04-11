import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.utils.data as data
import csv
from PIL import Image
import random
from torch.utils.data import Sampler
import numpy as np
import random
import clip
from transformers import FlavaFeatureExtractor
import pandas as pd
import pickle

# from pytorch_metric_learning import samplers
device = "cuda" if torch.cuda.is_available() else "cpu" 
        
class CelebA(torch.utils.data.Dataset):
    def __init__(self, config, mode="train"):
        self.mode = mode
        self.config = config
        self.trans = get_transforms(config)
                    
        if mode == "train" and not config.meta2:
            self.dataset = pd.read_csv(os.path.join(config.dataset_root, f"combined.csv"))
        else:
            self.dataset = pd.read_csv(os.path.join(config.dataset_root, f"{mode}.csv"))
            
        if config.vlm == "clip":
            _, self.processor = clip.load("ViT-B/32", device)
        elif config.vlm == "clip50":
            _, self.processor = clip.load("RN50", device)
        
        elif config.vlm == "flava":
            self.processor = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")
                
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        example = os.path.join(self.config.dataset_root, "img_align_celeba", self.dataset.iloc[idx]["image_id"])
        image = Image.open(example).convert("RGB")
        
        task = int(self.dataset.iloc[idx]["Attractive"])
        sens = int(self.dataset.iloc[idx]["Male"])
        
        if task == -1:
            task = 0
        if sens == -1:
            sens = 0
        
        if self.config.vlm.startswith("clip"):
            return self.processor(image), task, sens, 0
        elif self.config.vlm == "flava":
            return self.processor(image, return_tensors="pt").pixel_values.squeeze(0), task, sens, 0
        else:
            return self.trans(image), task, sens, 0
        
def get_transforms(config):
    if config.dataset == "utk":
        size = (200, 200)
    elif config.dataset == "fairface":
        size = (224, 224)
    elif config.dataset == "celeba":
        size = (200, 200) # original = 178, 218
    elif config.dataset == "cifar10s":
        size = (32, 32)

    trans = transforms.Compose(
        [
            transforms.Resize(size),  # low res might cause drop in acc --> check this
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return trans


def get_dataloader(config, mode="train"):
    if config.dataset == "celeba":
        print("Getting Celeb A")
        dataset = CelebA(config, mode)
    else:
        raise(NotImplementedError) ## To be released later

    dataloader = data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True if mode in ["train", "quiz"] else False,
        pin_memory=True,
        prefetch_factor=16
    )

    return dataloader
