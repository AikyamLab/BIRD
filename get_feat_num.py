import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet, AlexNet, ShuffleNetV2
from torchvision.models.resnet import BasicBlock, Bottleneck
# from mobilenet_feat import mobilenet_v2_feat
from main import config
from models import prepare_models
import json

os.environ["WANDB_MODE"] = "disabled"
feature_dict = {}

for model in ["resnet18", "resnet50", "shufflenetv2", "resnet34"]:
    
    print(model)

    config.AT = True
    config.student = model
    config.teacher = model
    config.use_pretrained = False

    random_input = torch.randn(2, 3, 200, 200)

    test_model = prepare_models(config)

    feat = test_model(random_input)[1]

    feature_dict[model] = [f.shape for f in feat]
    
    print(feature_dict)


with open("feat_shapes.json", "w") as f:
    json.dump(feature_dict, f, indent=6)
