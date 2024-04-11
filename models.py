import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet, ShuffleNetV2
from torchvision.models.resnet import (
    BasicBlock,
    Bottleneck,
)
from models_cifar10s import SimpleCNN, plane_cifar10_book

# CLIP
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_models(config, student=True):
    adv = []
    if student and config.adv and config.dataset == "utk":
        adv = [2, 4] # gender, race
    elif student and config.adv and config.dataset == "fairface":
        adv = [2, 7] # gender, race
    elif student and config.adv and config.dataset == "celeba":
        adv = [2, 2] # gender, young
    elif student and config.adv and config.dataset == "cifar10s":
        adv = [2, 2] # color, color
    
    check = config.student if student else config.teacher
    
    # deep models
    if check == "shufflenetv2":
        model_task = shufflenet_v2_x0_5_feat(
            adv=adv, pretrained=config.use_pretrained
        )
        model_task.fc = nn.Linear(
            model_task.fc.in_features, config.num_task_classes
        )
    elif check == "resnet18":
        model_task = resnet18_feat(adv=adv, pretrained=config.use_pretrained)
        model_task.fc = nn.Linear(
            model_task.fc.in_features, config.num_task_classes
        )
    elif check == "resnet34":
        model_task = resnet34_feat(adv=adv, pretrained=config.use_pretrained)
        model_task.fc = nn.Linear(
            model_task.fc.in_features, config.num_task_classes
        )
    elif check == "cnn":
        model_task = SimpleCNN(layers=plane_cifar10_book['6'], adv=adv)
    
    # vlms
    elif check == "clip" and config.vlm == "clip":
        model_task = CLIP(in_channels=512, num_classes=config.num_task_classes, adv=adv)
        model_task.float()
        
    elif check == "clip50" and config.vlm == "clip50":
        model_task = CLIP(in_channels=1024, num_classes=config.num_task_classes, adv=adv)
        model_task.float()
    
    elif check == "flava" and config.vlm == "flava":
        model_task = Flava(num_classes=config.num_task_classes, adv=adv)
        model_task.float()
        
    else:
        raise (NotImplementedError)
    
    return model_task


def all_ones_init(m):
    if isinstance(m, nn.Linear):
        nn.init.eye_(m.weight)
        nn.init.zeros_(m.bias)
        

###########################################################################################################

class CLIP(nn.Module):
    def __init__(self, in_channels=512, num_classes=3, adv=[]):
        super().__init__()
        self.lin = nn.Linear(in_channels, 512)
        self.project = nn.Linear(512, num_classes)
        heads, self.adv_classifiers = [], []
        
        heads = [nn.Linear(512, cls) for cls in adv]

        if len(heads) > 0:
            self.adv_classifiers = nn.ModuleList(heads)
    
    def forward(self, x_embed):
        embed = self.lin(x_embed)
        
        # adversarial
        adv_outputs = [module(embed) for module in self.adv_classifiers]
        
        return self.project(embed), [0, 0, 0, embed.unsqueeze(-1).unsqueeze(-1)], embed, adv_outputs


class Flava(nn.Module):
    def __init__(self, num_classes=3, adv=[]):
        super().__init__()
        self.lin = nn.Linear(768, 512)
        self.project = nn.Linear(512, num_classes)
        heads, self.adv_classifiers = [], []
        
        heads = [nn.Linear(512, cls) for cls in adv]

        if len(heads) > 0:
            self.adv_classifiers = nn.ModuleList(heads)

    
    def forward(self, x_embed):
        x_embed = torch.mean(x_embed, 1)
        
        embed = self.lin(x_embed)
        
        # adversarial
        adv_outputs = [module(embed) for module in self.adv_classifiers]
        
        return self.project(embed), [0, 0, 0, embed.unsqueeze(-1).unsqueeze(-1)], embed, adv_outputs


class ResNetFeat(ResNet):
    def __init__(self, adv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adv_classifiers = nn.ModuleList([nn.Linear(self.fc.in_features, cls) for cls in adv])

    def _forward_impl(self, x, phi=None):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        feat = torch.flatten(x, 1)
        if phi is not None:
            feat = phi(feat)
            
        # z = self.proj(x)
        x = self.fc(feat)        

        # adversarial
        adv_outputs = [module(feat) for module in self.adv_classifiers]

        return x, [x1, x2, x3, x4], feat, adv_outputs

    def forward(self, x, phi=None):
        return self._forward_impl(x, phi=phi)


class ShuffleNetV2Feat(ShuffleNetV2):
    def __init__(self, adv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adv_classifiers = nn.ModuleList([nn.Linear(self.fc.in_features, cls) for cls in adv])

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x1 = self.stage2(x)
        x2 = self.stage3(x1)
        x3 = self.stage4(x2)
        x4 = self.conv5(x3)
        feat = x4.mean([2, 3])  # globalpool
        # z = self.proj(x)
        x = self.fc(feat)

        # adversarial
        adv_outputs = [module(feat) for module in self.adv_classifiers]
        
        return x, [x1, x2, x3, x4], feat, adv_outputs

    def forward(self, x):
        return self._forward_impl(x)


def _shufflenetv2(arch, pretrained, progress, *args, adv=[], **kwargs):
    model = ShuffleNetV2Feat(adv, *args, **kwargs)

    return model


def shufflenet_v2_x0_5_feat(adv=[], pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2(
        "shufflenetv2_x0.5",
        pretrained,
        progress,
        [4, 8, 4],
        [24, 48, 96, 192, 1024],
        adv=adv,
        **kwargs,
    )


def _resnet(arch, block, layers, pretrained, progress, adv, **kwargs):
    model = ResNetFeat(adv, block, layers, **kwargs)
    return model


def resnet18_feat(adv=[], pretrained=False, progress=True, **kwargs):
    return _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, adv=adv, **kwargs
    )

def resnet34_feat(adv=[], pretrained=False, progress=True, **kwargs):
    return _resnet(
        "resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, adv=adv, **kwargs
    )


def resnet50_feat(adv=[], pretrained=False, progress=True, **kwargs):
    return _resnet(
        "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, adv=adv, **kwargs
    )

