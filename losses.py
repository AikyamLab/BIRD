import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def softargmax(x, beta=1e10):
    x_range = torch.linspace(0, 1, x.shape[-1]).to(x.device)
    x = torch.softmax(x * beta, -1)
    out = torch.sum((x.shape[-1] - 1) * x * x_range, dim=-1)
    # print(out)
    return out


def equalized_odds_loss(pred, labels, sens, config):
    
    if "spd" in config.outer_loss:
        return statistical_parity_difference(pred, labels, sens, config)
    else:
        # pred: BS, Num_classes
        sens = sens.to(labels.device)
        num_labels = len(torch.unique(labels, sorted=False))
        num_sens = len(torch.unique(sens, sorted=False))
        
        # loss = torch.zeros(num_labels, num_sens)
        loss_max = torch.zeros(num_labels)
        
        for label_class in range(num_labels):
            loss_label = []
            idx_y = labels == label_class
            pred_y = pred[idx_y, ...]
            labels_y = labels[idx_y, ...]
            for sens_class in range(num_sens):
                idx_s = sens == sens_class
                idx_y_s = torch.bitwise_and(idx_y, idx_s)
                
                # L_y_a - L_ce
                # L_ce = F.cross_entropy(pred, labels)
                pred_y_a = pred[idx_y_s, ...]
                labels_y_a = labels[idx_y_s, ...]
                if pred_y_a.shape[0] > 0:
                    L_y_a = F.cross_entropy(pred_y_a, labels_y_a)
                    L_y = F.cross_entropy(pred_y, labels_y)
                    
                    # loss[label_class][sens_class] = torch.abs(L_y_a - L_y) #+ L_ce
                    if config.outer_loss == "mse":
                        loss_label.append(F.mse_loss(L_y_a, L_y))
                    else:
                        loss_label.append(torch.abs(L_y_a - L_y))
            
            if len(loss_label) > 0:
                if config.outer_loss_sum == "mean":
                    loss_max[label_class] = sum(loss_label)/len(loss_label)
                else:
                    loss_max[label_class] = max(loss_label)
        
            
        return torch.mean(loss_max)
        # return (torch.mean(loss_max) + torch.mean(loss))/2 ## v5, v6
    

def statistical_parity_difference(pred, labels, sens, config):
    """
        SPD: (num_positives (protected) / total_protected) - (num_positives (unprotected) / total_unprotected)
    """
    # pred: BS, Num_classes
    sens = sens.to(labels.device)
    num_labels = len(torch.unique(labels, sorted=False))
    num_sens = len(torch.unique(sens, sorted=False))
    
    # loss = torch.zeros(num_labels, num_sens)
    loss_max = torch.zeros(num_labels)
    
    for label_class in range(num_labels):
        loss_label = []

        for sens_class in range(num_sens):
            idx_s = sens == sens_class
            idx_not_s = ~idx_s

            pred_a, pred_not_a = pred[idx_s, ...], pred[idx_not_s, ...]
            labels_a, labels_not_a = labels[idx_s, ...], labels[idx_not_s, ...]
            if pred_a.shape[0] > 0:
                L_a = F.cross_entropy(pred_a, labels_a)
                L_not_a = F.cross_entropy(pred_not_a, labels_not_a)
                # L_not_a = F.cross_entropy(pred, labels)
                
                if config.outer_loss == "spd_mse":
                    loss_label.append(F.mse_loss(L_a, L_not_a))
                else:
                    loss_label.append(torch.abs(L_a - L_not_a))
        
        if len(loss_label) > 0:
            if config.outer_loss_sum == "spd_mean":
                loss_max[label_class] = sum(loss_label)/len(loss_label)
            else:
                loss_max[label_class] = max(loss_label)
    
           
    return torch.mean(loss_max)
    


def loss_fn_kd(outputs, labels, teacher_outputs, config):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    """
    alpha = config.alpha
    T = config.temperature
    loss_ce = nn.CrossEntropyLoss()(outputs, labels)

    KD_loss = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1)
    ) * (alpha * T * T) + loss_ce * (1.0 - alpha)

    return KD_loss, loss_ce


# FitNet
class Hint(nn.Module):
    """
    FitNets: Hints for Thin Deep Nets
    https://arxiv.org/pdf/1412.6550.pdf
    """

    def __init__(self):
        super(Hint, self).__init__()

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(fm_s, fm_t)

        return loss
    
# class FairWithoutDemo(nn.Module):
#     def __init__(self, config):
#         super(FairWithoutDemo, self).__init__()
#         self.config = config
    
#     # softmax label
#     def forward(self, outputs, teacher_outputs):
#         T = self.config.temperature
#         soft_teacher = F.softmax(teacher_outputs / T, dim=1)
        
#         return nn.CrossEntropyLoss()(outputs, soft_teacher)
    
class FairWithoutDemo(nn.Module):
    def __init__(self, config):
        super(FairWithoutDemo, self).__init__()
        self.T = config.temperature

    def forward(self, pred, teacher_outputs):
        pred = F.log_softmax(pred / self.T, dim=1)
        soft_teacher = F.softmax(teacher_outputs / self.T, dim=1)
        
        return torch.mean(torch.sum(-soft_teacher * pred, 1))
   
# V2 
# class FairWithoutDemo(nn.Module):
#     def __init__(self, config):
#         super(FairWithoutDemo, self).__init__()
#         self.T = config.temperature

#     def forward(self, pred, teacher_outputs):
#         import ipdb;ipdb.set_trace()
#         pred = F.log_softmax(pred / self.T, dim=1)
#         soft_teacher = F.softmax(teacher_outputs / self.T, dim=1)
        
#         return nn.KLDivLoss(reduction="batchmean")(pred, soft_teacher)


class NST(nn.Module):
    """
    Like What You Like: Knowledge Distill via Neuron Selectivity Transfer
    https://arxiv.org/pdf/1707.01219.pdf
    """

    def __init__(self):
        super(NST, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), fm_s.size(1), -1)
        fm_s = F.normalize(fm_s, dim=2)

        fm_t = fm_t.view(fm_t.size(0), fm_t.size(1), -1)
        fm_t = F.normalize(fm_t, dim=2)

        loss = (
            self.poly_kernel(fm_t, fm_t).mean()
            + self.poly_kernel(fm_s, fm_s).mean()
            - 2 * self.poly_kernel(fm_s, fm_t).mean()
        )

        return loss

    def poly_kernel(self, fm1, fm2):
        fm1 = fm1.unsqueeze(1)
        fm2 = fm2.unsqueeze(2)
        out = (fm1 * fm2).sum(-1).pow(2)

        return out


class AttentionTransfer(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""

    def __init__(self, p=2, mobilenet_student=False, depth=0):
        super(AttentionTransfer, self).__init__()
        self.p = p
        self.mobilenet_student = mobilenet_student
        self.depth = depth

    def forward(self, g_s, g_t):
        g_t = [F.adaptive_avg_pool2d(gt, gs.shape[-2:]) for gs, gt in zip(g_s, g_t)]
        # return mean
        return sum(
            [self.at_loss(f_s, f_t.detach()) for f_s, f_t in zip(g_s, g_t)]
        ) / len(g_s)

    def at_loss(self, x, y):
        return (self.at(x) - self.at(y)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


class Losses(nn.Module):
    """
    Wrapper function that implements baselines
    """

    def __init__(self, config, margin=1.0):
        super(Losses, self).__init__()
        self.config = config
        self.margin = margin

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        attributes: list,
    ):
        task, sens = attributes

        sens = sens.to(task.device)
        num_task_classes = len(torch.unique(task, sorted=False))
        num_sens_classes = len(torch.unique(sens, sorted=False))

        loss_sens = torch.zeros(num_task_classes, num_sens_classes)

        for task_class in range(num_task_classes):
            idx_y = task == task_class
            teacher_group_avg = []

            teacher_group_avg = []
            for sens_class in range(num_sens_classes):
                idx_s = sens == sens_class
                idx_y_s = torch.bitwise_and(idx_y, idx_s)

                teacher_group_avg.append(teacher_features[idx_y_s, ...])

            teacher_group_avg = self.average_max(teacher_group_avg)
            for sens_class in range(num_sens_classes):
                idx_s = sens == sens_class
                idx_y_s = torch.bitwise_and(idx_y, idx_s)

                student_group_cond = student_features[idx_y_s, ...]

                loss_sens[task_class][sens_class] = self.mmd_loss(
                    student_group_cond, teacher_group_avg
                )

        return torch.mean(loss_sens)
        
    # minority oversampling: teacher avg
    def average_max(self, teacher_feat):
        # import ipdb;ipdb.set_trace()
        max_bs = max(teacher_feat, key=lambda x: x.size(0)).size(0)
        tensors = []
        for tensor in teacher_feat:
            if tensor.size(0) < max_bs:
                # repeat
                if tensor.size(0) > 0:
                    # 15 10
                    if max_bs % tensor.size(0) != 0:
                        times = (max_bs // tensor.size(0)) + 1
                    else:
                        times = max_bs // tensor.size(0)
                    
                    tensors.append(tensor.repeat(times, 1, 1, 1)[:max_bs, ...])
                        

            else:
                tensors.append(tensor)

        # import pdb;pdb.set_trace()
        return sum(tensors) / len(tensors)

    def get_repeated(self, feats):
        max_bs = max(feats, key=lambda x: x.size(0)).size(0)
        tensors = []
        for tensor in feats:
            if tensor.size(0) < max_bs:
                # repeat
                if tensor.size(0) > 0:
                    # 15 10
                    if max_bs % tensor.size(0) != 0:
                        times = (max_bs // tensor.size(0)) + 1
                    else:
                        times = max_bs // tensor.size(0)

                    tensors.append(tensor.repeat(times, 1, 1, 1)[:max_bs, ...])

            else:
                tensors.append(tensor)

        return tensors

    def mmd_loss(self, feat1, feat2):
        if feat1.shape[0] > 0 and feat2.shape[0] > 0:
            feat1, feat2 = self.get_repeated([feat1, feat2])
            return NST()(feat1, feat2) * self.config.lambda_ / 2
        else:
            return 0.
