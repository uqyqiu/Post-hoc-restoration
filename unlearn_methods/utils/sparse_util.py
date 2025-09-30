import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import time
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def prune_model_custom(model, mask_dict):
    """Pruning with custom mask (all conv layers)"""
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            mask_name = name + ".weight_mask"
            if mask_name in mask_dict.keys():
                prune.CustomFromMask.apply(
                    m, "weight", mask=mask_dict[name + ".weight_mask"]
                )

def extract_mask(model_dict):
    new_dict = {}
    for key in model_dict.keys():
        if "mask" in key:
            new_dict[key] = model_dict[key]
    return new_dict

def check_sparsity(model):
    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list + float(m.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(m.weight == 0))

    if zero_sum:
        remain_weight_ratio = 100 * (1 - zero_sum / sum_list)
    else:
        remain_weight_ratio = None

    return remain_weight_ratio

def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)

def prune_l1(model, prune_rate=0.95):
    parameters_to_prune = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m, "weight"))
    
    if not parameters_to_prune:
        return

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_rate,
    )

def iterative_unlearn(unlearn_iter_func):
    def _wrapped(data_loaders, model, criterion, unlearn_epochs=10, unlearn_lr=0.01, momentum=0.9, weight_decay=5e-4, **kwargs):
        optimizer = torch.optim.SGD(
            model.parameters(),
            unlearn_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=unlearn_epochs)

        for epoch in range(unlearn_epochs):
            unlearn_iter_func(data_loaders, model, criterion, optimizer, epoch, **kwargs)
            scheduler.step()

    return _wrapped

def finetune_epoch(data_loaders, model, criterion, optimizer, epoch, with_l1=False, alpha=0.2, print_freq=50):
    train_loader = data_loaders["retain"]
    model.train()

    for i, (image, target) in enumerate(train_loader):
        image, target = image.cuda(), target.cuda()

        output = model(image)
        loss = criterion(output, target)

        if with_l1:
            loss += alpha * l1_regularization(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
