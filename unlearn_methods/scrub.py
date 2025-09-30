

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from argparse import Namespace
from .utils.scrub_utils import AverageMeter, accuracy

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = nn.functional.log_softmax(y_s / self.T, dim=1)
        p_t = nn.functional.softmax(y_t / self.T, dim=1)
        loss = nn.functional.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt, split, quiet=False):
    """One epoch distillation"""
    for module in module_list:
        module.train()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)

        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        loss_kd = 0

        if split == "minimize":
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        elif split == "maximize":
            loss = -loss_div
        else:
            raise NotImplementedError(split)

        if split == "minimize" and not quiet:
            acc1, _ = accuracy(logit_s, target, topk=(1,1))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
        elif split == "maximize" and not quiet:
            kd_losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if not quiet and split == "minimize":
            if idx % opt.print_freq == 0:
                print(f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})')
                sys.stdout.flush()

    if split == "minimize":
        if not quiet:
            print(f' * Acc@1 {top1.avg:.3f}')
        return top1.avg, losses.avg
    else:
        return kd_losses.avg

def scrub_unlearning(
    original_model, 
    unlearn_train_loader, 
    retain_train_loader, 
    device,
    unlearn_epochs: int = 10,
    msteps: int = 3,
    kd_T: int = 2,
    optim_name: str = 'adam',
    unlearn_lr: float = 0.0005,
    momentum: float = 0.9,
    weight_decay: float = 0.1,
    gamma: float = 1.0,
    alpha: float = 0.5,
    beta: float = 0.0,
    distill: str = 'kd',
    print_freq: int = 10,
    quiet: bool = True
):
    student_model = original_model
    teacher_model = original_model

    module_list = nn.ModuleList([student_model, teacher_model])
    trainable_list = nn.ModuleList([student_model])

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(kd_T)
    criterion_kd = DistillKL(kd_T) # Placeholder for other KD losses, not used in SCRUB's default setup

    criterion_list = nn.ModuleList([criterion_cls, criterion_div, criterion_kd])

    if optim_name == "sgd":
        optimizer = optim.SGD(trainable_list.parameters(), lr=unlearn_lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_name == "adam": 
        optimizer = optim.Adam(trainable_list.parameters(), lr=unlearn_lr, weight_decay=weight_decay)
    elif optim_name == "rmsprop":
        optimizer = optim.RMSprop(trainable_list.parameters(), lr=unlearn_lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError(optim_name)

    if device == 'cuda':
        module_list.cuda()
        criterion_list.cuda()

    # Create a namespace object to hold parameters for train_distill
    args = Namespace(
        distill=distill,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        print_freq=print_freq
    )

    for epoch in range(1, unlearn_epochs + 1):
        if not quiet:
            print(f"==> Epoch: {epoch} <==")
            print("==> Scrub unlearning ...")

        maximize_loss = 0
        if epoch <= msteps:
            maximize_loss = train_distill(epoch, unlearn_train_loader, module_list, criterion_list, optimizer, args, "maximize", quiet=quiet)
        
        train_acc, train_loss = train_distill(epoch, retain_train_loader, module_list, criterion_list, optimizer, args, "minimize", quiet=quiet)

        if not quiet:
            print(f"  Maximize loss: {maximize_loss:.4f}\t Minimize loss: {train_loss:.4f}\t Train Acc: {train_acc:.2f}%")

    return student_model
