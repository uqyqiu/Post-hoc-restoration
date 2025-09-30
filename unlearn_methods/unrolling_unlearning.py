import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

def unrolling(orig_model, fgt_loader, remain_loader, device, 
            fotgetting_only=False, unlearn_lr=None, sigma=None):
    incremental_model = copy.deepcopy(orig_model)
    incremental_model.train()

    optimizer = torch.optim.SGD(incremental_model.parameters(), lr=unlearn_lr)
    criterion = nn.CrossEntropyLoss()

    grad_list = []
    epochs = 1
    
    # for m in incremental_model.modules():
    #     if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
    #         m.eval()

    for e in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(fgt_loader):
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = incremental_model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()

            rec_grads = []
            for param in incremental_model.parameters():
                rec_grads.append(param.grad.cpu().detach())
            grad_list.append(rec_grads)

            optimizer.step()

        if not fotgetting_only:
            for batch_idx, (inputs, targets) in enumerate(remain_loader):
                optimizer.zero_grad()
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = incremental_model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    unlearn_model = copy.deepcopy(incremental_model)

    incrementaled_params = {}
    for i, (name, param) in enumerate(incremental_model.named_parameters()):
        incrementaled_params[name] = param.clone()
        for grads_in_step in grad_list:
            incrementaled_params[name] += sigma * grads_in_step[i].to(device)

    for name, params in unlearn_model.named_parameters():
        params.data.copy_(incrementaled_params[name])

    return unlearn_model
