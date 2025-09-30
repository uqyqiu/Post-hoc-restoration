import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.train_tools import textnet_train_one_epoch
from models.AllCNN import AllCNN
from models.TextNet import TextNet
from models.ResNet import ResNet
from models.ViT import ViT

def relearn(unlearn_model, loader, relearn_epochs, relearn_lr, device):        
    relearn_model = copy.deepcopy(unlearn_model).to(device)

    for name, param in relearn_model.named_parameters():
        param.requires_grad = True
        
    relearn_model.train()

    criterion = nn.CrossEntropyLoss()

    if isinstance(unlearn_model, ViT):
        optimizer = torch.optim.SGD(relearn_model.parameters(), lr=relearn_lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(relearn_model.parameters(), lr=relearn_lr)
    # optimizer = torch.optim.SGD(relearn_model.parameters(), lr=0.00001, momentum=0.9)

    for epoch in range(relearn_epochs):
        in_epoch_loss = []

        if isinstance(unlearn_model, AllCNN) or isinstance(unlearn_model, ResNet) or isinstance(unlearn_model, ViT):
            for step, (batch_x, batch_y) in enumerate(loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                output = relearn_model(batch_x)

                loss = criterion(output, batch_y)
                loss.backward()
                in_epoch_loss.append(loss.item())
                optimizer.step() 

            # print("epoch: {}, loss: {}".format(epoch, np.mean(in_epoch_loss)))
        else:
            optimizer = torch.optim.SGD(relearn_model.parameters(), lr=relearn_lr)
            textnet_train_one_epoch(relearn_model, loader, optimizer, criterion, epoch, device)
            print(f"epoch: {epoch}")

    return relearn_model