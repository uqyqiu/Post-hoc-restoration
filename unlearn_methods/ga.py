
import torch
import torch.nn as nn
from copy import deepcopy
import math

def unlearn_ga(original_model, unlearn_train_loader, device,
               unlearn_lr=0.00001, unlearn_epochs=30):
    """
    Gradient Ascent (GA).
    """
    unlearn_model = deepcopy(original_model)
    unlearn_model.to(device)
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=unlearn_lr)
    criterion = nn.CrossEntropyLoss()

    unlearn_model.train()
    # for m in unlearn_model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.eval()

    for ep in range(unlearn_epochs):
        for batch, (x, y) in enumerate(unlearn_train_loader):
            x, y = x.to(device), y.to(device)
            pred_y = unlearn_model(x)
            loss = -criterion(pred_y, y)  # Negative loss for ascent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if math.isnan(loss.item()):
                break
        if math.isnan(loss.item()):
            break
            
    return unlearn_model
