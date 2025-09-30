
import torch
import torch.nn as nn
from copy import deepcopy

def unlearn_rl(original_model, unlearn_train_loader, retain_train_loader, unlearn_class, device,
               lr=0.1, epochs=20, momentum=0.9, weight_decay=5e-4, milestones=[60, 120, 160], gamma=0.2):
    """
    Retraining from scratch on the retain dataset (RL baseline).

    Args:
        original_model: The original model architecture (will be re-initialized).
        unlearn_train_loader: DataLoader for the data to be unlearned (not used).
        retain_train_loader: DataLoader for the data to be retained.
        unlearn_class (int or list): The class or classes to be unlearned (not directly used, as data is pre-filtered).
        device: The device to run the training on (e.g., "cuda" or "cpu").
        lr (float): Learning rate for training.
        epochs (int): Number of epochs for training.
        momentum (float): Momentum for the SGD optimizer.
        weight_decay (float): Weight decay for the SGD optimizer.
        milestones (list): List of epoch indices for learning rate decay.
        gamma (float): Multiplicative factor for learning rate decay.

    Returns:
        torch.nn.Module: The retrained model.
    """
    # Re-initialize the model to train from scratch
    retrain_model = deepcopy(original_model)
    for layer in retrain_model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    retrain_model.to(device)

    optimizer = torch.optim.SGD(retrain_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    retrain_model.train()
    for ep in range(epochs):
        for batch, (x, y) in enumerate(retain_train_loader):
            x, y = x.to(device), y.to(device)
            pred_y = retrain_model(x)
            loss = criterion(pred_y, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return retrain_model
