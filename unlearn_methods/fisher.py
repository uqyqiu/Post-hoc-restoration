
import torch
import torch.nn as nn
from copy import deepcopy
from .utils.helpers import hessian, get_mean_var

def unlearn_fisher(original_model, retain_train_loader, unlearn_class, device,
                   alpha=1e-7, unlearn_epochs=3):
    """
    Fisher Unlearning method.

    Args:
        original_model: The original model to be unlearned.
        retain_train_loader: DataLoader for the data to be retained (used to calculate Hessian).
        unlearn_class (int or list): The class or classes to be unlearned.
        device: The device to run the training on (e.g., "cuda" or "cpu").
        alpha (float): A hyperparameter to scale the variance.

    Returns:
        torch.nn.Module: The unlearned model.
    """
    unlearn_model = deepcopy(original_model)
    unlearn_model.to(device)
    for ep in range(unlearn_epochs):
        for p in unlearn_model.parameters():
            p.data0 = p.data.clone()
            
        hessian(unlearn_model, retain_train_loader, device)
        
        num_classes = -1
        if hasattr(original_model, 'fc'):
            num_classes = original_model.fc.out_features
        elif hasattr(original_model, 'module') and hasattr(original_model.module, 'fc'):
            num_classes = original_model.module.fc.out_features
        else:
            for layer in reversed(list(original_model.modules())):
                if isinstance(layer, nn.Linear):
                    num_classes = layer.out_features
                    break

        for i, p in enumerate(unlearn_model.parameters()):
            mu, var = get_mean_var(p, unlearn_class, num_classes, alpha=alpha)
            p.data = mu + var.sqrt() * torch.empty_like(p.data).normal_()
        
    return unlearn_model
