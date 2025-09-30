
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from .utils.helpers import save_gradient_ratio

def unlearn_salun(original_model, unlearn_train_loader, unlearn_class, device,
                  unlearn_lr=0.01, unlearn_epochs=8, threshold=0.5):
    """
    Saliency Unlearning (SalUn) method.

    Args:
        original_model: The original model to be unlearned.
        unlearn_train_loader: DataLoader for the data to be unlearned.
        retain_train_loader: DataLoader for the data to be retained (not used).
        unlearn_class (int or list): The class or classes to be unlearned.
        device: The device to run the training on (e.g., "cuda" or "cpu").
        unlearn_lr (float): Learning rate for the unlearning process.
        unlearn_epochs (int): Number of epochs for unlearning.
        threshold (float): The percentile of gradients to mask.

    Returns:
        torch.nn.Module: The unlearned model.
    """
    unlearn_model = deepcopy(original_model)
    unlearn_model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Get saliency map (gradients)
    gradients = save_gradient_ratio(unlearn_train_loader, unlearn_model, criterion, device, unlearn_lr=unlearn_lr)

    # Create mask from gradients
    hard_dict = {}
    all_elements = -torch.cat([tensor.flatten() for tensor in gradients.values()])
    threshold_index = int(len(all_elements) * threshold)
    positions = torch.argsort(all_elements)
    ranks = torch.argsort(positions)
    start_index = 0
    for key, tensor in gradients.items():
        num_elements = tensor.numel()
        tensor_ranks = ranks[start_index : start_index + num_elements]
        
        threshold_tensor = torch.zeros_like(tensor_ranks)
        threshold_tensor[tensor_ranks < threshold_index] = 1
        threshold_tensor = threshold_tensor.reshape(tensor.shape)
        hard_dict[key] = threshold_tensor
        start_index += num_elements

    # Unlearn using the mask
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=unlearn_lr)
    unlearn_model.train()
    for m in unlearn_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

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
    if isinstance(unlearn_class, int):
        unlearn_class = [unlearn_class]
    remain_class = list(set(range(num_classes)) - set(unlearn_class))

    for ep in range(unlearn_epochs):
        for batch, (x, y) in enumerate(unlearn_train_loader):
            x = x.to(device)
            y = torch.from_numpy(np.random.choice(remain_class, size=x.shape[0])).to(device)
            pred_y = unlearn_model(x)
            loss = criterion(pred_y, y)
            optimizer.zero_grad()
            loss.backward()
            
            # Apply mask
            for name, param in unlearn_model.named_parameters():
                if param.grad is not None:
                    param.grad *= hard_dict[name].to(device)

            optimizer.step()

    return unlearn_model
