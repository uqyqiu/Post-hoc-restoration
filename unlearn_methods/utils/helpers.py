
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy

# Note: Other utility functions from the original utils.py (like data loaders, MIA attacks, etc.) 
# have been excluded to keep this file focused on direct dependencies for the unlearning methods.

def get_representation_matrix(net, x, batch_list=[24, 100, 100, 125, 125, 250, 250, 256, 256]): 
    """
    GPU-accelerated version following the original algorithm more closely.
    Uses forward hooks to capture activations and implements proper conv processing.
    """
    net.eval()
    activations = {}
    hooks = []

    # Hook to capture the INPUT tensor of a layer
    def get_activation(name):
        def hook(model, input, output):
            # The input to a layer is a tuple; the tensor is the first element.
            activations[name] = input[0].detach()
        return hook

    # Define target layers to hook into. This logic can be customized.
    # Here, we default to a generic search for Conv2d and Linear layers.
    target_layers = []
    for name, module in net.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            target_layers.append((name, module))

    # Register the forward hooks
    for name, layer in target_layers:
        if layer is not None:
            hooks.append(layer.register_forward_hook(get_activation(name)))

    # Perform a forward pass to trigger the hooks
    with torch.no_grad():
        # Ensure data and model are on the same device
        device = next(net.parameters()).device
        x = x.to(device)
        _ = net(x)

    # Clean up by removing the hooks
    for hook in hooks:
        hook.remove()

    # Process the captured activations into representation matrices
    mat_list = []
    for i, (layer_name, layer) in enumerate(target_layers):
        if layer_name not in activations:
            continue

        act = activations[layer_name]

        # Use the batch_list to slice the activation tensor if provided
        if batch_list and i < len(batch_list):
            bsz = batch_list[i]
            act = act[:bsz]

        # For Conv2d layers, apply the efficient 'im2col' logic
        if isinstance(layer, nn.Conv2d):
            patches = torch.nn.functional.unfold(
                act,
                kernel_size=layer.kernel_size,
                dilation=layer.dilation,
                padding=layer.padding,
                stride=layer.stride
            )
            # Reshape to (C*kH*kW, B*L) to match the desired format
            mat = patches.transpose(1, 2).reshape(-1, patches.shape[1]).transpose(0, 1)
            mat_list.append(mat)

        # For Linear layers and others, use the standard transpose
        else:
            if act.ndim > 2:
                act = act.reshape(act.shape[0], -1)
            mat_list.append(act.transpose(0, 1))

    return mat_list

def update_GPM(mat_list, threshold, feature_list=[]):
    print ('Threshold: ', threshold)    
    if not feature_list:
        # After First Task - compute SVD on GPU
        for i in range(len(mat_list)):
            activation = mat_list[i]  # GPU tensor
            
            # Perform SVD on GPU using PyTorch
            try:
                U, S, Vh = torch.linalg.svd(activation, full_matrices=False)
                
                # Compute singular value ratios on GPU
                sval_total = (S**2).sum()
                sval_ratio = (S**2) / sval_total
                sval_cumsum = torch.cumsum(sval_ratio, dim=0)
                
                # Find the number of components to keep
                r = torch.sum(sval_cumsum < threshold).item()
                
                # Store the feature matrix (convert to numpy for compatibility)
                feature_list.append(U[:, 0:r].cpu().numpy())
                
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                # Fallback to CPU if GPU runs out of memory
                activation_cpu = activation.cpu().numpy()
                U, S, Vh = np.linalg.svd(activation_cpu, full_matrices=False)
                sval_total = (S**2).sum()
                sval_ratio = (S**2) / sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold)
                feature_list.append(U[:, 0:r])
    else:
        # Update GPM - perform matrix operations on GPU
        for i in range(len(mat_list)):
            activation = mat_list[i]  # GPU tensor
            feature_torch = torch.from_numpy(feature_list[i]).cuda().float()
            proj_matrix = torch.mm(feature_torch, feature_torch.t())
            act_hat_torch = activation - torch.mm(proj_matrix, activation)

            try:
                U, S, Vh = torch.linalg.svd(activation, full_matrices=False)
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                # Fallback to CPU if GPU runs out of memory or fails to converge
                activation_cpu = activation.cpu().numpy()
                U, S, Vh = np.linalg.svd(activation_cpu, full_matrices=False)
                U, S = torch.from_numpy(U).cuda(), torch.from_numpy(S).cuda()
            sval_total = (S**2).sum()
            
            U, S, Vh = torch.linalg.svd(act_hat_torch, full_matrices=False)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2) / sval_total
            accumulated_sval = (sval_total - sval_hat) / sval_total
            
            # Find r on GPU
            sval_cumsum = torch.cumsum(sval_ratio, dim=0)
            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            
            if r == 0:
                print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                continue
            U_new = U[:, 0:r].cpu().numpy()
            
            Ui = np.hstack((feature_list[i], U_new))
            if Ui.shape[1] > Ui.shape[0]:
                feature_list[i] = Ui[:, 0:Ui.shape[0]]
            else:
                feature_list[i] = Ui
    print('-'*40)
    print('Gradient Constraints Summary')
    print('-'*40)
    for i in range(len(feature_list)):
        print ('Layer {} : {}/{}'.format(i+1,feature_list[i].shape[1], feature_list[i].shape[0]))
    print('-'*40)                
    return feature_list

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def compute_maximum_length(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def hessian(model, train_loader, device):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    
    for p in model.parameters():
        p.grad2_acc = 0
    
    for data, orig_target in train_loader:
        data, orig_target = data.to(device), orig_target.to(device)
        output = model(data)
        prob = torch.nn.functional.softmax(output, dim=-1).data

        for y in range(output.shape[1]):
            target = torch.empty_like(orig_target).fill_(y)
            loss = loss_fn(output, target)
            model.zero_grad()
            loss.backward(retain_graph=True)
            for p in model.parameters():
                if p.requires_grad:
                    p.grad2_acc += torch.mean(prob[:, y]) * p.grad.data.pow(2)

    for p in model.parameters():
        p.grad2_acc /= len(train_loader)

def get_mean_var(p, unlearn_class, num_classes, alpha=1e-7):
    var = 1. / (p.grad2_acc + 1e-8)
    var = var.clamp(max=1e3)
    if p.size(0) == num_classes:
        var = var.clamp(max=1e2)
    var = alpha * var
    
    if p.ndim > 1:
        var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
    mu = p.data0.clone()

    if p.size(0) == num_classes:
        mu[unlearn_class] = 0
        var[unlearn_class] = 0.0001
        var *= 10
    elif p.ndim == 1:
        var *= 10
    return mu, var

def save_gradient_ratio(unlearn_train_loader, model, criterion, device, unlearn_lr=0.01, momentum=0.9):
    optimizer = torch.optim.SGD(
        model.parameters(),
        unlearn_lr,
        momentum=momentum
    )
    gradients = {}
    model.eval()
    for name, param in model.named_parameters():
        gradients[name] = 0

    for i, (image, target) in enumerate(unlearn_train_loader):
        image = image.to(device)
        target = target.to(device)
        output_clean = model(image)
        loss = -criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])
    
    return gradients