import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import PowerTransformer
from skimage.filters import threshold_otsu

def get_logits_and_rep(net, loader, device):
    net.eval()
    pred = np.array([])
    label = np.array([])
    logits = []
    rep_all = []
    softmax_logits_all = []
    with torch.no_grad():
        if 'Text' not in net.__class__.__name__:
            for i, (x, y) in enumerate(loader):
                x = torch.tensor(x.clone().detach().numpy(), dtype=torch.float32).to(device)
                y = y.to(device)
                label = np.concatenate((label, y.cpu().numpy()))

                rep = net.get_representation(x)
                y_pred = net(x)
                
                rep_all.append(rep.cpu().detach())
                softmax_logits = F.softmax(y_pred, dim=1)
                logits.append(y_pred.cpu().detach().numpy())
                softmax_logits_all.append(softmax_logits.cpu().detach().numpy())
                _, predicted = torch.max(y_pred.data, 1)
                predicted = predicted.cpu().numpy()
                pred = np.concatenate((pred, predicted))
        else:
            for i, (y, x, offsets) in enumerate(loader):
                y, x, offsets = y.to(device), x.to(device), offsets.to(device)
                label = np.concatenate((label, y.cpu().numpy()))

                y_pred = net(x, offsets)
                
                softmax_logits = F.softmax(y_pred, dim=1)
                logits.append(y_pred.cpu().detach().numpy())
                softmax_logits_all.append(softmax_logits.cpu().detach().numpy())
                _, predicted = torch.max(y_pred.data, 1)
                predicted = predicted.cpu().numpy()
                pred = np.concatenate((pred, predicted))

    # record logits
    logits = np.concatenate(logits)
    softmax_logits_all = np.concatenate(softmax_logits_all)
    rep_all = torch.cat(rep_all)
    return logits, softmax_logits_all, rep_all, pred, label

def yeo_johnson(data):
    safe_data = np.copy(data).astype(np.float64)
    data_range = np.max(safe_data) / np.min(safe_data) if np.min(safe_data) > 0 else np.inf
    
    if data_range > 1e6:
        epsilon = 0.001
        log_data = np.log(safe_data + epsilon)
        
        data_mean = np.mean(log_data)
        data_std = np.std(log_data)
        if data_std > 0:
            normalized_data = (log_data - data_mean) / data_std
            max_abs = np.max(np.abs(normalized_data))
            if max_abs > 5:
                scale_factor = 5.0 / max_abs
                normalized_data = normalized_data * scale_factor
            else:
                scale_factor = 1.0
        else:
            normalized_data = log_data
            scale_factor = 1.0
            data_std = 1.0
        
        pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=True)
        transformed_data = pt.fit_transform(normalized_data)
    else:
        pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=True)
        transformed_data = pt.fit_transform(data)
    return pt, transformed_data

def find_otsu_threshold(data, verbose=False):
    if verbose:
        print("--- Running Otsu's method ---")
    threshold = threshold_otsu(data)
    if verbose:
        print(f"Otsu threshold: {threshold:.4f}")
    return threshold