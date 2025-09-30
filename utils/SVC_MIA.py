import numpy as np
import torch
import torch.nn.functional as F
from cuml.svm import SVC
from sklearn.svm import SVC

def get_x_y_from_data_dict(data, device):
    x, y = data.values()
    if isinstance(x, list):
        x, y = x[0].to(device), y[0].to(device)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * torch.log(p), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

def _infer_num_classes(model: torch.nn.Module) -> int:
    if hasattr(model, 'num_classes'):
        return int(model.num_classes)
    if hasattr(model, 'classifier'):
        for m in reversed(list(model.classifier.modules())):
            if hasattr(m, 'out_features'):
                return int(m.out_features)
    return 10

def collect_prob(data_loader, model):
    if data_loader is None:
        num_classes = _infer_num_classes(model)
        return torch.zeros([0, num_classes]), torch.zeros([0], dtype=torch.long)

    prob_list, targets_list = [], []
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in data_loader:
            if device.type == 'cuda':
                torch.cuda.empty_cache()

                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    data, target = batch
                    if not isinstance(data, torch.Tensor) or not isinstance(target, torch.Tensor):
                        continue
                    if data.numel() == 0 or target.numel() == 0:
                        continue

                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                else:
                    try:
                        data, target = get_x_y_from_data_dict(batch, device)
                    except:
                        continue

                # Forward pass with error handling
                output = model(data)
                if output.numel() == 0:
                    continue

                prob_list.append(F.softmax(output, dim=-1).cpu())
                targets_list.append(target.cpu())


    if not prob_list:
        num_classes = _infer_num_classes(model)
        return torch.zeros([0, num_classes]), torch.zeros([0], dtype=torch.long)

    return torch.cat(prob_list), torch.cat(targets_list)

def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test, standardize=False):
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test  = shadow_test.shape[0]

    X_shadow_train = shadow_train.cpu().numpy().reshape(n_shadow_train, -1)
    X_shadow_test  = shadow_test.cpu().numpy().reshape(n_shadow_test, -1)

    min_size = min(n_shadow_train, n_shadow_test)
    if min_size > 0:
        if n_shadow_train > min_size:
            rng = np.random.default_rng()
            indices = rng.choice(n_shadow_train, size=min_size, replace=False)
            X_shadow_train = X_shadow_train[indices]
        elif n_shadow_test > min_size:
            rng = np.random.default_rng()
            indices = rng.choice(n_shadow_test, size=min_size, replace=False)
            X_shadow_test = X_shadow_test[indices]

    X_shadow = np.vstack([X_shadow_train, X_shadow_test]).astype(np.float32, copy=False)
    Y_shadow = np.concatenate(
        [np.ones(X_shadow_train.shape[0]), np.zeros(X_shadow_test.shape[0])]
    ).astype(np.int32, copy=False)

    X_to_fit = X_shadow
    mean, std = None, None
    if standardize and X_shadow.size > 0:
        mean = X_shadow.mean(axis=0, keepdims=True)
        std  = X_shadow.std(axis=0,  keepdims=True)
        std[std < 1e-6] = 1e-6
        X_to_fit = (X_shadow - mean) / std
        X_to_fit = X_to_fit.astype(np.float32, copy=False)

    clf = SVC(C=3.0, kernel="rbf", gamma="auto")
    accs = []

    if n_target_train > 0:
        X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
        if standardize and mean is not None:
            X_target_train = (X_target_train - mean) / std
        X_target_train = X_target_train.astype(np.float32, copy=False)
        acc_train = clf.predict(X_target_train).mean()
        accs.append(acc_train)

    if n_target_test > 0:
        X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
        if standardize and mean is not None:
            X_target_test = (X_target_test - mean) / std
        X_target_test = X_target_test.astype(np.float32, copy=False)
        acc_test = 1 - clf.predict(X_target_test).mean()
        accs.append(acc_test)

    return np.mean(accs)

def SVC_MIA(shadow_train, target_train, target_test, shadow_test, model):
    shadow_train_prob, shadow_train_labels = collect_prob(shadow_train, model)
    shadow_test_prob, shadow_test_labels = collect_prob(shadow_test, model)
    target_train_prob, target_train_labels = collect_prob(target_train, model)
    target_test_prob, target_test_labels = collect_prob(target_test, model)

    def compute_all_features(prob, labels):
        if prob.numel() == 0:
            empty_1d = torch.zeros((0, 1), device=prob.device, dtype=torch.float32)
            empty_fusion = torch.zeros((0, 5), device=prob.device, dtype=torch.float32)
            return {"correctness": empty_1d, "confidence": empty_1d, "entropy": empty_1d,
                    "loss": empty_1d, "margin": empty_1d, "fusion": empty_fusion}

        max_class_idx = prob.size(1) - 1

        valid_mask = (labels >= 0) & (labels <= max_class_idx)

        if not valid_mask.all():
            # Keep only valid samples   
            prob = prob[valid_mask]
            labels = labels[valid_mask]

            if prob.numel() == 0:
                # All samples were invalid, return empty tensors
                empty_1d = torch.zeros((0, 1), device=prob.device, dtype=torch.float32)
                empty_fusion = torch.zeros((0, 5), device=prob.device, dtype=torch.float32)
                return {"correctness": empty_1d, "confidence": empty_1d, "entropy": empty_1d,
                        "loss": empty_1d, "margin": empty_1d, "fusion": empty_fusion}

        corr = (torch.argmax(prob, dim=1) == labels).int().unsqueeze(1).float()
        conf = torch.gather(prob, 1, labels[:, None])
        entr = entropy(prob).unsqueeze(1)
        loss = -(conf.clamp_min(1e-12)).log()
        
        tmp = prob.clone()
        tmp[torch.arange(len(labels)), labels] = -1.0
        p_second = tmp.max(dim=1, keepdim=True).values
        margin = conf - p_second

        fusion = torch.cat([corr, conf, entr, loss, margin], dim=1)

        return {"correctness": corr, "confidence": conf, "entropy": entr,
                "loss": loss, "margin": margin, "fusion": fusion}

    st_feats = compute_all_features(shadow_train_prob, shadow_train_labels)
    sh_feats = compute_all_features(shadow_test_prob, shadow_test_labels)
    tt_feats = compute_all_features(target_train_prob, target_train_labels)
    th_feats = compute_all_features(target_test_prob, target_test_labels)
    
    results = {}
    feature_names = ["correctness", "confidence", "entropy", "loss", "margin", "fusion"]
    
    for feature in feature_names:
        standardize = feature in ["entropy", "loss", "margin", "fusion"]
        
        acc = SVC_fit_predict(
            st_feats[feature], sh_feats[feature],
            tt_feats[feature], th_feats[feature],
            standardize=standardize
        )
        results[feature] = acc
        
    return results