import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from .utils.helpers import get_representation_matrix, update_GPM

def unlearn_unsc(original_model, train_loader, unlearn_train_loader, unlearn_class, device, 
                 unlearn_lr=0.04, unlearn_epochs=25):
    # Ensure unlearn_class is a list for consistency
    if isinstance(unlearn_class, int):
        unlearn_class = [unlearn_class]
    
    unlearn_model = deepcopy(original_model)
    unlearn_model.to(device)
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=unlearn_lr)
    criterion = nn.CrossEntropyLoss()

    # Calculate the projection matrix from the retain data
    # This logic is adapted from the notebook cell that calculates `Proj_mat_lst`
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

    train_targets = np.array(train_loader.dataset.targets)
    
    # Calculate projection matrices for each iteration
    Proj_mat_lst = []
    rep_epochs = 1
    for i in range(rep_epochs):
        merged_feat_mat = []
        print(f'{"="*10} Representation Iteration {i} {"="*10}')
        for cls_id in range(num_classes):
            if cls_id in unlearn_class:
                continue
            cls_indices = np.where(np.isin(train_targets, cls_id))[0]
            cls_sampler = torch.utils.data.SubsetRandomSampler(cls_indices)
            cls_loader = torch.utils.data.DataLoader(train_loader.dataset, 
                batch_size=train_loader.batch_size, sampler=cls_sampler)

            for batch, (x, y) in enumerate(cls_loader):
                # Use smaller batch sizes to reduce memory consumption
                mat_list = get_representation_matrix(unlearn_model, x.to(device), batch_list=[24, 100, 100, 125, 125, 250, 250, 256, 256])
                break # Only use one batch per class to approximate

            threshold = 0.97 + 0.003 * cls_id
            merged_feat_mat = update_GPM(mat_list, threshold, merged_feat_mat)
            proj_mat = [torch.Tensor(np.dot(layer_basis, layer_basis.transpose())) for layer_basis in merged_feat_mat]
            Proj_mat_lst.append(proj_mat)

    # Unlearning process
    original_model.eval()
    for i in range(rep_epochs):
        unlearn_model.train()
        for m in unlearn_model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                
        for ep in range(unlearn_epochs):
            for batch, (x, y) in enumerate(unlearn_train_loader):
                x, y = x.to(device), y.to(device)

                with torch.no_grad():
                    masked_output = original_model(x)
                    masked_output[:, unlearn_class] = -np.inf
                    pseudo_y = torch.topk(masked_output, k=1, dim=1).indices.reshape(-1)

                pred_y = unlearn_model(x)
                loss = criterion(pred_y, pseudo_y)
                optimizer.zero_grad()
                loss.backward()

                kk = 0
                for k, (m,params) in enumerate(unlearn_model.named_parameters()):
                    # print(m, params.size())
                    if len(params.size())!=1:
                        sz =  params.grad.data.size(0)
                        params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                Proj_mat_lst[i][kk].cuda()).view(params.size())
                        kk +=1
                    elif len(params.size())==1:
                        params.grad.data.fill_(0)
                optimizer.step()
            print('[train] epoch {}, batch {}, loss {}'.format(ep, batch, loss.item()))
    return unlearn_model
