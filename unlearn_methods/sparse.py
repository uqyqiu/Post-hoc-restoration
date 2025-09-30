import torch
import copy
from .utils import sparse_util as utils

@utils.iterative_unlearn
def finetune(data_loaders, model, criterion, optimizer, epoch, **kwargs):
    """Simple finetuning on the retain set."""
    return utils.finetune_epoch(data_loaders, model, criterion, optimizer, epoch, **kwargs)

# def sparse_unlearning(
#     original_model,
#     retain_train_loader,
#     device,
#     unlearn_epochs=10,
#     unlearn_lr=0.01,
#     momentum=0.9,
#     weight_decay=5e-4,
#     prune_rate=0.95,
#     **kwargs
# ):
#     """This function implements the 'prune first, then unlearn' paradigm."""
    
#     model = copy.deepcopy(original_model)
#     model.to(device)

#     criterion = torch.nn.CrossEntropyLoss()

#     utils.prune_l1(model, prune_rate)

#     data_loaders = {"retain": retain_train_loader}
#     finetune(
#         data_loaders, 
#         model, 
#         criterion, 
#         unlearn_epochs=unlearn_epochs, 
#         unlearn_lr=unlearn_lr, 
#         momentum=momentum, 
#         weight_decay=weight_decay, 
#         **kwargs
#     )
    
#     return model

def sparse_unlearning(original_model, retain_train_loader, device,
                    unlearn_epochs=10, unlearn_lr=0.01, momentum=0.9,
                    weight_decay=5e-4, prune_rate=0.95, **kwargs 
                    ):
    """This function implements the 'FT_prune_bi' algorithm."""
    
    model = copy.deepcopy(original_model)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    data_loaders = {"retain": retain_train_loader}
    finetune(
        data_loaders, 
        model, 
        criterion, 
        unlearn_epochs=unlearn_epochs, 
        unlearn_lr=unlearn_lr, 
        momentum=momentum, 
        weight_decay=weight_decay, 
        **kwargs
    )

    utils.prune_l1(model, prune_rate)

    initialization = original_model.state_dict()
    current_mask = utils.extract_mask(model.state_dict())
    utils.prune_model_custom(model, current_mask)
    model.load_state_dict(initialization, strict=False)

    finetune(
        data_loaders, 
        model, 
        criterion, 
        unlearn_epochs=unlearn_epochs, 
        unlearn_lr=unlearn_lr, 
        momentum=momentum, 
        weight_decay=weight_decay, 
        **kwargs
    )
    
    return model
