import copy
from utils.train_tools import fit_one_cycle


def retrain(
        epochs, max_lr, model, train_loader, val_loader, weight_decay, grad_clip, opt_func, device, output_activation=True
):
    retrained_model = copy.deepcopy(model).to(device)
    # Note: This function call needs to be updated to match the correct fit_one_cycle signature
    # The current call is incorrect and needs proper optimizer_type, optimizer_params, scheduler_type, scheduler_params
    raise NotImplementedError("This function call needs to be updated to match the fit_one_cycle signature")

    return retrained_model
