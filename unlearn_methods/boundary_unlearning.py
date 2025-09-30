import copy
import utils
# from trainer import eval, loss_picker, optimizer_picker
import numpy as np
import torch
from torch import nn
from .adv_generator import LinfPGD, inf_generator, FGSM
import tqdm
import time
# from models import init_params as w_init

from models.ResNet import ResNet
from models.ViT import ViT  

def boundary_shrink(ori_model, train_forget_loader, forget_class, device, bound=0.1, step=8 / 255, iter=5, poison_epoch=10, dataset='mnist', unlearn_lr=None):
    start = time.time()
    norm = (dataset == 'cifar10') or (dataset == 'cifar100')
    # norm = True  # None#True if data_name != "mnist" else False
    random_start = True  # False if attack != "pgd" else True

    test_model = copy.deepcopy(ori_model).to(device)
    unlearn_model = copy.deepcopy(ori_model).to(device)
    start_time = time.time()
    adv = LinfPGD(test_model, bound, step, iter, norm, random_start, device)
    # adv = FGSM(test_model, bound, norm, random_start, device)
    forget_data_gen = inf_generator(train_forget_loader)
    batches_per_epoch = len(train_forget_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=unlearn_lr, momentum=0.9)
    
    # Set poison_epoch based on dataset and model
    if 'cifar' in dataset:
        if isinstance(unlearn_model, ResNet) and dataset == 'cifar100':
            poison_epoch=20
        elif isinstance(unlearn_model, ViT) and dataset in ['cifar10', 'cifar100']:
            poison_epoch=1
        else:
            poison_epoch=10
    elif dataset in ['mnist', 'mnistKuzushiji', 'mnistFashion']:
        poison_epoch=8
    
    # optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=0.000001, momentum=0.9)

    num_hits = 0
    num_sum = 0
    nearest_label = []

    for itr in tqdm.tqdm(range(poison_epoch * batches_per_epoch)):

        x, y = forget_data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        test_model.eval()
        x_adv = adv.perturb(x, y, target_y=None, model=test_model, device=device)
        adv_logits = test_model(x_adv)
        if isinstance(adv_logits, tuple):
            adv_logits = adv_logits[0]
        pred_label = torch.argmax(adv_logits, dim=1)
        if itr >= (poison_epoch - 1) * batches_per_epoch:
            nearest_label.append(pred_label.tolist())
        num_hits += (y != pred_label).float().sum()
        num_sum += y.shape[0]

        # adv_train
        unlearn_model.train()
        
        # for m in unlearn_model.modules():
        #         if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        #             m.eval()

        unlearn_model.zero_grad()
        optimizer.zero_grad()

        ori_logits = unlearn_model(x)
        if isinstance(ori_logits, tuple):
            ori_logits = ori_logits[0]
        ori_loss = criterion(ori_logits, pred_label)

        loss = ori_loss  # - KL_div
        loss.backward()
        optimizer.step()

    print('attack success ratio:', (num_hits / num_sum).float())
    # print(nearest_label)
    print('boundary shrink time:', (time.time() - start_time))
    # np.save('nearest_label', nearest_label)
    # torch.save(unlearn_model, '{}boundary_shrink_unlearn_model.pth'.format(path))

    return unlearn_model

# Parameter Initialization
def init_params(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.zeros_(m.bias)

def boundary_expanding(ori_model, train_forget_loader, forget_class, device, dataset='mnist', unlearn_lr=None):
    start = time.time()

    if dataset == 'agnews':
        n_filter2 = 64
        # n_filter2 = int(192 * 0.5)
        num_classes = 4
    elif dataset == 'cifar100':
        if ori_model.__class__.__name__ == 'AllCNN':
            n_filter2 = int(192 * 1)
        if ori_model.__class__.__name__ == 'ResNet':
            n_filter2 = 512
        if ori_model.__class__.__name__ == 'ViT':
            n_filter2 = 768
        num_classes = 100
    else:
        if ori_model.__class__.__name__ == 'AllCNN':
            n_filter2 = int(192 * 1)
        if ori_model.__class__.__name__ == 'ResNet':
            n_filter2 = 512
        if ori_model.__class__.__name__ == 'ViT':
            n_filter2 = 768
        # n_filter2 = int(192 * 0.5)
        num_classes = 10

    narrow_model = copy.deepcopy(ori_model).to(device)
    # feature_extrator = narrow_model.features
    feature_extrator = nn.Sequential()
    for l in list(narrow_model.children())[:-1]:
        feature_extrator.add_module(str(len(feature_extrator)), l)
    classifier = narrow_model.classifier

    widen_classifier = nn.Linear(n_filter2, num_classes + 1)
    init_params(widen_classifier)
    # if dataset == 'agnews':
    #     widen_model = copy.deepcopy(ori_model)
    #     widen_model.classifier = widen_classifier
    # else:
    #     widen_model = nn.Sequential(feature_extrator, widen_classifier)
    widen_model = copy.deepcopy(ori_model)
    widen_model.classifier = widen_classifier
    widen_model = widen_model.to(device)

    # dict = widen_classifier.state_dict()

    for name, params in classifier.named_parameters():
        # print(name, params.data.shape)
        if 'weight' in name:
            widen_classifier.state_dict()['weight'][0:num_classes, ] = classifier.state_dict()[name][:, ]
        elif 'bias' in name:
            widen_classifier.state_dict()['bias'][0:num_classes, ] = classifier.state_dict()[name][:, ]

    forget_data_gen = inf_generator(train_forget_loader)
    batches_per_epoch = len(train_forget_loader)
    

    criterion = nn.CrossEntropyLoss()
    if dataset == 'agnews':
        optimizer = torch.optim.SGD(widen_model.parameters(), lr=unlearn_lr)
        finetune_epochs = 10
    elif 'cifar' in dataset:
        optimizer = torch.optim.SGD(widen_model.parameters(), lr=unlearn_lr, momentum=0.9)
        if isinstance(ori_model, ResNet) and dataset == 'cifar100':
            finetune_epochs=20
        elif isinstance(ori_model, ViT) and dataset in ['cifar10', 'cifar100']:
            finetune_epochs=1
        else:
            finetune_epochs=10
    elif dataset in ['mnist', 'mnistKuzushiji', 'mnistFashion']:
        optimizer = torch.optim.SGD(widen_model.parameters(), lr=unlearn_lr, momentum=0.9)
        finetune_epochs=8
    # optimizer = optimizer_picker(optimization, widen_model.parameters(), lr=0.00001, momentum=0.9)
    # centr_optimizer = optimizer_picker(optimization, widen_model.parameters(), lr=0.00001, momentum=0.9)
    # adv_optimizer = optimizer_picker(optimization, adv_model.parameters(), lr=0.001, momentum=0.9)

    for itr in tqdm.tqdm(range(finetune_epochs * batches_per_epoch)):
        if dataset != 'agnews':
            x, y = forget_data_gen.__next__()
            x = x.to(device)
            y = y.to(device)

            widen_logits = widen_model(x)
        else:
            y, text, offsets = forget_data_gen.__next__()
            y, text, offsets = y.to(device), text.to(device), offsets.to(device)

            widen_logits = widen_model(text, offsets)

        # target label
        target_label = torch.ones_like(y, device=device)
        target_label *= num_classes

        # adv_train
        widen_model.train()
        # for m in widen_model.modules():
        #     if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        #         m.eval()

        widen_model.zero_grad()
        optimizer.zero_grad()

        widen_loss = criterion(widen_logits, target_label)

        widen_loss.backward()
        optimizer.step()


    pruned_classifier = nn.Linear(n_filter2, num_classes)
    for name, params in widen_model.classifier.named_parameters():
        # print(name)
        if 'weight' in name:
            pruned_classifier.state_dict()['weight'][:, ] = widen_model.classifier.state_dict()[name][0:num_classes, ]
        elif 'bias' in name:
            pruned_classifier.state_dict()['bias'][:, ] = widen_model.classifier.state_dict()[name][0:num_classes, ]

    # pruned_model = nn.Sequential(feature_extrator, pruned_classifier)
    pruned_model = copy.deepcopy(widen_model)
    pruned_model.classifier = pruned_classifier
    pruned_model = pruned_model.to(device)

    end = time.time()
    print('Time Consuming:', end - start, 'secs')

    unlearned_model = copy.deepcopy(ori_model)
    # copy the parameters from pruned model to unlearned model
    p_keys = list(pruned_model.state_dict().keys())
    u_keys = list(unlearned_model.state_dict().keys())
    new_dict = {}
    for idx, key in enumerate(pruned_model.state_dict()):
        new_dict[u_keys[idx]] = pruned_model.state_dict()[p_keys[idx]]

    unlearned_model.load_state_dict(new_dict)

    return unlearned_model
