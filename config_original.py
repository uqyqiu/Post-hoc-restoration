from models.AllCNN import AllCNN
from models.ResNet import ResNet18, ResNet34, ResNet50

from utils.metric import test_all_in_one
from utils.train_tools import fit_one_cycle
from data.dataset import get_dataset

import torch
import itertools
import random

def get_unlearn_classes_set(num_classes, n_group=1, n_unlearn_classes=1, 
                            fixed_seed=None, dataset_name=None, 
                            specific_classes=None):
    if fixed_seed is not None:
            random.seed(fixed_seed)

    if n_unlearn_classes == 1:
        if dataset_name == 'cifar100' or dataset_name == 'tinyimagenet':
            if n_group == 1 and specific_classes is None:
                raise ValueError(f'n_group must be greater than 1 for cifar100')
            if specific_classes is not None:
                if isinstance(specific_classes[0], list):
                    return specific_classes
                else:
                    return [[i] for i in specific_classes]
            else:
                selected_classes = random.sample(range(num_classes), n_group)
                return [[i] for i in selected_classes]
        else:
            return [[i] for i in range(num_classes)]
    else:
        # Calculate total possible combinations to check if we need all or just a sample
        import math
        total_combinations = math.comb(num_classes, n_unlearn_classes)
        
        if n_group >= total_combinations:
            # If we need all combinations, generate them all
            all_combinations = list(itertools.combinations(range(num_classes), n_unlearn_classes))
            return [list(combo) for combo in all_combinations]
        else:
            # Randomly sample n_group unique combinations without generating all combinations
            selected_combinations = set()
            attempts = 0
            max_attempts = min(n_group * 100, total_combinations * 2)  # Prevent infinite loop
            
            while len(selected_combinations) < n_group and attempts < max_attempts:
                # Generate a random combination
                combo = tuple(sorted(random.sample(range(num_classes), n_unlearn_classes)))
                selected_combinations.add(combo)
                attempts += 1
            
            return [list(combo) for combo in selected_combinations]

def get_configs(dataset_name, model_name, n_group=10, n_unlearn_classes=1, seed=42):
    config_name = f'{dataset_name}-{model_name}'

    specific_classes = [45,18,13,33,12,59,58,79,41,5]

    CONFIGS = {
        'mnist-AllCNN': {
            'model': AllCNN(n_channels=1, num_classes=10),
            'batch_size': 128,
            'epochs': 30,
            'optimizer_type': torch.optim.SGD,
            'optimizer_params': {'lr': 0.01, 'weight_decay': 1e-4, 'momentum': 0.9},
            'scheduler_type': torch.optim.lr_scheduler.MultiStepLR,
            'scheduler_params': {'milestones': [30, 50], 'gamma': 0.1},
            'grad_clip': None,
            'training_func': fit_one_cycle,
            'test_func': test_all_in_one,
            'unlearn_classes_set': get_unlearn_classes_set(num_classes=10, 
                                                            n_group=n_group,
                                                            n_unlearn_classes=n_unlearn_classes,
                                                            dataset_name='mnist',
                                                            fixed_seed=seed),
        },
        'cifar10-AllCNN': {
            'model': AllCNN(n_channels=3, num_classes=10),
            'batch_size': 128,
            'epochs':180,
            'optimizer_type': torch.optim.SGD,
            'optimizer_params': {'lr': 0.1, 'weight_decay': 1e-4, 'momentum': 0.9},
            'scheduler_type': torch.optim.lr_scheduler.MultiStepLR,
            'scheduler_params': {'milestones': [90, 135], 'gamma': 0.1},
            'grad_clip': None,
            'training_func': fit_one_cycle,
            'test_func': test_all_in_one,
            'unlearn_classes_set': get_unlearn_classes_set(num_classes=10, 
                                                        n_group=n_group,
                                                        n_unlearn_classes=n_unlearn_classes,
                                                        dataset_name='cifar10',
                                                        fixed_seed=seed),
        },
        'cifar10-ResNet18': {
            'model': ResNet18(num_classes=10),
            'batch_size': 128,
            'epochs': 120,
            'optimizer_type': torch.optim.SGD,
            'optimizer_params': {'lr': 0.01, 'weight_decay': 5e-4, 'momentum': 0.9},
            'scheduler_type': torch.optim.lr_scheduler.MultiStepLR,
            'scheduler_params': {'milestones': [60, 90], 'gamma': 0.1},
            'grad_clip': None,
            'training_func': fit_one_cycle,
            'test_func': test_all_in_one,
            'unlearn_classes_set': get_unlearn_classes_set(num_classes=10, 
                                                            n_group=n_group,
                                                            n_unlearn_classes=n_unlearn_classes,
                                                            dataset_name='cifar10',
                                                            fixed_seed=seed),
        },
        'mnistFashion-AllCNN': {
            'model': AllCNN(n_channels=1, num_classes=10),
            'batch_size': 128,
            'epochs': 60,
            'optimizer_type': torch.optim.SGD,
            'optimizer_params': {'lr': 0.01, 'weight_decay': 1e-4, 'momentum': 0.9},
            'scheduler_type': torch.optim.lr_scheduler.MultiStepLR,
            'scheduler_params': {'milestones': [25, 45], 'gamma': 0.1},
            'grad_clip': None,
            'training_func': fit_one_cycle,
            'test_func': test_all_in_one,
            'unlearn_classes_set': get_unlearn_classes_set(num_classes=10, 
                                                            n_group=n_group,
                                                            n_unlearn_classes=n_unlearn_classes,
                                                            fixed_seed=seed,
                                                            dataset_name='mnistFashion',
                                                            specific_classes=None),
        },
        'cifar100-ResNet34': {
            'model': ResNet34(num_classes=100),
            'batch_size': 128,
            'epochs': 200,
            'optimizer_type': torch.optim.SGD,
            'optimizer_params': {'lr': 0.01, 'weight_decay': 1e-4, 'momentum': 0.9},
            'scheduler_type': torch.optim.lr_scheduler.MultiStepLR,
            'scheduler_params': {'milestones': [100, 150], 'gamma': 0.1},
            'grad_clip': None,
            'verbose': False,
            'training_func': fit_one_cycle,
            'test_func': test_all_in_one,
            'unlearn_classes_set': get_unlearn_classes_set(num_classes=100, 
                                                        n_group=n_group,
                                                        n_unlearn_classes=n_unlearn_classes,
                                                        fixed_seed=seed,
                                                        dataset_name='cifar100',
                                                        specific_classes=None),
        },
        'tinyimagenet-ResNet50': {
            'model': ResNet50(num_classes=200),
            'batch_size': 128,
            'epochs': 120,
            'optimizer_type': torch.optim.SGD,
            'optimizer_params': {'lr': 0.1, 'weight_decay': 1e-4, 'momentum': 0.9},
            'scheduler_type': torch.optim.lr_scheduler.MultiStepLR,
            'scheduler_params': {'milestones': [60, 90], 'gamma': 0.1},
            'grad_clip': None,
            'training_func': fit_one_cycle,
            'test_func': test_all_in_one,
            'unlearn_classes_set': get_unlearn_classes_set(num_classes=200, 
                                                        n_group=n_group,
                                                        n_unlearn_classes=n_unlearn_classes,
                                                        fixed_seed=seed,
                                                        dataset_name='tinyimagenet',
                                                        specific_classes=None),
        },

    }

    if config_name not in CONFIGS:
        raise ValueError(f'Config {config_name} not found! Please config it in config.py')
    return CONFIGS[config_name]

