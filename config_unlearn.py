from models.AllCNN import AllCNN
from models.ResNet import ResNet18

from utils.metric import test_all_in_one
from utils.train_tools import fit_one_cycle
from data.dataset import get_dataset

import torch

def get_unlearn_configs(dataset_name, model_name, n_unlearn_classes=1, unlearn_method=None):
    config_name = f'{dataset_name}-{model_name}'
    if n_unlearn_classes == 1 and unlearn_method == 'boundary_shrink':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 5e-5,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 5e-4,
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 5e-4,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 3e-4,
            },
            'cifar100-AllCNN': {
                'unlearn_lr': 5e-4,
            },
            'cifar100-ResNet': {
                'unlearn_lr': 0.000173,
            },
        }
    elif n_unlearn_classes == 1 and unlearn_method == 'boundary_expanding':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 5e-4,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 1e-5,     
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 1e-3,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 1e-5,
            },
            'cifar100-AllCNN': {
                'unlearn_lr': 1e-3,
            },
            'cifar100-ResNet': {
                'unlearn_lr': 1e-5,
            },
        }
    elif n_unlearn_classes == 1 and unlearn_method == 'embedding_shift':
        CONFIGS = {
            'mnist-AllCNN': {
                'shift_epoch': 10,
                'shift_lr': 5 * 1e-5,
                'shift_lamb': 1,
                'refine_epoch': 1,
                'refine_lr': 2 * 1e-3,
            },
            'mnistFashion-AllCNN': {
                'shift_epoch': 10,
                'shift_lr': 1 * 1e-5,
                'shift_lamb': 1,
                'refine_epoch': 1,
                'refine_lr': 2 * 1e-3,
            },
            
            'cifar10-AllCNN': {
                'shift_epoch': 10,
                'shift_lr': 7 * 1e-4,
                'shift_lamb': 1,
                'refine_epoch': 1,
                'refine_lr': 1 * 1e-2,
            },
            
            'cifar10-ResNet18': {
                'shift_epoch': 10,
                'shift_lr': 1 * 1e-4,
                'shift_lamb': 1,
                'refine_epoch': 1,
                'refine_lr': 4 * 1e-3,
            },
            'cifar100-ResNet34': {
                'shift_epoch': 10,
                'shift_lr': 1 * 1e-5,
                'shift_lamb': 1,
                'refine_epoch': 1,
                'refine_lr': 3 * 1e-5,
            },
        }
    elif n_unlearn_classes == 1 and unlearn_method == 'unrolling':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 0.25,
                'sigma': 0.05,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 0.1,
                'sigma': 0.14,
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 0.2,
                'sigma': 0.16,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 0.02,
                'sigma': 0.01,
            },
            'cifar100-ResNet34': {
                'unlearn_lr': 0.01,
                'sigma': 0.02,
            },
        }
    elif n_unlearn_classes == 1 and unlearn_method == 'unrolling_f':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 0.02,
                'sigma': 0.02,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 0.005,
                'sigma': 0.01,
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 0.1,
                'sigma': 0.15,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 0.05,
                'sigma': 0.04,
            },
            'cifar100-ResNet34': {
                'unlearn_lr': 0.008,
                'sigma': 0.03,
            },
        }
    elif unlearn_method == 'unsc':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 0.01,
                'unlearn_epochs': 10,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 0.01,
                'unlearn_epochs': 15,
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 0.1,
                'unlearn_epochs': 25,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 0.05,
                'unlearn_epochs': 25,
            },
            'cifar100-ResNet34': {
                'unlearn_lr': 0.05,
                'unlearn_epochs': 25,
            },
        }
    elif n_unlearn_classes == 1 and unlearn_method == 'salun':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 1 * 1e-5,
                'unlearn_epochs': 8,
                'threshold': 0.3,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 4 * 1e-5,
                'unlearn_epochs': 8,
                'threshold': 0.3,
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 4 * 1e-4,
                'unlearn_epochs': 10,
                'threshold': 0.3,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 1 * 1e-4,
                'unlearn_epochs': 10,
                'threshold': 0.6,
            },
            'cifar100-ResNet34': {
                'unlearn_lr': 0.00102,
                'unlearn_epochs': 3,
                'threshold': 0.493,
            },
        }
    elif n_unlearn_classes == 1 and unlearn_method == 'ga':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 1 * 1e-5,
                'unlearn_epochs': 10,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 1 * 1e-4,
                'unlearn_epochs': 15,
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 1 * 1e-4,
                'unlearn_epochs': 30,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 5 * 1e-4,
                'unlearn_epochs': 10,
            },
            'cifar100-ResNet34': {
                'unlearn_lr': 5 * 1e-4,
                'unlearn_epochs': 10,
            },
        }
    elif n_unlearn_classes == 1 and unlearn_method == 'fisher':
        CONFIGS = {
            'mnist-AllCNN': {
                'alpha': 1e-7,
                'unlearn_epochs': 3,
            },
            'mnistFashion-AllCNN': {
                'alpha': 1 * 1e-7,
                'unlearn_epochs': 3,
            },
            'cifar10-AllCNN': {
                'alpha': 1e-7,
                'unlearn_epochs': 3,   
            },
            'cifar10-ResNet18': {
                'alpha': 1e-8,
                'unlearn_epochs': 3,
            },
            'cifar100-ResNet34': {
                'alpha': 1e-8,
                'unlearn_epochs': 3,
            },
        }
    elif n_unlearn_classes == 1 and unlearn_method == 'BadT':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 7 * 1e-3,
                'unlearn_epochs': 7,
                'KL_temperature': 1,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 5 * 1e-2,
                'unlearn_epochs': 5,
                'KL_temperature': 1,
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 0.07,
                'unlearn_epochs': 10,
                'KL_temperature': 1.0,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 0.00001,
                'unlearn_epochs': 3,
                'KL_temperature': 1,
            },
            'cifar100-ResNet34': {
                'unlearn_lr': 0.017,
                'unlearn_epochs': 10,
                'KL_temperature': 1,
            },
        }
    elif unlearn_method == 'sparse':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 0.01,
                'unlearn_epochs': 10,
                'prune_rate': 0.95,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 0.01,
                'unlearn_epochs': 10,
                'prune_rate': 0.95,
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 0.01,
                'unlearn_epochs': 10,
                'prune_rate': 0.95,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 0.01,
                'unlearn_epochs': 10,
                'prune_rate': 0.95,
            },
            'cifar100-ResNet34': {
                'unlearn_lr': 0.01,
                'unlearn_epochs': 10,
                'prune_rate': 0.95,
            },
        }
    elif n_unlearn_classes == 1 and unlearn_method == 'scrub':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 2 * 1e-5,
                'unlearn_epochs': 3,
                'msteps': 3,
                'alpha': 0.35,
                'gamma': 1.0,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 2 * 1e-5,
                'unlearn_epochs': 3,
                'msteps': 3,
                'alpha': 0.64,
                'gamma': 1.0,
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 5 * 1e-5,
                'unlearn_epochs': 3,
                'msteps': 3,
                'alpha': 0.5,
                'gamma': 1.0,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 1 * 1e-5,
                'unlearn_epochs': 3,
                'msteps': 3,
                'alpha': 0.57,
                'gamma': 1.0,
            },
            'cifar100-ResNet34': {
                'unlearn_lr': 2 * 1e-5,
                'unlearn_epochs': 3,
                'msteps': 3,
                'alpha': 0.7,
                'gamma': 1.0,
            },
        }
    elif n_unlearn_classes != 1 and unlearn_method == 'boundary_shrink':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 5e-5,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 5e-4,
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 5e-4,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 3e-4,
            },
            'cifar100-AllCNN': {
                'unlearn_lr': 5e-5,
            },
            'cifar100-ResNet34': {
                'unlearn_lr': 1 * 1e-4,
            },
        }
    elif n_unlearn_classes != 1 and unlearn_method == 'boundary_expanding':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 3e-5,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 1e-5,
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 1e-3,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 1e-5,
            },
            'cifar100-AllCNN': {
                'unlearn_lr': 5e-5,
            },
            'cifar100-ResNet34': {
                'unlearn_lr': 1 * 1e-5,
            },
        }
    elif n_unlearn_classes != 1 and unlearn_method == 'embedding_shift':
        CONFIGS = {
            'mnist-AllCNN': {
                'shift_epoch': 10,
                'shift_lr': 8 * 1e-5,
                'shift_lamb': 1,
                'refine_epoch': 1,
                'refine_lr': 3 * 1e-3,
            },
            'mnistFashion-AllCNN': {
                'shift_epoch': 10,
                'shift_lr': 2 * 1e-5,
                'shift_lamb': 1,
                'refine_epoch': 1,
                'refine_lr': 1 * 1e-3,
            },
            'cifar10-AllCNN': {
                'shift_epoch': 10,
                'shift_lr': 1 * 1e-3,
                'shift_lamb': 1,
                'refine_epoch': 1,
                'refine_lr': 1 * 1e-2,
            },
            'cifar10-ResNet18': {
                'shift_epoch': 10,
                'shift_lr': 5 * 1e-4,
                'shift_lamb': 1,
                'refine_epoch': 1,
                'refine_lr': 5 * 1e-3,
            },
            'cifar100-ResNet34': {
                'shift_epoch': 10,
                'shift_lr': 2 * 1e-4,
                'shift_lamb': 1,
                'refine_epoch': 1,
                'refine_lr': 2 * 1e-2,
            },
        }
    elif n_unlearn_classes != 1 and unlearn_method == 'salun':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 2 * 1e-5,
                'unlearn_epochs': 10,
                'threshold': 0.7,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 7 * 1e-5,
                'unlearn_epochs': 10,
                'threshold': 0.3,
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 4 * 1e-4,
                'unlearn_epochs': 8,
                'threshold': 0.44,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 1 * 1e-4,
                'unlearn_epochs': 10,
                'threshold': 0.5,
            },
            'cifar100-ResNet34': {
                'unlearn_lr': 8 * 1e-4,
                'unlearn_epochs': 5,
                'threshold': 0.3,
            },
        }
    elif n_unlearn_classes != 1 and unlearn_method == 'ga':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 2 * 1e-5,
                'unlearn_epochs': 10,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 2 * 1e-5,
                'unlearn_epochs': 40,
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 5 * 1e-4,
                'unlearn_epochs': 15,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 2 * 1e-4,
                'unlearn_epochs': 15,
            },
            'cifar100-ResNet34': {
                'unlearn_lr': 1 * 1e-6,
                'unlearn_epochs': 25,
            },
        }
    elif n_unlearn_classes != 1 and unlearn_method == 'fisher':
        CONFIGS = {
            'mnist-AllCNN': {
                'alpha': 1e-7,
                'unlearn_epochs': 3,
            },
            'mnistFashion-AllCNN': {
                'alpha': 1e-7,
                'unlearn_epochs': 3,
            },
            'cifar10-AllCNN': {
                'alpha': 1e-7,
                'unlearn_epochs': 3,
            },
            'cifar10-ResNet18': {
                'alpha': 1e-8,
                'unlearn_epochs': 3,
            },
            'cifar100-ResNet34': {
                'alpha': 1e-8,
                'unlearn_epochs': 3,
            },
        }
    elif n_unlearn_classes != 1 and unlearn_method == 'BadT':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 0.01,
                'unlearn_epochs': 3,
                'KL_temperature': 1,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 0.04,
                'unlearn_epochs': 5,
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 5 * 1e-2,
                'unlearn_epochs': 10,
                'KL_temperature': 1,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 0.06,
                'unlearn_epochs': 10,
                'KL_temperature': 1,
            },
            'cifar100-ResNet34': {
                'unlearn_lr': 0.02,
                'unlearn_epochs': 10,
                'KL_temperature': 1,
            },
        }

    elif n_unlearn_classes != 1 and unlearn_method == 'scrub':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 2 * 1e-5,
                'unlearn_epochs': 3,
                'msteps': 3,
                'alpha': 0.3,
                'gamma': 1.0,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 2 * 1e-5,
                'unlearn_epochs': 3,
                'msteps': 3,
                'alpha': 0.3,
                'gamma': 1.0,
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 5 * 1e-5,
                'unlearn_epochs': 3,
                'msteps': 3,
                'alpha': 0.3,
                'gamma': 1.0,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 1e-5,
                'unlearn_epochs': 3,
                'msteps': 3,
                'alpha': 0.6,
                'gamma': 1.0,
            },
            'cifar100-ResNet34': {
                'unlearn_lr': 1.7 * 1e-5,
                'unlearn_epochs': 3,
                'msteps': 3,
                'alpha': 0.6,
                'gamma': 1.0,
            },
        }
    elif n_unlearn_classes != 1 and unlearn_method == 'unrolling':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 0.004,
                'sigma': 0.01,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 0.24,
                'sigma': 0.1,
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 0.2,
                'sigma': 0.2,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 0.02,
                'sigma': 0.01,
            },
            'cifar100-ResNet34': {
                'unlearn_lr': 0.01,
                'sigma': 0.025,
            },
        }
    elif n_unlearn_classes != 1 and unlearn_method == 'unrolling_f':
        CONFIGS = {
            'mnist-AllCNN': {
                'unlearn_lr': 0.004,
                'sigma': 0.01,
            },
            'mnistFashion-AllCNN': {
                'unlearn_lr': 0.004,
                'sigma': 0.01,
            },
            'cifar10-AllCNN': {
                'unlearn_lr': 0.1,
                'sigma': 0.15,
            },
            'cifar10-ResNet18': {
                'unlearn_lr': 0.035,
                'sigma': 0.03,
            },
            'cifar100-ResNet34': {
                'unlearn_lr': 0.0024,
                'sigma': 0.022,
            },
        }
    if config_name not in CONFIGS:
        raise ValueError(f'Config {config_name} not found! Please config it in config.py')
    return CONFIGS[config_name]
