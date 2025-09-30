import os
import copy
import argparse
import numpy as np

from data.dataset import KuzushijiMNIST, get_subset, get_dataset, statstic_info
from utils.seed import set_seed

def prepare_data(args):
    seed = args.seed
    trials = args.trials
    request_rate = args.request_rate
    val_rate = args.val_rate
    dataset_name = args.dataset
    data_root = args.data_root
    
    prepared_data_save_path_template = os.path.join(args.save_path, 'prepared_data', '%s', 'trial_%s')
    
    # Get dataset
    raw_train_set, raw_test_set = get_dataset(dataset_name, data_root=data_root)
    num_classes = len(raw_train_set.classes)
    raw_train_set_stat = statstic_info(raw_train_set)
    print(f'Dataset: {dataset_name}')
    print(f'Train sample num: {raw_train_set_stat[0]}, num_classes: {raw_train_set_stat[1]}, class_sample_num: {raw_train_set_stat[2]}')
    
    for trial in range(trials):
        print(f'\n{"-"*10} Trial {trial}, set seed as {seed+trial} {"-"*10}')
        set_seed(seed + trial)

        train_set = copy.deepcopy(raw_train_set)
        test_set = copy.deepcopy(raw_test_set)

        shuffled_idx = np.arange(len(train_set))
        np.random.shuffle(shuffled_idx)
        
        if request_rate == 0:
            all_train_idx = shuffled_idx
            request_idx = np.array([])  # Empty request set
            
            val_at = int(len(all_train_idx) * (1 - val_rate))
            val_idx = all_train_idx[val_at:]
            train_idx = all_train_idx[:val_at]
            
            print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Request: {len(request_idx)} | Total: {len(train_idx) + len(val_idx) + len(request_idx)}')
            print(f'Train idx sample: {train_idx[:10]}')
            
            # Save indexes
            save_path = prepared_data_save_path_template % (dataset_name, trial)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(os.path.join(save_path, 'train_idx.npy'), train_idx)
            np.save(os.path.join(save_path, 'val_idx.npy'), val_idx)
            np.save(os.path.join(save_path, 'request_idx.npy'), request_idx)
            
            print('No request data (request_rate = 0)')
            
        else:
            split_at = int(len(train_set) * (1 - request_rate))
            train_idx, request_idx = shuffled_idx[:split_at], shuffled_idx[split_at:]

            val_at = int(len(train_idx) * (1 - val_rate))
            val_idx = train_idx[val_at:]
            train_idx = train_idx[:val_at]
            print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Request: {len(request_idx)} | Total: {len(train_idx) + len(val_idx) + len(request_idx)}')
            print(f'Train idx sample: {train_idx[:10]}')

            save_path = prepared_data_save_path_template % (dataset_name, trial)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(os.path.join(save_path, 'train_idx.npy'), train_idx)
            np.save(os.path.join(save_path, 'val_idx.npy'), val_idx)
            np.save(os.path.join(save_path, 'request_idx.npy'), request_idx)

            request_set = get_subset(train_set, request_idx)
            print(f'Statistic info of request set: {statstic_info(request_set)}')
            
        print(f'Data prepared and saved to: {save_path}')

def main():
    parser = argparse.ArgumentParser(description='Prepare data splits for training')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['mnist', 'mnistFashion', 'mnistKuzushiji', 'cifar10', 'cifar100', 'tinyimagenet', 'svhn'],
                        help='Dataset name')
    parser.add_argument('--seed', type=int, default=7, help='Random seed')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials')
    parser.add_argument('--request_rate', type=float, default=0.0, help='Request data rate (0 means no request data)')
    parser.add_argument('--val_rate', type=float, default=0.2, help='Validation data rate')
    parser.add_argument('--data_root', type=str, default='../datasets/', help='Root directory for datasets')
    parser.add_argument('--save_path', type=str, default='../runs/', help='Path to save prepared data')
    
    args = parser.parse_args()
    print(f'Arguments: {args}')
    
    prepare_data(args)

if __name__ == '__main__':
    main()