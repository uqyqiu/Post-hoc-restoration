import torch
import argparse
from config_original import get_configs
from utils.seed import set_seed
from utils.metric import test_all_in_one
from utils.unlearn_tools import get_idx_by_unlearn_class
from data.dataset import get_dataloader, get_dataset, get_subset
from utils.recover_tools import yeo_johnson, find_otsu_threshold, get_logits_and_rep
import os
import numpy as np
import copy
from utils.metric import accuracy
import pandas as pd
import torch.nn as nn
from models.AllCNN import AllCNN

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str, default='mnist', choices=['mnist', 'mnistFashion', \
                                                                    'mnistKuzushiji', 'cifar10', \
                                                                    'cifar100', 'tinyimagenet'])
parser.add_argument('--model', type=str, default='AllCNN', choices=['AllCNN', 'ResNet18', 'ResNet34', 'ResNet50', 'ViT'])
parser.add_argument('--unlearn-method', type=str, default='embedding_shift',
                    choices=['retrain', 'embedding_shift', 'boundary_shrink', \
                        'boundary_expanding', 'embedding_shift_CE', \
                        'unrolling', 'unrolling_f', 'unsc', 'salun', \
                        'ga', 'fisher', 'BadT', 'sparse', 'scrub'])
parser.add_argument('--trials', type=int, default=3)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--init_node', action='store_true', default=False, 
                      help='Whether to initlise the node for unlearned class')
parser.add_argument('--n_unlearn_classes', type=int, default=2)
parser.add_argument('--n_group', type=int, default=10)
parser.add_argument('--seed_unlearn_class', type=int, default=3407)

args = parser.parse_args()
# methods_list = [
#     'retrain',
#     'embedding_shift',
#     'boundary_shrink',
#     'boundary_expanding',
#     'unrolling',
#     'unrolling_f',
#     'unsc',
#     'salun',
#     'ga',
#     'fisher',
#     'BadT', 
#     'sparse', 
#     'scrub'
# ]

dataset_model_mapping = {
    'mnist': ['AllCNN'],
    'mnistFashion': ['AllCNN'],
    'cifar10': ['ResNet18'],
    'cifar100': ['ResNet34']
}

prepared_data_path_template = '../runs/prepared_data/%s/trial_%s/'
save_path_template = '../runs/unlearned_models/%s/trial_%s/uncls_%s/'

# Initialize results list to store all results
results = []


models_for_dataset = dataset_model_mapping[args.dataset]
unlearn_method = args.unlearn_method
for model in models_for_dataset:
    set_seed(args.seed)
    trials = args.trials
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    

    raw_train_set, raw_test_set = get_dataset(args.dataset)
    num_classes = len(raw_train_set.classes)

    # get model and training config
    CONFIGS = get_configs(args.dataset, model, 
                n_group=args.n_group, 
                n_unlearn_classes=args.n_unlearn_classes,
                seed=args.seed_unlearn_class)
    unlearn_classes_set = CONFIGS['unlearn_classes_set']

    print("========Argument=======")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print(f'unlearn_classes_set: {unlearn_classes_set}')
    print("======================\n")

    for trial in range(trials):
        print('\n' + '=' * 20 + f'Trial {trial}, set seed {args.seed + trial}' + '=' * 20)
        set_seed(args.seed + trial)

        for unlearn_class in unlearn_classes_set:
            print(f'{"-" * 10} Unlearn class {unlearn_class} {"-" * 10}')
            
            try:
                prepared_data_path = prepared_data_path_template % (args.dataset, trial)
                print(prepared_data_path)
                if not os.path.exists(prepared_data_path):
                    raise ValueError(f'{prepared_data_path} does not exist!')
                save_path = save_path_template % (
                            args.dataset, 
                            trial, 
                            "_".join(map(str, unlearn_class)) if isinstance(unlearn_class, (list, tuple)) \
                            else str(unlearn_class))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                train_idx = np.load(prepared_data_path + 'train_idx.npy')
                val_idx = np.load(prepared_data_path + 'val_idx.npy')
                request_idx = np.load(prepared_data_path + 'request_idx.npy')

                # for simulating the unseen unlearning requests
                request_set = get_subset(raw_train_set, request_idx)
                # for training the original model
                train_set = get_subset(raw_train_set, train_idx)
                val_set = get_subset(raw_train_set, val_idx)

                # prepare unlearning idx
                unlearn_train_idx = get_idx_by_unlearn_class(train_set.targets, [unlearn_class])
                unlearn_val_idx = get_idx_by_unlearn_class(val_set.targets, [unlearn_class])
                unlearn_request_idx = get_idx_by_unlearn_class(request_set.targets, [unlearn_class])
                unlearn_test_idx = get_idx_by_unlearn_class(raw_test_set.targets, [unlearn_class])

                retain_train_idx = np.setdiff1d(np.arange(len(train_set)), unlearn_train_idx)
                retain_val_idx = np.setdiff1d(np.arange(len(val_set)), unlearn_val_idx)
                retain_request_idx = np.setdiff1d(np.arange(len(request_set)), unlearn_request_idx)
                retain_test_idx = np.setdiff1d(np.arange(len(raw_test_set)), unlearn_test_idx)

                # prepare unlearning set
                unlearn_train_set = get_subset(train_set, unlearn_train_idx)
                unlearn_val_set = get_subset(val_set, unlearn_val_idx)
                unlearn_request_set = get_subset(request_set, unlearn_request_idx)
                # prepare retain set
                retain_train_set = get_subset(train_set, retain_train_idx)
                retain_val_set = get_subset(val_set, retain_val_idx)
                retain_request_set = get_subset(request_set, retain_request_idx)
                assert len(unlearn_train_set) + len(retain_train_set) == len(train_set)
                assert len(unlearn_val_set) + len(retain_val_set) == len(val_set)
                assert len(unlearn_request_set) + len(retain_request_set) == len(request_set)

                # prepare dataloader
                test_loader = get_dataloader(raw_test_set, CONFIGS['batch_size'], shuffle=False)

                unlearn_model = copy.deepcopy(CONFIGS['model'])
                model_save_title = f'unlearn_model_{unlearn_method}_{unlearn_model.__class__.__name__}.pt'
                if args.unlearn_method == 'sparse':
                    from unlearn_methods.utils.sparse_util import prune_l1
                    from config_unlearn import get_unlearn_configs
                    n_unlearn_classes = args.n_unlearn_classes
                    UNLEARN_CONFIG = get_unlearn_configs(args.dataset, model, n_unlearn_classes, args.unlearn_method)
                    prune_rate = UNLEARN_CONFIG['prune_rate']
                    prune_l1(unlearn_model, prune_rate)
                    unlearn_model.load_state_dict(torch.load(save_path + model_save_title, weights_only=True), strict=False)
                    for module in unlearn_model.modules():
                        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and torch.nn.utils.prune.is_pruned(module):
                            torch.nn.utils.prune.remove(module, 'weight')
                else:
                    unlearn_model.load_state_dict(torch.load(save_path + model_save_title, weights_only=True))
                unlearn_model.to(device)

                post_net = copy.deepcopy(unlearn_model)
                if args.init_node:
                    # Reinitialize the output head for unlearn class
                    if isinstance(post_net, AllCNN):
                        classifier_weight = post_net.classifier[0].weight.data
                        classifier_bias = post_net.classifier[0].bias.data
                    else:  # ResNet18
                        classifier_weight = post_net.classifier.weight.data
                        classifier_bias = post_net.classifier.bias.data

                    if isinstance(unlearn_class, (list, tuple)):
                        for cls in unlearn_class:
                            nn.init.normal_(classifier_weight[cls:cls+1], mean=0.0, std=0.01)
                            nn.init.zeros_(classifier_bias[cls:cls+1])
                    else:
                        nn.init.normal_(classifier_weight[unlearn_class:unlearn_class+1], mean=0.0, std=0.01)
                        nn.init.zeros_(classifier_bias[unlearn_class:unlearn_class+1])

                post_logits, post_softmax_logits, post_rep, post_pred, ground_truth = get_logits_and_rep(post_net, test_loader, device)
                post_unlearn_softmax_logits = post_softmax_logits[:, [unlearn_class]]

                if args.n_unlearn_classes > 1:
                    init_post_unlearn_softmax_logits = post_unlearn_softmax_logits.squeeze(axis=1)
                    post_unlearn_softmax_logits_sum = init_post_unlearn_softmax_logits.sum(axis=1)
                    post_unlearn_softmax_logits = copy.deepcopy(post_unlearn_softmax_logits_sum)
                else:
                    pass

                post_unlearn_softmax_logits = post_unlearn_softmax_logits.reshape(-1, 1)
                _, transformed_data = yeo_johnson(post_unlearn_softmax_logits)
                cluster_data = transformed_data.flatten()
                threshold = find_otsu_threshold(cluster_data, verbose=False)
                
                # recover
                post_unlearn_idx = np.where(cluster_data > threshold)[0]
                reconstruct_pred = copy.deepcopy(post_pred)
                if args.n_unlearn_classes > 1:
                    infer_post_unlearn_softmax = init_post_unlearn_softmax_logits[post_unlearn_idx, :]
                    unlearn_ground_truth = ground_truth[post_unlearn_idx]
                    selected_unlearned_pred = np.ones(len(post_unlearn_idx))
                    for i in range(len(selected_unlearned_pred)):
                        selected_unlearned_pred[i] = unlearn_class[np.argmax(infer_post_unlearn_softmax[i])]
                    l_idx = 0
                    for i in range(len(post_pred)):
                        if i in post_unlearn_idx:
                            reconstruct_pred[i] = selected_unlearned_pred[l_idx]
                            l_idx += 1
                else:
                    for i in range(len(post_pred)):
                        if i in post_unlearn_idx:
                            reconstruct_pred[i] = unlearn_class
                    unlearn_ground_truth = ground_truth[post_unlearn_idx]
                    selected_unlearned_pred = np.ones(len(post_unlearn_idx)) * unlearn_class

                selected_unlearned_acc = accuracy(selected_unlearned_pred, unlearn_ground_truth) # Recall
                unlearn_test_res = test_all_in_one(post_net, copy.deepcopy(test_loader), unlearn_test_idx, retain_test_idx, unlearn_classes_set, device, output_activation=False)
                unlearn_unlearn_acc = unlearn_test_res['unlearn_acc'][0]
                unlearn_remain_acc = unlearn_test_res['remain_acc'][0]
                unlearn_overall_acc = unlearn_test_res['overall_acc'][0]

                reconstruct_unlearn_pred = reconstruct_pred[unlearn_test_idx]
                reconstruct_unlearn_label = ground_truth[unlearn_test_idx]
                reconstruct_unlearn_acc = accuracy(reconstruct_unlearn_pred, reconstruct_unlearn_label)

                reconstruct_remain_pred = reconstruct_pred[retain_test_idx]
                reconstruct_remain_label = ground_truth[retain_test_idx]
                reconstruct_remain_acc = accuracy(reconstruct_remain_pred, reconstruct_remain_label)
                reconstruct_overall_acc = accuracy(reconstruct_pred, ground_truth)
                print(f'{unlearn_unlearn_acc}\t{unlearn_remain_acc}\t{unlearn_overall_acc}\t{selected_unlearned_acc}\t{reconstruct_unlearn_acc}\t{reconstruct_remain_acc}\t{reconstruct_overall_acc}')
            
                # Store results
                result_row = {
                    'dataset': args.dataset,
                    'model': model,
                    'unlearn_method': args.unlearn_method,
                    'unlearn_class': unlearn_class,
                    'trial': trial,
                    'unlearn_unlearn_acc': unlearn_unlearn_acc,
                    'unlearn_remain_acc': unlearn_remain_acc,
                    'unlearn_overall_acc': unlearn_overall_acc,
                    'selected_unlearned_acc': selected_unlearned_acc,
                    'reconstruct_unlearn_acc': reconstruct_unlearn_acc,
                    'reconstruct_remain_acc': reconstruct_remain_acc,
                    'reconstruct_overall_acc': reconstruct_overall_acc
                }
                results.append(result_row)
                
            except Exception as e:
                print(f'Error occurred for {args.dataset}-{model}-{args.unlearn_method}-{unlearn_class}-trial{trial}: {str(e)}')
                print('Recording N/A results and continuing...')
                
                # Store N/A results for error cases
                result_row = {
                    'dataset': args.dataset,
                    'model': model,
                    'unlearn_method': args.unlearn_method,
                    'unlearn_class': unlearn_class,
                    'trial': trial,
                    'unlearn_unlearn_acc': 'N/A',
                    'unlearn_remain_acc': 'N/A',
                    'unlearn_overall_acc': 'N/A',
                    'selected_unlearned_acc': 'N/A',
                    'reconstruct_unlearn_acc': 'N/A',
                    'reconstruct_remain_acc': 'N/A',
                    'reconstruct_overall_acc': 'N/A'
                }
                results.append(result_row)
                continue
        results.append({key: '' for key in result_row.keys()})
        results.append({key: '' for key in result_row.keys()})

# Store results
df_results = pd.DataFrame(results)
summary_results = []
valid_df = df_results[(df_results['dataset'] != '')]

if not valid_df.empty:
    valid_df = valid_df.copy()
    valid_df['unlearn_class_str'] = valid_df['unlearn_class'].apply(
        lambda x: str(x) if isinstance(x, (list, tuple)) else str(x)
    )
    grouped = valid_df.groupby(['dataset', 'model', 'unlearn_method', 'unlearn_class_str'])
    
    for (dataset, model, unlearn_method, unlearn_class_str), group in grouped:
        if len(group) > 0:
            metrics = ['unlearn_unlearn_acc', 'unlearn_remain_acc', 'unlearn_overall_acc', 
                      'selected_unlearned_acc', 'reconstruct_unlearn_acc', 
                      'reconstruct_remain_acc', 'reconstruct_overall_acc']
            
            original_unlearn_class = group['unlearn_class'].iloc[0]
            
            summary_row = {
                'dataset': dataset,
                'model': model,
                'unlearn_method': unlearn_method,
                'unlearn_class': original_unlearn_class,
                'unlearn_class_str': unlearn_class_str,
                'num_trials': len(group)
            }
            
            for metric in metrics:
                numeric_values = group[metric][group[metric] != 'N/A']
                if len(numeric_values) > 0:
                    summary_row[f'{metric}_mean'] = numeric_values.astype(float).mean()
                    summary_row[f'{metric}_std'] = numeric_values.astype(float).std()
                else:
                    summary_row[f'{metric}_mean'] = 'N/A'
                    summary_row[f'{metric}_std'] = 'N/A'
            
            summary_results.append(summary_row)

df_summary = pd.DataFrame(summary_results)

overall_results = []
if not df_summary.empty:
    overall_grouped = df_summary.groupby(['dataset', 'model', 'unlearn_method'])
    
    for (dataset, model, unlearn_method), group in overall_grouped:
        if len(group) > 0:
            metrics = ['unlearn_unlearn_acc', 'unlearn_remain_acc', 'unlearn_overall_acc', 
                      'selected_unlearned_acc', 'reconstruct_unlearn_acc', 
                      'reconstruct_remain_acc', 'reconstruct_overall_acc']
            
            overall_row = {
                'dataset': dataset,
                'model': model,
                'unlearn_method': unlearn_method,
                'unlearn_class': 'overall',
                'unlearn_class_str': 'overall',
                'num_trials': group['num_trials'].iloc[0]
            }
            
            for metric in metrics:
                numeric_means = group[f'{metric}_mean'][group[f'{metric}_mean'] != 'N/A']
                if len(numeric_means) > 0:
                    overall_row[f'{metric}_mean'] = numeric_means.astype(float).mean()
                    overall_row[f'{metric}_std'] = numeric_means.astype(float).std()
                else:
                    overall_row[f'{metric}_mean'] = 'N/A'
                    overall_row[f'{metric}_std'] = 'N/A'
            
            overall_results.append(overall_row)

all_results = summary_results + overall_results
df_all = pd.DataFrame(all_results)

if not df_all.empty:
    df_all['sort_key'] = df_all['unlearn_class'].apply(lambda x: 999 if x == 'overall' else (int(x) if str(x).isdigit() else 0))
    df_all = df_all.sort_values(['dataset', 'model', 'unlearn_method', 'sort_key']).drop('sort_key', axis=1)

# Save results
output_dir = '../results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if args.n_unlearn_classes > 1:
    output_file = f'recover_raw_multi_{args.n_unlearn_classes}.csv'
else:
    output_file = f'recover_raw_single.csv'

raw_output_file = os.path.join(output_dir, output_file)
summary_output_file = os.path.join(output_dir, f'recover_summary_{args.n_unlearn_classes}.csv')

df_results.to_csv(raw_output_file, index=False)
df_all.to_csv(summary_output_file, index=False)
