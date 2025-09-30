import os
import torch
import numpy as np
import argparse
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from config_original import get_configs
from utils.seed import set_seed
from utils.unlearn_tools import get_idx_by_unlearn_class
from data.dataset import get_dataset, get_subset
from utils import SVC_MIA
from config_unlearn import get_unlearn_configs # For 'sparse' method

def dataset_convert_to_test(dataset, args=None):
    if args.dataset == "tinyimagenet":
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif args.dataset == "cifar10":
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    elif args.dataset == "cifar100":
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                               (0.26733428587941854, 0.25643846292120615, 0.2761504713263903))
        ])
    else:
        test_transform = transforms.Compose([transforms.ToTensor()])
    
    d = dataset
    while hasattr(d, "dataset"):
        d = d.dataset
    d.transform = test_transform
    if hasattr(d, "train"):
        d.train = False

def extract_targets_from_dataset(args, dataset):
    if args.dataset == 'mnist' or args.dataset == 'mnistFashion':
        targets = dataset.targets
        targets = targets.numpy()
        return targets
    
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        targets = dataset.targets
        return targets

# ----------------------------
# Metric helpers
# ----------------------------
def _tpr_at(fpr: np.ndarray, tpr: np.ndarray, fpr_target: float) -> float:
    if fpr_target <= fpr[0]: return float(tpr[0])
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    return float(tpr[np.clip(idx, 0, len(tpr) - 1)])

def mia_metrics(scores_member, scores_nonmember, higher_is_member=True):
    """Calculate AUROC, AUPRC and other metrics for given MIA scores."""
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    s_m = np.asarray(scores_member, dtype=np.float64).flatten()
    s_n = np.asarray(scores_nonmember, dtype=np.float64).flatten()

    if not higher_is_member: s_m, s_n = -s_m, -s_n

    y = np.concatenate([np.ones_like(s_m), np.zeros_like(s_n)])
    s = np.concatenate([s_m, s_n])

    if len(np.unique(y)) < 2: return {"AUROC": 0.5, "AUPRC": 0.5, "TPR@1FPR": 0.0, "TPR@5FPR": 0.0}

    auroc = roc_auc_score(y, s)
    auprc = average_precision_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    
    return {"AUROC": float(auroc), "AUPRC": float(auprc), "TPR@1FPR": _tpr_at(fpr, tpr, 0.01), "TPR@5FPR": _tpr_at(fpr, tpr, 0.05)}

def evaluate_mia_auc(args, model, train_set, test_set, 
        retain_train_idx, retain_test_idx, unlearn_train_idx, unlearn_test_idx, 
        kinds=None, batch_size=256, num_workers=4, verbose=True):
    dataset_convert_to_test(train_set, args=args)
    dataset_convert_to_test(test_set, args=args)

    forget_member_loader = _make_loader(train_set, unlearn_train_idx, batch_size, num_workers)
    forget_nonmember_loader = _make_loader(test_set, unlearn_test_idx, batch_size, num_workers)
    retain_member_loader = _make_loader(train_set, retain_train_idx, batch_size, num_workers)
    retain_nonmember_loader = _make_loader(test_set, retain_test_idx, batch_size, num_workers)

    feats_f_member = _build_features(forget_member_loader, model, kinds=kinds)
    feats_f_nonmember = _build_features(forget_nonmember_loader, model, kinds=kinds)
    feats_r_member = _build_features(retain_member_loader, model, kinds=kinds)
    feats_r_nonmember = _build_features(retain_nonmember_loader, model, kinds=kinds)

    results = {"F": {}, "R": {}}

    for k in kinds:
        higher_is_member = (k in ["confidence", "correctness"])
        resF = mia_metrics(feats_f_member[k], feats_f_nonmember[k], higher_is_member=higher_is_member)
        resR = mia_metrics(feats_r_member[k], feats_r_nonmember[k], higher_is_member=higher_is_member)
        results["F"][k] = resF
        results["R"][k] = resR
        if verbose: print(f"[{k:<12}] F-AUC={resF['AUROC']:.3f} | R-AUC={resR['AUROC']:.3f} "
                      f"| F-TPR@5%={resF['TPR@5FPR']:.3f} | R-TPR@5%={resR['TPR@5FPR']:.3f}")
    return results

def _make_loader(dataset, indices, batch_size, num_workers=4):
    if indices is None or len(indices) == 0: return None
    return DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def _to_2d(x):
    if x.ndim == 1: x = x.reshape(-1, 1)
    return x

def _build_features(loader, model, kinds=None):
    if loader is None:
        return {k: np.array([]) for k in kinds}
    prob, labels = SVC_MIA.collect_prob(loader, model)
    if prob.numel() == 0:
        return {k: np.array([]) for k in kinds}
        
    corr = (torch.argmax(prob, dim=1) == labels).int()
    conf = torch.gather(prob, 1, labels[:, None])
    entr = SVC_MIA.entropy(prob)
    loss = -(conf.clamp_min(1e-12)).log()
    
    tmp = prob.clone()
    tmp[torch.arange(len(labels)), labels] = -1.0
    p_second = tmp.max(dim=1, keepdim=True).values
    margin = conf - p_second
    
    corr = corr.unsqueeze(1).float()
    entr = entr.unsqueeze(1)
    fusion = torch.cat([corr, conf, entr, loss, margin], dim=1)
    
    return {
        "correctness": _to_2d(corr.cpu().numpy()), "confidence": _to_2d(conf.cpu().numpy()),
        "entropy": _to_2d(entr.cpu().numpy()), "loss": _to_2d(loss.cpu().numpy()),
        "margin": _to_2d(margin.cpu().numpy()), "fusion": _to_2d(fusion.cpu().numpy())
        }

def _loader_size(loader):
      return 0 if loader is None else len(loader.dataset)

def run_mia(args, model, train_set, test_set, 
            retain_train_idx, retain_test_idx, 
            unlearn_train_idx, unlearn_test_idx, 
            batch_size, num_workers, mia_trials=5):
    dataset_convert_to_test(train_set, args=args)
    dataset_convert_to_test(test_set, args=args)

    shadow_train_loader = _make_loader(train_set, retain_train_idx, batch_size, num_workers)
    shadow_test_loader = _make_loader(test_set, retain_test_idx, batch_size, num_workers)
    forget_train_loader = _make_loader(train_set, unlearn_train_idx, batch_size, num_workers)
    
    results = {}
    
    results["forget_efficacy"] = SVC_MIA.SVC_MIA(
        shadow_train=shadow_train_loader,
        shadow_test=shadow_test_loader,
        target_train=None,
        target_test=forget_train_loader,
        model=model,
    )
    
    privacy_results_list = []
    
    if len(retain_train_idx) > 1 and len(retain_test_idx) > 1:
        try:
            all_train_targets = extract_targets_from_dataset(args, train_set)
            retain_train_labels = all_train_targets[retain_train_idx]
            retain_test_labels = extract_targets_from_dataset(args, test_set)[retain_test_idx]

            # print(f"--- Running {mia_trials} internal MIA trials for privacy evaluation ---")
            for i in range(mia_trials):
                seed = args.seed_unlearn_class + i
                shadow_privacy_train_idx, target_privacy_train_idx = train_test_split(
                    retain_train_idx, test_size=0.5, random_state=seed, stratify=retain_train_labels)
                shadow_privacy_test_idx, target_privacy_test_idx = train_test_split(
                    retain_test_idx, test_size=0.5, random_state=seed, stratify=retain_test_labels)

                shadow_privacy_train_loader = _make_loader(train_set, shadow_privacy_train_idx, batch_size, num_workers)
                shadow_privacy_test_loader = _make_loader(test_set, shadow_privacy_test_idx, batch_size, num_workers)
                target_privacy_train_loader = _make_loader(train_set, target_privacy_train_idx, batch_size, num_workers)
                target_privacy_test_loader = _make_loader(test_set, target_privacy_test_idx, batch_size, num_workers)

                if all(_loader_size(l) > 0 for l in [shadow_privacy_train_loader, shadow_privacy_test_loader, target_privacy_train_loader, target_privacy_test_loader]):
                    trial_result = SVC_MIA.SVC_MIA(
                        shadow_train=shadow_privacy_train_loader, shadow_test=shadow_privacy_test_loader,
                        target_train=target_privacy_train_loader, target_test=target_privacy_test_loader,
                        model=model)
                    privacy_results_list.append(trial_result)

            if privacy_results_list:
                df_privacy = pd.DataFrame(privacy_results_list)
                mean_privacy_results = df_privacy.mean().to_dict()
                results["training_privacy_on_retain"] = mean_privacy_results
            else:
                results["training_privacy_on_retain"] = None

        except Exception as e:
            print(f"Warning: Could not perform privacy evaluation due to an error: {e}")
            results["training_privacy_on_retain"] = None
    else:
        print("Warning: Not enough data in retain set to perform privacy evaluation.")
        results["training_privacy_on_retain"] = None
        
    return results

def print_mia_results(results: dict, kinds: list):
    def _print_kv(title: str, data: dict):
        if data and isinstance(data, dict):
            print(title)
            for k in kinds:
                if k in data and data.get(k) is not np.nan:
                    print(f"  {k:<12}: {data[k]:.3f}")
    
    _print_kv("\n== Forget Efficacy (TNR on forgotten data) ==", results.get("forget_efficacy"))
    _print_kv("== Training Privacy (Attack Acc on retain data) ==", results.get("training_privacy_on_retain"))

def main():
    parser = argparse.ArgumentParser(description='MIA Analysis for Unlearning Methods')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'cifar100'])
    parser.add_argument('--model', type=str, default='AllCNN', choices=['AllCNN', 'ResNet18', 'ResNet34'])
    parser.add_argument('--unlearn-method', type=str, default='retrain')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--n_group', type=int, default=10)
    parser.add_argument('--n_unlearn_classes', type=int, default=1)
    parser.add_argument('--seed_unlearn_class', type=int, default=3407)
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    methods_list = [
    'retrain',
    'embedding_shift',
    'boundary_shrink',
    'boundary_expanding',
    'unrolling',
    'unrolling_f',
    'unsc',
    'salun',
    'ga',
    'fisher',
    'BadT', 
    'sparse', 
    'scrub'
    ]
    dataset_model_mapping = {
    'mnist': ['AllCNN'],
    'mnistFashion': ['AllCNN'],
    'cifar10': ['ResNet18'],
    'cifar100': ['ResNet34']
    }
    
    prepared_data_path_template = './runs/prepared_data/%s/trial_%s/'
    save_path_template = './runs/unlearned_models/%s/trial_%s/uncls_%s/'

    all_rows = []
    # for dataset in ['mnist', 'mnistFashion', 'cifar10', 'cifar100']:
    for dataset in ['cifar100']:
        if args.n_unlearn_classes > 1:
            if dataset == 'cifar100':
                args.n_unlearn_classes = 10
            else:
                args.n_unlearn_classes = 2

        models_for_dataset = dataset_model_mapping[dataset]
        for model in models_for_dataset:
            for unlearn_method in methods_list:
                args.model = model
                args.unlearn_method = unlearn_method
                args.dataset = dataset
                seed = args.seed
                set_seed(seed)
                trials = args.trials
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                raw_train_set, raw_test_set = get_dataset(dataset)
                CONFIGS = get_configs(dataset, args.model, 
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
                    print('\n' + '=' * 20 + f'Trial {trial}, set seed {seed + trial}' + '=' * 20)
                    set_seed(seed + trial)

                    for unlearn_class in unlearn_classes_set:
                        print(f'{"-" * 10} Unlearn class {unlearn_class} {"-" * 10}')
                        prepared_data_path = prepared_data_path_template % (dataset, trial)
                        if not os.path.exists(prepared_data_path):
                            raise ValueError(f'{prepared_data_path} does not exist!')
                        save_path = save_path_template % (
                                dataset, 
                                trial, 
                                "_".join(map(str, unlearn_class)) if isinstance(unlearn_class, (list, tuple)) \
                                else str(unlearn_class))
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        model_path = os.path.join(save_path, f'unlearn_model_{unlearn_method}_{CONFIGS["model"].__class__.__name__}.pt')

                        train_idx = np.load(os.path.join(prepared_data_path, 'train_idx.npy'))
                        train_set = get_subset(raw_train_set, train_idx)

                        unlearn_train_idx = get_idx_by_unlearn_class(train_set.targets, [unlearn_class])
                        unlearn_test_idx = get_idx_by_unlearn_class(raw_test_set.targets, [unlearn_class])
                        retain_train_idx = np.setdiff1d(np.arange(len(train_set)), unlearn_train_idx)
                        retain_test_idx = np.setdiff1d(np.arange(len(raw_test_set)), unlearn_test_idx)

                        unlearn_model = copy.deepcopy(CONFIGS['model'])
                        if args.unlearn_method == 'sparse':
                            from unlearn_methods.utils.sparse_util import prune_l1
                            UNLEARN_CONFIG = get_unlearn_configs(dataset, args.model, args.n_unlearn_classes, unlearn_method)
                            prune_l1(unlearn_model, UNLEARN_CONFIG['prune_rate'])
                            unlearn_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device), strict=False)
                            for module in unlearn_model.modules():
                                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and torch.nn.utils.prune.is_pruned(module):
                                    torch.nn.utils.prune.remove(module, 'weight')
                        else:
                            unlearn_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
                        unlearn_model.to(device)

                        # ----------------------------
                        # MIA Analysis
                        # ----------------------------
                        # MIA features
                        kinds = ["correctness", "confidence", "entropy", "loss", "margin", "fusion"]

                        print("--- Performing AUC-based MIA Analysis ---")
                        results_auc = evaluate_mia_auc(
                            args=args, model=unlearn_model, train_set=train_set, test_set=raw_test_set,
                            retain_train_idx=retain_train_idx, retain_test_idx=retain_test_idx,
                            unlearn_train_idx=unlearn_train_idx, unlearn_test_idx=unlearn_test_idx, 
                            kinds=kinds, verbose=args.verbose)
                        
                        print("--- Performing SVC-based MIA Analysis ---")
                        results_mia = run_mia(
                            args=args, model=unlearn_model, train_set=train_set, test_set=raw_test_set,
                            retain_train_idx=retain_train_idx, retain_test_idx=retain_test_idx,
                            unlearn_train_idx=unlearn_train_idx, unlearn_test_idx=unlearn_test_idx,
                            batch_size=256, num_workers=4, mia_trials=args.trials)
                        print_mia_results(results_mia, kinds)

                        def _safe_float(v):
                            try: return float(v)
                            except (ValueError, TypeError): return np.nan

                        unlearn_class_str = str(unlearn_class) if isinstance(unlearn_class, list) else str(unlearn_class)
                        row = {'dataset': dataset, 'model': model, 'unlearn_method': unlearn_method,
                               'unlearn_class': unlearn_class_str, 'trial': trial}

                        fe_m = results_mia.get('forget_efficacy', {}) or {}
                        tpr_m = results_mia.get('training_privacy_on_retain', {}) or {}
                        
                        for k in kinds:
                            Fk, Rk = results_auc.get('F', {}).get(k, {}), results_auc.get('R', {}).get(k, {})
                            row[f'F_{k}_AUC'] = _safe_float(Fk.get('AUROC'))
                            row[f'R_{k}_AUC'] = _safe_float(Rk.get('AUROC'))
                            row[f'F_{k}_TPR5'] = _safe_float(Fk.get('TPR@5FPR'))
                            row[f'R_{k}_TPR5'] = _safe_float(Rk.get('TPR@5FPR'))
                        for k in kinds:
                            row[f'fe_{k}'] = _safe_float(fe_m.get(k))
                            row[f'tpr_{k}'] = _safe_float(tpr_m.get(k))
                        all_rows.append(row)

    if all_rows:
        df = pd.DataFrame(all_rows)
        output_dir = './results'
        os.makedirs(output_dir, exist_ok=True)

        if args.n_unlearn_classes > 1:
            raw_path = os.path.join(output_dir, f'recover_logits_raw_MIA_multi_class_{args.dataset}_v2.csv')
        else:
            raw_path = os.path.join(output_dir, f'recover_logits_raw_MIA_{args.dataset}_v2.csv')
        df.to_csv(raw_path, index=False)
        print(f"\nSaved all raw MIA results to: {raw_path}")

        metric_cols = [col for col in df.columns if col not in ['dataset', 'model', 'unlearn_method', 'unlearn_class', 'trial']]

        groupby_keys_per_class = ['dataset', 'model', 'unlearn_method', 'unlearn_class']
        summary_per_class_rows = []
        for keys, group in df.groupby(groupby_keys_per_class):
            row = dict(zip(groupby_keys_per_class, keys))
            row['num_trials'] = len(group['trial'].unique())
            for col in metric_cols:
                vals = pd.to_numeric(group[col], errors='coerce').dropna()
                row[f'{col}_mean'] = vals.mean()
                row[f'{col}_std'] = vals.std()
            summary_per_class_rows.append(row)
        df_summary_per_class = pd.DataFrame(summary_per_class_rows)

        groupby_keys_overall = ['dataset', 'model', 'unlearn_method']
        overall_summary_rows = []
        for keys, group in df.groupby(groupby_keys_overall):
            row = dict(zip(groupby_keys_overall, keys))
            row['unlearn_class'] = 'overall' 
            row['num_trials'] = group['trial'].nunique() 
            for col in metric_cols:
                vals = pd.to_numeric(group[col], errors='coerce').dropna()
                row[f'{col}_mean'] = vals.mean() 
                row[f'{col}_std'] = vals.std()
            overall_summary_rows.append(row)
        df_summary_overall = pd.DataFrame(overall_summary_rows)

        if not df_summary_per_class.empty and not df_summary_overall.empty:
            df_combined = pd.concat([df_summary_per_class, df_summary_overall], ignore_index=True)
            df_combined['unlearn_class_str'] = df_combined['unlearn_class'].astype(str)
            df_combined = df_combined.sort_values(by=['dataset', 'model', 'unlearn_method', 'unlearn_class_str']).drop(columns=['unlearn_class_str'])

            if args.n_unlearn_classes > 1:
                combined_path = os.path.join(output_dir, f'recover_logits_summary_MIA_multi_class_{args.dataset}_v2.csv')
            else:
                combined_path = os.path.join(output_dir, f'recover_logits_summary_MIA_{args.dataset}_v2.csv')
            df_combined.to_csv(combined_path, index=False)
            print(f"Saved combined summary to: {combined_path}")
        else:
            print("Warning: Could not generate combined summary due to missing data.")

if __name__ == '__main__':
    main()