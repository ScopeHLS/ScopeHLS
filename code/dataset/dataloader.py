from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random

from config.config import CONFIG

def split_dataset_by_groups(dataset, val_ratio, seed=CONFIG.random_seed, stage2=False):
    random.seed(seed)

    grouped_indices = defaultdict(lambda: defaultdict(lambda: {'valid': [], 'invalid': []}))

    for idx in range(len(dataset)):
        item = dataset[idx]
        version = item['version']
        kernel = item['kernel_name']
        is_invalid = (item['latency'] == 0)

        if stage2 and is_invalid:
            continue

        key = 'invalid' if is_invalid else 'valid'
        grouped_indices[version][kernel][key].append(idx)

    train_indices, val_indices, test_indices = [], [], []

    for version, kernels in grouped_indices.items():
        for kernel, split_dict in kernels.items():
            for key, indices in split_dict.items():
                if len(indices) < CONFIG.min_samples_per_group:
                    train_indices.extend(indices)
                    continue

                train, temp = train_test_split(indices, test_size=val_ratio, random_state=seed)
                val, test = train_test_split(temp, test_size=0.25, random_state=seed)
                train_indices.extend(train)
                val_indices.extend(val)
                test_indices.extend(test)

    return train_indices, val_indices, test_indices

def get_dataloaders(dataset, batch_size=8, seed=CONFIG.random_seed, shuffle=True, num_workers=0):
    train_idx, val_idx, test_idx = split_dataset_by_groups(dataset, val_ratio=CONFIG.validation_ratio, seed=seed)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return train_loader, val_loader, test_loader

def get_dataloaders_stage2(dataset, batch_size=8, seed=CONFIG.random_seed, shuffle=True, num_workers=0):
    train_idx, val_idx, test_idx = split_dataset_by_groups(dataset, val_ratio=CONFIG.validation_ratio, seed=seed)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return train_loader, val_loader, test_loader

def get_dataloaders_fine_tune(dataset, batch_size=8, seed=CONFIG.random_seed, shuffle=True, num_workers=0):
    train_idx, val_idx, test_idx = split_dataset_by_groups(dataset, val_ratio=CONFIG.validation_ratio_fine_tune, seed=seed)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return train_loader, val_loader, test_loader
