"""
Subset selection strategies for upweighting experiments.
Drop this file into the sft/ directory alongside utils.py.
"""
from collections import Counter
import numpy as np


def select_monofact_subset(train_dataset, subset_fraction=0.05, seed=1217):
    """
    Select examples that are monofacts (appear exactly once) in the training set.
    
    Args:
        train_dataset: HF Dataset with columns ["x", "y", "names", "gold", ...]
        subset_fraction: fraction of dataset to select (default 0.05 = 5%)
        seed: random seed for reproducibility
    
    Returns:
        list of indices into train_dataset to use for upweighting
    """
    rng = np.random.default_rng(seed)
    subset_size = int(subset_fraction * len(train_dataset))
    
    # Count frequency of each unique example by its target text (y)
    ys = train_dataset["y"]
    y_counts = Counter(ys)
    
    # Group indices by frequency
    freq_to_indices = {}
    for idx, y in enumerate(ys):
        freq = y_counts[y]
        if freq not in freq_to_indices:
            freq_to_indices[freq] = []
        freq_to_indices[freq].append(idx)
    
    # Start with monofacts (freq=1), then fill from next-rarest if needed
    selected = []
    for freq in sorted(freq_to_indices.keys()):
        candidates = freq_to_indices[freq]
        if len(selected) + len(candidates) <= subset_size:
            selected.extend(candidates)
        else:
            remaining = subset_size - len(selected)
            selected.extend(rng.choice(candidates, size=remaining, replace=False).tolist())
            break
    
    # If we somehow don't have enough (very unlikely), pad from all remaining
    if len(selected) < subset_size:
        all_indices = set(range(len(train_dataset)))
        remaining_indices = list(all_indices - set(selected))
        needed = subset_size - len(selected)
        selected.extend(rng.choice(remaining_indices, size=needed, replace=False).tolist())
    
    n_monofacts = len([i for i in selected if y_counts[ys[i]] == 1])
    print(f"[MonofactSelect] Selected {len(selected)} examples for upweighting")
    print(f"[MonofactSelect] {n_monofacts}/{len(selected)} are true monofacts (freq=1)")
    print(f"[MonofactSelect] Total monofacts in training set: {len(freq_to_indices.get(1, []))}/{len(train_dataset)}")
    
    return selected


def select_random_subset(train_dataset, subset_fraction=0.05, seed=1217):
    """
    Select a random subset of examples (proper random, not positional like the original code).
    
    Args:
        train_dataset: HF Dataset with columns ["x", "y", "names", "gold", ...]
        subset_fraction: fraction of dataset to select (default 0.05 = 5%)
        seed: random seed for reproducibility
    
    Returns:
        list of indices into train_dataset to use for upweighting
    """
    rng = np.random.default_rng(seed)
    subset_size = int(subset_fraction * len(train_dataset))
    selected = rng.choice(len(train_dataset), size=subset_size, replace=False).tolist()
    
    # Report how many of the randomly selected happen to be monofacts
    ys = train_dataset["y"]
    y_counts = Counter(ys)
    n_monofacts = len([i for i in selected if y_counts[ys[i]] == 1])
    print(f"[RandomSelect] Selected {len(selected)} examples for upweighting")
    print(f"[RandomSelect] {n_monofacts}/{len(selected)} happen to be monofacts")
    
    return selected
