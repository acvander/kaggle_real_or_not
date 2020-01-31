from typing import Dict

from sklearn.model_selection import StratifiedKFold


def k_fold_split(train_data: Dict, shuffle: bool = False):
    x = range(len(train_data['keyword']))
    y = train_data['target'][:, 0]
    splitter = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=2020)
    splits = list(splitter.split(x, y))
    return splits
