"""Block-aware chronological cross-validation splitter.

`ChronoGroupsSplit` holds one block per class out per fold so each fold gets
balanced class coverage in test while preserving chronological grouping in
train. Consumed by ``run_grid_search_cv`` and ``run_permutation_test``.
"""

import numpy as np


class ChronoGroupsSplit:

    def __init__(
        self,
        warn_if_blocks_ignored: bool = False,
        allow_mixed_label_groups: bool = False,
    ):
        self.warn_if_blocks_ignored = warn_if_blocks_ignored
        self.allow_mixed_label_groups = allow_mixed_label_groups

    def split(self, X, y, groups):
        y = np.asarray(y)
        groups = np.asarray(groups)
        Xidcs = np.arange(len(y))

        if self.allow_mixed_label_groups:
            unique_groups = sorted(set(groups))
            splits = []
            for test_group in unique_groups:
                test_mask = groups == test_group
                train_mask = ~test_mask
                splits.append((Xidcs[train_mask], Xidcs[test_mask]))
            return splits

        gm = {k: list(set(groups[y == k])) for k in set(y)}
        for v in gm.values():
            v.sort()
        assert all(
            [set(s).intersection(groups[y != k]) == set() for k, s in gm.items()]
        ), "Groups are not unique in label"

        set_lens = [len(v) for v in gm.values()]
        if not all([e == set_lens[0] for e in set_lens]):
            min_set = min(set_lens)
            # Drop surplus groups beyond the minority class count.
            gm = {k: v[:min_set] for k, v in gm.items()}

        test_idxs = np.asarray(list(zip(*[v for v in gm.values()])))
        splits = [
            (
                Xidcs[~np.isin(groups, test_idx_row)],
                Xidcs[np.isin(groups, test_idx_row)],
            )
            for test_idx_row in test_idxs
        ]
        return splits

    def get_n_splits(self, X=None, y=None, groups=None):
        if y is None or groups is None:
            raise ValueError("y and groups must be provided to get n_splits")
        y = np.asarray(y)
        groups = np.asarray(groups)
        if self.allow_mixed_label_groups:
            return len(set(groups))
        gm = {k: list(set(groups[y == k])) for k in set(y)}
        return min(len(v) for v in gm.values())
