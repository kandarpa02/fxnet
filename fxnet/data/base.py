from typing import Tuple
import math
from ..backend.backend import xp
from ..src.ndarray.base import array

lib = xp()  # backend: numpy or cupy

# ---------------------------------------
# Utility to check all arrays have same length
def _check_lengths(arrays):
    n = len(arrays[0])
    for a in arrays:
        if len(a) != n:
            raise ValueError("All input arrays must have the same length")
    return n

# ---------------------------------------
class ArrayLoader:
    def __init__(
        self,
        *arrays,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        split: Tuple[int, ...] | None = None,
        part: int = 0,
        seed: int | None = None,
    ):
        self.arrays = arrays
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = lib.random.default_rng(seed)

        # ---- validate ----
        n = _check_lengths(arrays)

        # ---- base indices ----
        indices = lib.arange(n, dtype=lib.int64)

        # ---- split ----
        if split is not None:
            if not all(isinstance(s, int) and s > 0 for s in split):
                raise ValueError("split must be positive integers")

            total = sum(split)
            if total > 100:
                raise ValueError("split percentages must sum to <= 100")

            if total < 100:
                split = split + (100 - total,)

            sizes = [(n * s) // 100 for s in split]
            sizes[-1] = n - sum(sizes[:-1])

            # ---- GPU/CPU-safe conversion ----
            sizes_array = lib.array(sizes, dtype=lib.int64)
            bounds = lib.cumsum(lib.concatenate([lib.array([0], dtype=lib.int64), sizes_array]))

            if part >= len(sizes):
                raise ValueError("part index out of range")

            indices = indices[bounds[part]:bounds[part + 1]]

        self.base_indices = indices
        self.num_samples = len(indices)

        # ---- batches ----
        if drop_last:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = math.ceil(self.num_samples / self.batch_size)

        # ---- epoch state ----
        self._epoch_indices = None
        self.reset()

    # -------------------------------------------------
    def reset(self):
        """Call once per epoch"""
        if self.shuffle:
            self._epoch_indices = lib.random.permutation(self.base_indices)
        else:
            self._epoch_indices = self.base_indices


    # -------------------------------------------------
    def __len__(self):
        return self.num_batches

    # -------------------------------------------------
    def __getitem__(self, batch_idx: int):
        if batch_idx < 0 or batch_idx >= self.num_batches:
            raise IndexError("batch index out of range")

        start = batch_idx * self.batch_size
        end = start + self.batch_size

        if start >= self.num_samples:
            raise IndexError("batch index out of range")

        batch_idx = self._epoch_indices[start:end]

        # ---- GPU/CPU-safe batch allocation ----
        batch = [
            array(lib.take(a.np, batch_idx, axis=0))
            for a in self.arrays
        ]

        return batch[0] if len(batch) == 1 else batch

def array_loader(
        *args,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        split: tuple | None = None,
        part: int = 0,
    ):
    
    loader = ArrayLoader(*args, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, split=split, part=part)
    return loader
