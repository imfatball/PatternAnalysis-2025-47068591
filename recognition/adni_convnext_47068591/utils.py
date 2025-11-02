# utils.py (or inline in train.py if you prefer)
import random, math
from collections import defaultdict
from torch.utils.data import Sampler

class RandomSubjectBatchSampler(Sampler):
    """
    Yields batches with at most one slice per subject.
    Expects dataset.samples = [(path, label, subject_id), ...]
    """
    def __init__(self, dataset, batch_size=16, drop_last=False, seed=42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)

        buckets = defaultdict(list)
        for idx, (_p, _y, sid) in enumerate(dataset.samples):
            buckets[sid].append(idx)
        self.buckets = {k: v[:] for k, v in buckets.items()}

    def set_epoch(self, epoch=0):
        rng = random.Random(self.seed + int(epoch))
        # shuffle subject order and per-subject slice order each epoch
        self.subjects = list(self.buckets.keys())
        rng.shuffle(self.subjects)
        self._buckets = {sid: lst[:] for sid, lst in self.buckets.items()}
        for sid in self._buckets:
            rng.shuffle(self._buckets[sid])

    def __iter__(self):
        if not hasattr(self, "_buckets"):
            self.set_epoch(0)
        active = [sid for sid, lst in self._buckets.items() if lst]
        rng = random.Random(self.seed)
        rng.shuffle(active)

        while active:
            pick = active[: self.batch_size]
            batch = [self._buckets[sid].pop() for sid in pick]
            # keep only subjects that still have slices
            active = [sid for sid in active if self._buckets[sid]]
            # refill with other subjects that still have slices
            rest = [sid for sid, lst in self._buckets.items() if lst and sid not in active]
            rng.shuffle(rest)
            active = active + rest
            if len(batch) == self.batch_size:
                yield batch
            elif not self.drop_last and batch:
                yield batch
            if all(len(lst) == 0 for lst in self._buckets.values()):
                break

    def __len__(self):
        n = sum(len(v) for v in self.buckets.values())
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
