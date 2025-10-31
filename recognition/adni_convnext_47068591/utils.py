# utils.py
import random
import math
from collections import defaultdict
from torch.utils.data import Sampler

class RandomSubjectBatchSampler(Sampler):
    """
    Yields *batches of indices* where each batch contains at most one slice
    per subject. It cycles through subjects until all slices are consumed.

    - dataset.samples must be [(path, label, subject_id), ...]
    - batch_size: number of distinct subjects per batch
    - drop_last: drop the last incomplete batch
    """
    def __init__(self, dataset, batch_size=16, drop_last=False, seed=42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.rng = random.Random(seed)

        # Build subject -> list of indices
        buckets = defaultdict(list)
        for idx, (_p, _y, sid) in enumerate(dataset.samples):
            buckets[sid].append(idx)

        # Shuffle indices within each subject bucket
        for sid in buckets:
            self.rng.shuffle(buckets[sid])

        self.buckets = dict(buckets)
        self.subjects = list(self.buckets.keys())
        self.num_samples = sum(len(v) for v in self.buckets.values())

    def __iter__(self):
        # Work on local copies so multiple epochs are independent
        buckets = {sid: lst[:] for sid, lst in self.buckets.items()}

        # Set of subjects that still have slices left
        active = [sid for sid, lst in buckets.items() if len(lst) > 0]
        self.rng.shuffle(active)

        while active:
            # Choose up to batch_size distinct subjects for this batch
            pick = active[: self.batch_size]
            batch = []

            # Pop one slice per chosen subject
            for sid in pick:
                batch.append(buckets[sid].pop())

            # Remove subjects that ran out of slices; keep others in pool
            active = [sid for sid in active if len(buckets[sid]) > 0]

            # Refill the pool with remaining subjects (shuffled) if it's small
            if len(active) < self.batch_size:
                # Bring in any other subjects that still have slices
                rest = [sid for sid, lst in buckets.items() if len(lst) > 0 and sid not in active]
                self.rng.shuffle(rest)
                active = active + rest

            # Yield this batch
            if len(batch) == self.batch_size:
                yield batch
            else:
                if not self.drop_last and len(batch) > 0:
                    yield batch
                # If drop_last, just discard the small tail and finish.

            # Stop if no slices left anywhere
            if all(len(lst) == 0 for lst in buckets.values()):
                break

    def __len__(self):
        # Number of *batches* per epoch
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)
