# ==================================================================================
# RandomSubjectSampler
# ----------------------------------------------------------------------------------
# Ensures each batch contains slices from different subjects to reduce correlation.
# ==================================================================================
import random
from torch.utils.data import Sampler

class RandomSubjectSampler(Sampler[int]):
    def __init__(self, dataset, batch_size: int, drop_last: bool = True, seed: int = 42):
        super().__init__(None)
        self.ds = dataset
        self.bs = batch_size
        self.drop_last = drop_last
        self.rng = random.Random(seed)
        self.sid2idxs = {}
        for i, (_p, _y, sid) in enumerate(self.ds.samples):
            self.sid2idxs.setdefault(sid, []).append(i)
        self.subjects = list(self.sid2idxs.keys())

    def __iter__(self):
        subs = self.subjects[:]
        self.rng.shuffle(subs)
        batch = []
        for sid in subs:
            batch.append(self.rng.choice(self.sid2idxs[sid]))
            if len(batch) == self.bs:
                for j in batch:
                    yield j
                batch = []
        if batch and not self.drop_last:
            for j in batch:
                yield j

    def __len__(self):
        n = len(self.subjects)
        return n - (n % self.bs) if self.drop_last else n
