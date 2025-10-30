import random
from torch.utils.data import Sampler
from collections import defaultdict

class BalancedClassSubjectSampler(Sampler[int]):
    """Each batch: half AD subjects, half NC subjects (1 slice per subject)."""
    def __init__(self, dataset, batch_size: int, drop_last: bool = True, seed: int = 42):
        super().__init__(None)
        assert batch_size % 2 == 0, "Use an even batch size for balanced sampling."
        self.ds = dataset
        self.bs = batch_size
        self.half = batch_size // 2
        self.drop_last = drop_last
        self.rng = random.Random(seed)

        sid2idxs = defaultdict(list)
        sid2label = {}
        for i, (_p, y, sid) in enumerate(self.ds.samples):
            sid2idxs[sid].append(i)
            sid2label[sid] = y

        self.pos_sids = [sid for sid, y in sid2label.items() if y == 1]
        self.neg_sids = [sid for sid, y in sid2label.items() if y == 0]
        self.sid2idxs = sid2idxs

    def __iter__(self):
        pos = self.pos_sids[:]; neg = self.neg_sids[:]
        self.rng.shuffle(pos); self.rng.shuffle(neg)
        i = j = 0
        while i + self.half <= len(pos) and j + self.half <= len(neg):
            batch = []
            for sid in pos[i:i+self.half]:
                batch.append(self.rng.choice(self.sid2idxs[sid]))
            for sid in neg[j:j+self.half]:
                batch.append(self.rng.choice(self.sid2idxs[sid]))
            self.rng.shuffle(batch)
            for k in batch: yield k
            i += self.half; j += self.half
        # ignore tail if drop_last else yield leftover (optional)
    def __len__(self):
        n_batches = min(len(self.pos_sids), len(self.neg_sids)) // self.half
        return n_batches * self.bs