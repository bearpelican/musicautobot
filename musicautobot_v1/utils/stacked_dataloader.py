"Dataloader wrapper that can combine and handle multiple dataloaders for multitask training"
from fastai.callback import Callback
from typing import Callable

__all__ = ['StackedDataBunch']

# DataLoading
class StackedDataBunch():
    def __init__(self, dbs, num_it=100):
        self.dbs = dbs
        self.train_dl = StackedDataloader([db.train_dl for db in self.dbs], num_it)
        self.valid_dl = StackedDataloader([db.valid_dl for db in self.dbs], num_it)
        self.train_ds = None
        self.path = dbs[0].path
        self.device = dbs[0].device
        self.vocab = dbs[0].vocab
        self.empty_val = False

    def add_tfm(self,tfm:Callable)->None:
        for dl in self.dbs: dl.add_tfm(tfm)

    def remove_tfm(self,tfm:Callable)->None:
        for dl in self.dbs: dl.remove_tfm(tfm)

# Helper functions
class StackedDataset(Callback):
    def __init__(self, dss):
        self.dss = dss
    def __getattribute__(self, attr):
        if attr == 'dss': return super().__getattribute__(attr)
        def redirected(*args, **kwargs):
            for ds in self.dss:
                if hasattr(ds, attr): getattr(ds, attr)(*args, **kwargs)
        return redirected
    def __len__(self)->int: return sum([len(ds) for ds in self.dss])
    def __repr__(self): return '\n'.join([self.__class__.__name__] + [repr(ds) for ds in self.dss])

class StackedDataloader():
    def __init__(self, dls, num_it=100):
        self.dls = dls
        self.dataset = StackedDataset([dl.dataset for dl in dls if hasattr(dl, 'dataset')])
        self.num_it = num_it
        self.dl_idx = -1
        
    def __len__(self)->int: return sum([len(dl) for dl in self.dls])
    def __getattr__(self, attr):
        def redirected(*args, **kwargs):
            for dl in self.dls:
                if hasattr(dl, attr):
                    getattr(dl, attr)(*args, **kwargs)
        return redirected
        
    def __iter__(self):
        "Process and returns items from `DataLoader`."
        iters = [iter(dl) for dl in self.dls]
        self.dl_idx = -1
        while len(iters):
            self.dl_idx = (self.dl_idx+1) % len(iters)
            for b in range(self.num_it):
                try:
                    yield next(iters[self.dl_idx])
                except StopIteration as e:
                    iters.remove(iters[self.dl_idx])
                    break
#         raise StopIteration

    def new(self, **kwargs):
        "Create a new copy of `self` with `kwargs` replacing current values."
        new_dls = [dl.new(**kwargs) for dl in self.dls]
        return StackedDataloader(new_dls, self.num_it)
