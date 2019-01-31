from fastai.text import *

bs=256
bptt=250

data = TextLMDataBunch.load(path, cache_name='tmp_clc', bs=bs, bptt=bptt)
t = data.train_ds[0][0]
t.text[:50], t.data

learn = language_model_learner(data, drop_mult=1, clip=.2, bptt=bptt)
learn.unfreeze()

learn.fit_one_cycle(5, 1e-2, moms=(0.7,0.5))

learn.save('clc_first_run')