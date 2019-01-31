from fastai.text import *
import gpt

bs=8
bptt=500

data = TextLMDataBunch.load(path, cache_name='tmp_clc', bs=bs, bptt=bptt)

vocab = data.train_ds.vocab
vocab_size = len(vocab.itos)
config = gpt.OpenAIGPTConfig(vocab_size)

# model = bert.BertForPreTraining(config).cuda()
model = gpt.OpenAIGPTLMHeadModel(config).cuda()
model.reset = lambda: None

learn = LanguageLearner(data, model, bptt, clip=1)
learn.callbacks = []

learn.fit_one_cycle(4, 1e-3, div_factor=10, moms=(0.7,0.5))
learn.save('clc_first_run')