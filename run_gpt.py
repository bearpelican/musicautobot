from fastai.text import *
from fastai.distributed import *
import gpt


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='data/midi_transcribe_v1_simple/')
parser.add_argument('--cache', type=str, default='tmp_clc')
parser.add_argument("--local_rank", type=int)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--bptt", type=int, default=500)
parser.add_argument('--half', action='store_true', help='Use half precision')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay for adam')
parser.add_argument('--epochs', type=int, default=5, help='num epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
args = parser.parse_args()


torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

bs=args.batch_size
bptt=args.bptt
path = Path(args.path)
data = TextLMDataBunch.load(path, cache_name=args.cache, bs=bs, bptt=bptt)

vocab = data.train_ds.vocab
vocab_size = len(vocab.itos)
config = gpt.OpenAIGPTConfig(vocab_size)

# model = bert.BertForPreTraining(config).cuda()
model = gpt.OpenAIGPTLMHeadModel(config).cuda()
model.reset = lambda: None

learn = LanguageLearner(data, model, bptt, clip=1).distributed(args.local_rank)
learn.callbacks = []

learn.fit_one_cycle(args.epochs, args.lr, div_factor=10, moms=(0.7,0.5))
learn.save(f'{args.cache}_first_run')