import music21
import torch

from fastai.distributed import *
from fastai.text.models.transformer import *
from apex.optimizers import FusedAdam

import numpy as np

import sys
sys.path.insert(0, '..')
from src.fastai_data import *
from src.music_transformer import *
from src.serve import *
from src.msklm import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../data/midi/v18/sf4/')
parser.add_argument('--cache', type=str, default='tmp/all')
parser.add_argument('--save', type=str, default='first_run')
parser.add_argument('--load', type=str, default=None)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--bptt", type=int, default=1024)
parser.add_argument('--half', action='store_true', help='Use half precision')
parser.add_argument('--lamb', action='store_true', help='Use lamb optimizer')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay for adam')
parser.add_argument('--epochs', type=int, default=5, help='num epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--div_factor', type=int, default=10, help='learning rate div factor')
parser.add_argument('--save_every', action='store_true', help='Save every epoch')
parser.add_argument('--config', type=str, default='unilm_config', help='serve.py config name')
parser.add_argument('--no_transpose', action='store_true', help='No transpose data augmentation')
parser.add_argument('--data_parallel', action='store_true', help='DataParallel instead of DDP')
parser.add_argument('--s2s_mask_window', type=int, default=1,
                    help='Starting mask window size for sequence2sequence task. Basically teacher forcing')

args = parser.parse_args()
args.path = Path(args.path)

is_distributed = num_distrib() > 0

if args.local_rank != 0:
    f = open('/dev/null', 'w')
    sys.stdout = f

if is_distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')


path = Path(args.path)

from src import serve
config = getattr(serve, args.config)(vocab)

config['bptt'] = args.bptt
config['bs'] = args.batch_size

if args.no_transpose: config['transpose_range'] = (0, 1)

datasets = []

dl_tfms = [partial(mask_or_lm_tfm, p_mask=0.3)]
msklm_data = MusicDataBunch.load(args.path/'piano_duet', cache_name=args.cache, vocab=vocab, 
                              y_offset=1, dl_tfms=dl_tfms, **config)
datasets.append(msklm_data)

s2s_config = config.copy()
s2s_config['bs'] = s2s_config['bs'] // 4
s2s_config['bptt'] *= 2

m2c_dl_tfms = [s2s_tfm]
m2c_data = MusicDataBunch.load(args.path/'s2s_encode', cache_name=args.cache, 
                               preloader_cls=S2SPreloader, dl_tfms=m2c_dl_tfms,
                               shuffle_dl=True, **config)

c2m_dl_tfms = [partial(s2s_tfm, mlm_type=MLMType.C2M)]
c2m_data = MusicDataBunch.load(args.path/'s2s_encode', cache_name=args.cache, 
                               preloader_cls=S2SPreloader, dl_tfms=c2m_dl_tfms,
                               shuffle_dl=True, **config)

datasets.append(m2c_data)
datasets.append(c2m_data)

combined_data = CombinedData(datasets)

full_clip = None if args.half else 0.5

# Load Optimizer
eps = 1e-3 if args.half else 1e-6
opt_func = partial(FusedAdam, betas=(0.9,0.99), eps=eps)
if args.lamb:
    from src.lamb import Lamb
    opt_func = partial(Lamb, eps=eps)
    
# Load Learner
learn = mlm_model_learner(combined_data, config.copy(), 
                           loss_func=MLMLoss(),
                           clip=full_clip, opt_func=opt_func)

# Load custom data trainer - overwrite RNNTrainer
learn.metrics = [acc_ignore_pad, mask_acc, lm_acc, c2m_acc, m2c_acc]

from fastai.callbacks.rnn import RNNTrainer
learn.callbacks = [c for c in learn.callbacks if not isinstance(c, RNNTrainer)]
learn.callbacks.append(MLMTrainer(learn))
#learn.callbacks.append(MLMTrainer(learn, s2s_starting_mask_window=args.s2s_mask_window))

if args.load:
    load_path = Path(args.path)/args.load
    state = torch.load(load_path, map_location='cpu')
    get_model(learn.model).load_state_dict(state['model'], strict=False)
    learn.model.cuda()
if args.save:
    save_path = Path(args.path)/'s2s_encode'/learn.model_dir/args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path = Path(args.path)/'piano_duet'/learn.model_dir/args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
if args.half: learn = learn.to_fp16(clip=0.5, dynamic=True, max_scale=2**18)
if is_distributed: learn = learn.to_distributed(args.local_rank, cache_dir=args.cache+'/dist_logs')
if args.data_parallel: learn = learn.to_parallel()
if args.local_rank == 0: learn.callbacks.append(SaveModelCallback(learn, name=f'{args.save}_best'))

# if not args.lamb: learn.fit_one_cycle(2, args.lr/2, div_factor=50, pct_start=0.9) # no need for warmup with lamb
learn.fit_one_cycle(args.epochs, args.lr, div_factor=args.div_factor, pct_start=.3, final_div=50, wd=args.wd)

if args.local_rank == 0: learn.save(f'{args.save}')
