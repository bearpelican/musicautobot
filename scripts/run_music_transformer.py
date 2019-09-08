
import music21
import torch
import numpy as np
try: from apex.optimizers import FusedAdam
except: from torch.optim import Adam as FusedAdam

from fastai.distributed import *
from fastai.callbacks import SaveModelCallback
from fastai.text.models.transformer import *


import sys
sys.path.insert(0, '..')

from musicautobot.music_transformer import *


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../data/numpy/')
parser.add_argument('--data_file', type=str, default='musicitem_data_save.pkl')
parser.add_argument('--save', type=str, default='first_run')
parser.add_argument('--load', type=str, default=None)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--mem_len", type=int, default=512)
parser.add_argument("--bptt", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument('--half', action='store_true', help='Use half precision')
parser.add_argument('--lamb', action='store_true', help='Use lamb optimizer')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay for adam')
parser.add_argument('--epochs', type=int, default=5, help='num epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--div_factor', type=int, default=10, help='learning rate div factor')
parser.add_argument('--config', type=str, default='default_config', help='serve.py config name')
parser.add_argument('--no_transpose', action='store_true', help='No transpose data augmentation')
parser.add_argument('--parallel', action='store_true', help='Run in dataparallel')
parser.add_argument('--mask_steps', type=int, default=1, help='Attention mask - max number of random steps. Basically teacher forcing')

args = parser.parse_args()
is_distributed = num_distrib() > 0
if args.local_rank != 0:
    f = open('/dev/null', 'w')
    sys.stdout = f

if is_distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')


path = Path(args.path)

from musicautobot import config
config = getattr(config, args.config)()
config['encode_position'] = True
config['mask_steps'] = args.mask_steps

transpose_range = None if args.no_transpose else (0,12)
data = load_data(path, args.data_file, encode_position=config['encode_position'], dl_tfms=[batch_position_tfm],
                    bs=args.batch_size, bptt=args.bptt, transpose_range=transpose_range, num_workers=args.num_workers)

eps = 1e-2 if args.half else 1e-6
opt_func = partial(FusedAdam, betas=(0.9,0.99), eps=eps)
if args.lamb:
    from musicautobot.utils.lamb import Lamb
    opt_func = partial(Lamb, eps=eps)
    
load_path = path/args.load if args.load else None
learn = music_model_learner(data, config=config, drop_mult=1.5, opt_func=opt_func, pretrained_path=load_path)
if not args.half: learn.clip_grad(1.0)

if args.save:
    save_path = path/learn.model_dir/args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
if args.half: learn = learn.to_fp16(clip=1.0, dynamic=True, max_scale=2**18)
if is_distributed: learn = learn.to_distributed(args.local_rank, cache_dir=path/'dist_logs')
if args.parallel: learn = learn.to_parallel()
if args.local_rank == 0: learn.callbacks.append(SaveModelCallback(learn, name=f'{args.save}_best'))

learn.fit_one_cycle(args.epochs, args.lr, div_factor=args.div_factor, pct_start=0.2, final_div=200, wd=args.wd)

if args.local_rank == 0: learn.save(f'{args.save}', config=config)
