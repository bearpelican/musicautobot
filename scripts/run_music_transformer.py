
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
parser.add_argument('--half', action='store_true', help='Use half precision')
parser.add_argument('--lamb', action='store_true', help='Use lamb optimizer')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay for adam')
parser.add_argument('--epochs', type=int, default=5, help='num epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--div_factor', type=int, default=10, help='learning rate div factor')
parser.add_argument('--save_every', action='store_true', help='Save every epoch')
parser.add_argument('--config', type=str, help='serve.py config name')
parser.add_argument('--no_transpose', action='store_true', help='No transpose data augmentation')
parser.add_argument('--parallel', action='store_true', help='Run in dataparallel')

args = parser.parse_args()
is_distributed = num_distrib() > 0
if args.local_rank != 0:
    f = open('/dev/null', 'w')
    sys.stdout = f

if is_distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')


path = Path(args.path)

from src import config
config = getattr(config, args.config)()

if args.no_transpose: config['transpose_range'] = None
data = load_data(path, args.data_file, 
                    bs=args.batch_size, bptt=args.bptt, transpose_range=config['transpose_range'],
                    dl_tfms=mask_lm_tfm, preloader_cls=position_preloader)

opt_func = partial(FusedAdam, betas=(0.9,0.99), eps=1e-4)
if args.lamb:
    from src.lamb import Lamb
    opt_func = partial(Lamb, eps=1e-4)
    
learn = music_model_learner(data, config, drop_mult=1.5, opt_func=opt_func)
if not args.half: learn.clip_grad(0.5)

if args.load:
    state = torch.load(path/args.load, map_location='cpu')
    get_model(learn.model).load_state_dict(state['model'], strict=False)
    learn.model.cuda()
if args.save:
    save_path = path/learn.model_dir/args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
if args.half: learn = learn.to_fp16(clip=0.5, dynamic=True, max_scale=2**18)
if is_distributed: learn = learn.to_distributed(args.local_rank, cache_dir=args.cache+'/dist_logs')
if args.parallel: learn = learn.to_parallel()
if args.local_rank == 0: learn.callbacks.append(SaveModelCallback(learn, name=f'{args.save}_best'))
if args.local_rank == 0 and args.save_every: learn.callbacks.append(SaveModelCallback(learn, name=f'{args.save}_epoch', every='epoch'))
# learn.callbacks.append(EarlyStoppingCallback(learn))

if not args.lamb: learn.fit_one_cycle(2, args.lr/2, div_factor=50, pct_start=0.9) # no need for warmup with lamb
learn.fit_one_cycle(args.epochs, args.lr, div_factor=args.div_factor, pct_start=0.2, final_div=200, wd=args.wd)

if args.local_rank == 0: learn.save(f'{args.save}')
