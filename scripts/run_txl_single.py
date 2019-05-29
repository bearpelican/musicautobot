
import music21
import torch

from fastai.distributed import *
from fastai.text.models.transformer import *

import numpy as np

import sys
sys.path.insert(0, '..')
from src.fastai_data import *
from src.lmnp_transformer import *
from src.encode_data import VALTSEP, VALTBOS, PADDING_IDX, ENC_OFFSET
from src.serve import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../data/midi/v15/midi_encode/')
parser.add_argument('--cache', type=str, default='tmp/dmp')
parser.add_argument('--save', type=str, default='first_run')
parser.add_argument('--load', type=str, default=None)
parser.add_argument("--local_rank", type=int)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--mem_len", type=int, default=512)
parser.add_argument("--bptt", type=int, default=512)
parser.add_argument('--half', action='store_true', help='Use half precision')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay for adam')
parser.add_argument('--epochs', type=int, default=5, help='num epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--div_factor', type=int, default=25, help='learning rate div factor')
parser.add_argument('--save_every', action='store_true', help='Save every epoch')
parser.add_argument('--config', type=str, help='serve.py config name')

args = parser.parse_args()

if args.local_rank != 0:
    f = open('/dev/null', 'w')
    sys.stdout = f
    
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')


path = Path(args.path)

from src import serve
vocab_path = path/'tmp/all/'
if args.config is None: config = v15_config(vocab_path)
else:
    config = getattr(serve, args.config)(vocab_path)

config['bptt'] = args.bptt
config['bs'] = args.batch_size
data = load_music_data(path=path, cache_name=args.cache, vocab=vocab, y_offset=1, **config)

full_clip = None if args.half else 0.5

learn = music_model_learner(data, config, clip=full_clip)

if args.load:
    load_path = Path(args.path)/args.load
    state = torch.load(load_path, map_location='cpu')
    get_model(learn.model).load_state_dict(state['model'], strict=False)
    learn.model.cuda()
if args.save:
    save_path = Path(args.path)/learn.model_dir/args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
if args.half: learn = learn.to_fp16(clip=0.5, dynamic=True)
learn = learn.to_distributed(args.local_rank, cache_dir=args.cache+'/dist_logs')
if args.local_rank == 0: learn.callbacks.append(SaveModelCallback(learn, name=f'{args.save}_best'))
if args.local_rank == 0 and args.save_every: learn.callbacks.append(SaveModelCallback(learn, name=f'{args.save}_epoch', every='epoch'))
# learn.callbacks.append(EarlyStoppingCallback(learn))

learn.fit_one_cycle(args.epochs, args.lr, div_factor=args.div_factor, pct_start=0.1, final_div=50, wd=1e-4)

if args.local_rank == 0: learn.save(f'{args.save}')
