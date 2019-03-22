import music21
import torch

from fastai.distributed import *
from fastai.text.models.transformer import *

import numpy as np

import sys
sys.path.insert(0, '../src')
from fastai_data import *
from lmnp_transformer import *
from encode_data import VALTSEP, VALTBOS, PADDING_IDX, ENC_OFFSET

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../data/midi/v9/midi_encode/np/shortdur/')
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
parser.add_argument('--rand_transpose', action='store_true', help='Transpose data augmentation')
parser.add_argument('--rand_window', action='store_true', help='Random window size')
parser.add_argument('--gelu', action='store_true', help='Gelu activation')

args = parser.parse_args()

if args.local_rank != 0:
    f = open('/dev/null', 'w')
    sys.stdout = f
    
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

bs=args.batch_size
bptt=args.bptt
path = Path(args.path)
train_tfms = [partial(rand_transpose, enc_offset=ENC_OFFSET, rand_range=(0,12))] if args.rand_transpose else None
data = LMNPDataBunch.load(path, bs=bs, bptt=bptt, cache_name=args.cache, train_tfms=train_tfms)

VOCAB_SZ = create_vocab_sizes(path/'tmp/all')

N_COMPS = len(VOCAB_SZ)
N_EMBS = 128
EMB_IDXS = range(N_COMPS)
EMB_DIM = [N_EMBS]*len(EMB_IDXS)
EMB_MAP = list(zip(EMB_IDXS,VOCAB_SZ,EMB_DIM))

idx2embidx = { i:EMB_MAP[i] for i in range(N_COMPS) }
total_embs = sum([v[-1] for k,v in idx2embidx.items()])

config = tfmerXL_lm_config
config['emb_map'] = list(zip(EMB_IDXS,VOCAB_SZ,EMB_DIM))
config['idx_map'] = idx2embidx
config['loss_weights'] = [1,1] # note,duration
config['pad_idx'] = PADDING_IDX+ENC_OFFSET
config['bos_idx'] = VALTBOS+ENC_OFFSET
config['mask_type'] = MaskType.RandomWindow if args.rand_window else MaskType.Sequential
config['act'] = Activation.GeLU if args.gelu else Activation.ReLU

config['d_model'] = total_embs
config['mem_len'] = args.mem_len

config['resid_p'] = 0.1
config['attn_p'] = 0.1 # attention dropout
config['ff_p'] = 0.1
config['embed_p'] = 0.1 # embedding dropout
config['output_p'] = 0.1 # decoder dropout (before final linear layer)

full_clip = None if args.half else 0.25

learn = language_model_learner(data, config, clip=full_clip)

if args.load:
    load_path = Path(args.path)/args.load
    state = torch.load(load_path, map_location='cpu')
    get_model(learn.model).load_state_dict(state['model'], strict=False)
    learn.model.cuda()
if args.save:
    save_path = Path(args.path)/learn.model_dir/args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
if args.half: learn = learn.to_fp16(clip=0.25, dynamic=True)
learn = learn.to_distributed(args.local_rank, drop_last=True, shuffle=False)
if args.local_rank == 0: learn.callbacks.append(SaveModelCallback(learn, name=f'{args.save}_best'))
# learn.callbacks.append(EarlyStoppingCallback(learn))

learn.fit_one_cycle(args.epochs, args.lr, div_factor=25, moms=(0.7,0.5))

if args.local_rank == 0: learn.save(f'{args.save}')
