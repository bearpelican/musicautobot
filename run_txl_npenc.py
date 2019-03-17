from fastai.text import *
from fastai.callbacks.tracker import *
from fastai.distributed import *
from fastai_data import *

import music21
from enum import Enum
import torch
from fastai.text.models.awd_lstm import *
from fastai.text.models.transformer import *
from fastai_data import *
import numpy as np
import torch.nn as nn

from encode_data import *
import fastai_data
fastai_data.Y_OFFSET=1
# fastai_data.VAL_OFFSET=1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='data/midi/v8/midi_encode/np/shortdur_2c/')
parser.add_argument('--cache', type=str, default='tmp/dmp')
parser.add_argument('--save', type=str, default='first_run')
parser.add_argument('--load', type=str, default=None)
parser.add_argument("--local_rank", type=int)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--mem_len", type=int, default=512)
parser.add_argument("--bptt", type=int, default=256)
parser.add_argument('--half', action='store_true', help='Use half precision')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay for adam')
parser.add_argument('--epochs', type=int, default=5, help='num epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
args = parser.parse_args()

if args.local_rank != 0:
    f = open('/dev/null', 'w')
    sys.stdout = f
    
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

bs=args.batch_size
bptt=args.bptt
path = Path(args.path)
data = LMNPDataBunch.load(path, bs=bs, bptt=bptt, cache_name=args.cache)

MaskType = Enum('MaskType', 'NoMask Sequential RandomWindow Bert')


max_vocab_file = path/'tmp/all/max_vocab.npy'
if max_vocab_file.exists():
    VOCAB_SZ = np.load(max_vocab_file).tolist()
else:
    train_ids_file = path/'tmp/all/train_ids.npy'
    all_ids = np.load(train_ids_file)
    id_cat = np.concatenate(all_ids); id_cat.shape
    ax = tuple(range(len(id_cat.shape)-1))
    max_vocab = id_cat.max(axis=ax)+1
    np.save(max_vocab_file, max_vocab)
    VOCAB_SZ = max_vocab.tolist()

PAD_IDX=PADDING_IDX+ENC_OFFSET
N_BAR = 1
N_COMPS = len(VOCAB_SZ)
N_EMBS = 128
EMB_IDXS = range(N_COMPS)
EMB_DIM = [N_EMBS]*len(EMB_IDXS)
EMB_MAP = list(zip(EMB_IDXS,VOCAB_SZ,EMB_DIM))
LOSS_WEIGHTS = [1,1,1]
idx2embidx = { i:EMB_MAP[i] for i in range(N_COMPS) }


config = tfmerXL_lm_config
config['emb_map'] = EMB_MAP
config['idx_map'] = idx2embidx
# config['mask_type'] = MaskType.RandomWindow
config['mask_type'] = MaskType.Sequential

total_embs = sum([v[-1] for k,v in idx2embidx.items()])
config['d_model'] = total_embs * N_BAR
config['mem_len'] = args.mem_len
# config['d_inner'] = 1024 * N_BAR
# config['d_inner'] = config['d_model'] * 4

config['resid_p'] = 0.2
config['attn_p'] = 0.2
config['ff_p'] = 0.2
config['embed_p'] = 0.2
config['output_p'] = 0.2


full_clip = None if args.half else 0.25


class TransformerEmbed(nn.Module):
    def __init__(self, emb_map, idx_map, embed_p:float=0.1, **kwargs):
        super().__init__()
        # note, octave, duration, instrument
        self.idx_map = idx_map
        self.emb_map = emb_map
        embeddings = []
        for idx,in_d,out_d in emb_map:
            embeddings.append(nn.Embedding(in_d, out_d, padding_idx=PAD_IDX))
        self.embeddings = nn.ModuleList(embeddings)
        self.drop_emb = nn.Dropout(embed_p)
        
    def forward(self, x):
        # batch x bptt x (n,o,d,i)
        embs = []
        for i in range(x.shape[-1]):
            emb_idx = self.idx_map[i][0]
            embx = self.embeddings[emb_idx](x[...,i])
            embs.append(embx)
        emb = torch.stack(embs, dim=-2) # barlen x comp x emb
#         emb = emb.permute(0,1,4,2,3) # for conv - emb x barlen x comp
        return self.drop_emb(emb)

class TXLLinearDecoder(nn.Module):
    "To go on top of a RNNCore module and create a Language Model."
    initrange=0.1

    def __init__(self, tie_encoder:nn.Module=None, bias:bool=True, input_dim=None):
        super().__init__()
        n_hid,n_out = tie_encoder.embedding_dim, tie_encoder.num_embeddings
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        if bias: self.decoder.bias.data.zero_()
        if tie_encoder: self.decoder.weight = tie_encoder.weight
            
        if input_dim is not None and input_dim != n_hid:
            self.decoder = nn.Sequential(nn.Linear(input_dim, n_hid), self.decoder)

    def forward(self, input):
        return self.decoder(input)

class TransformerDec(nn.Module):
    def __init__(self, txl_emb, idx_map, output_p=0.0, out_bias=True, d_model=None, **kwargs):
        super().__init__()
        self.output_dp = RNNDropout(output_p)
        
        decoders = []
        for k,v in idx_map.items():
            emb = txl_emb.embeddings[v[0]]
            decoder = TXLLinearDecoder(tie_encoder=emb, bias=out_bias, input_dim=d_model)
            decoders.append(decoder)
            
        self.decoders = nn.ModuleList(decoders)
        
    def forward(self, input):
        raw_outputs, outputs = input
        output = self.output_dp(outputs[-1])
        res = []
        for idx,dec in enumerate(self.decoders):
            res.append(dec(output))
        return res, raw_outputs, outputs

def rand_window_mask(x_len,m_len,device):
    win_size,k = (np.random.randint(0,16)+1,0) if m_len > 0 else (1,1)
    mem_mask = np.zeros((x_len,m_len))
    tri_mask = np.triu(np.ones((x_len//win_size+1,x_len//win_size+1)),k=k)
    window_mask = tri_mask.repeat(win_size,axis=0).repeat(win_size,axis=1)[:x_len,:x_len]
    np_mask = np.concatenate((mem_mask, window_mask), axis=1)
    mask = torch.tensor(np_mask, device=device).byte()[None,None]; mask
    return mask


class LMNPTransformerXL(nn.Module):
    "TransformerXL model: https://arxiv.org/abs/1901.02860."
    def __init__(self, encoder, ctx_len:int, n_layers:int, n_heads:int, d_model:int, d_head:int, d_inner:int, 
                 resid_p:float=0., attn_p:float=0., ff_p:float=0., embed_p:float=0., bias:bool=False, scale:bool=True,
                 act:Activation=Activation.ReLU, double_drop:bool=True, attn_cls:Callable=MultiHeadRelativeAttention,
                 learned_pos_enc:bool=False, mask_type:MaskType=MaskType.Sequential, mem_len:int=0, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.pos_enc = nn.Embedding(ctx_len, d_model) if learned_pos_enc else PositionalEncoding(d_model)
        self.u = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.v = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.mem_len,self.n_layers,self.d_model,self.mask_type = mem_len,n_layers,d_model,mask_type
        self.init = False
        self.layers = nn.ModuleList([DecoderLayer(n_heads, d_model, d_head, d_inner, resid_p=resid_p, attn_p=attn_p,
                      ff_p=ff_p, bias=bias, scale=scale, act=act, double_drop=double_drop, 
                      attn_cls=attn_cls) for k in range(n_layers)])
    
    def reset(self):
        "Reset the internal memory."
        self.hidden = [next(self.parameters()).data.new(0) for i in range(self.n_layers+1)]

    def _update_mems(self, hids):
        if not getattr(self, 'hidden', False): return None
        assert len(hids) == len(self.hidden), 'len(hids) != len(self.hidden)'
        with torch.no_grad():
            for i in range(len(hids)):
                cat = torch.cat([self.hidden[i], hids[i]], dim=1)
                self.hidden[i] = cat[:,-self.mem_len:].detach()
    
    def select_hidden(self, idxs): self.hidden = [h[idxs] for h in self.hidden]
    
    def forward(self, x):
        #The hidden state has to be initiliazed in the forward pass for nn.DataParallel
        if self.mem_len > 0 and not self.init: 
            self.reset()
            self.init = True
        inp = self.encoder(x)
        if self.mask_type.value == MaskType.Bert.value:
            inp = inp.reshape(inp.shape[0], -1, inp.shape[-1]) # bs,bptt,comp,emb -> bs,bptt*comp,emb
        else:
            inp = inp.reshape(*inp.shape[:2], -1) # bs,bptt,comp,emb -> bs,bptt,comp,emb
        bs,x_len = inp.shape[:2]
        m_len = self.hidden[0].size(1) if hasattr(self, 'hidden') and len(self.hidden[0].size()) > 1 else 0
        seq_len = m_len + x_len
        
        if self.mask_type.value == MaskType.NoMask.value: 
            self.mask = None
        elif self.mask_type.value == MaskType.RandomWindow.value: 
            self.mask = rand_window_mask(x_len,m_len,x.device)
        elif self.mask_type.value == MaskType.Sequential.value: 
            self.mask = torch.triu(x.new_ones(x_len, seq_len), diagonal=1+m_len).byte()[None,None]
        else: 
            raise ValueError('Unhandled mask type:', self.mask_type)
        #[None,:,:None] for einsum implementation of attention
        hids = []
        pos = torch.arange(seq_len-1, -1, -1, device=inp.device, dtype=inp.dtype)
        pos_enc = self.pos_enc(pos)
        hids.append(inp)
        for i, layer in enumerate(self.layers):
            mem = self.hidden[i] if self.mem_len > 0 else None
            inp = layer(inp, r=pos_enc, u=self.u, v=self.v, mask=self.mask, mem=mem)
            hids.append(inp)
        core_out = inp[:,-x_len:]
        if self.mem_len > 0 : self._update_mems(hids)
        return (self.hidden if self.mem_len > 0 else [core_out]),[core_out]

def get_language_model(config:dict=None, drop_mult:float=1.):
    "Create a language model from `arch` and its `config`, maybe `pretrained`."
    for k in config.keys(): 
        if k.endswith('_p'): config[k] *= drop_mult
    init = config.pop('init') if 'init' in config else None
    
    embed = TransformerEmbed(**config)
    txl = LMNPTransformerXL(embed, **config)
    decoder = TransformerDec(embed, **config)
    model = SequentialRNN(txl, decoder)
    
    return model if init is None else model.apply(init)


def language_model_learner(data:DataBunch, config:dict=None, drop_mult:float=1., pretrained:bool=True,
                           **learn_kwargs) -> 'LanguageLearner':
    "Create a `Learner` with a language model from `data` and `arch`."
    model = get_language_model(config=config, drop_mult=drop_mult)
    learn = LanguageLearner(data, model, split_func=tfmer_lm_split, **learn_kwargs)
    return learn

class LMNPLoss(nn.Module):
    "Same as `func`, but flattens input and target."
    def __init__(self):
        super().__init__()
        self.fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        # not using func otherwise _loss_func_name2activ uses this attribute to get cross entropy loss

    def __repr__(self): return f"numpyenc loss of {self.fn}"

    def forward(self, inputs:Tensor, target:Tensor, **kwargs)->Rank0Tensor:
        losses = []
        for idx,input in enumerate(inputs):
#             if idx in CIDX_ALL: continue
            t = target[...,idx]
            input = input.view(-1,input.shape[-1])
            losses.append(self.fn(input, t.view(-1))*LOSS_WEIGHTS[idx])
        return sum(losses)

def lmnp_accuracy(inputs:Tensor, target:Tensor)->Rank0Tensor:
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    inputs = [i.argmax(dim=-1).unsqueeze(dim=-1) for i in inputs]
    input = torch.cat(inputs, dim=-1)
    target = target.view(input.shape)
    
    acc = (input==target).float().cpu().numpy()
    acc[target==PAD_IDX] = np.nan
    return torch.tensor(np.nanmean(acc), device=target.device)

learn = language_model_learner(data, config, clip=full_clip, loss_func=LMNPLoss(), metrics=[lmnp_accuracy])

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
learn.callbacks.append(EarlyStoppingCallback(learn))

learn.fit_one_cycle(args.epochs, args.lr, div_factor=25, moms=(0.7,0.5))

if args.local_rank == 0: learn.save(f'{args.save}')