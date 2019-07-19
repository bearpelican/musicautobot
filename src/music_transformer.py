from enum import Enum

import torch.nn as nn
import torch

from fastai.text import *
from fastai.text.models.transformer import *
from fastai.text.models.transformer import init_transformer
from fastai.text.learner import language_model_learner, get_language_model, _model_meta
from fastai.callbacks.tracker import *

from .encode_data import SAMPLE_FREQ

def window_mask(x_len, device, m_len=0, size=(1,1)):
    win_size,k = size
    mem_mask = torch.zeros((x_len,m_len), device=device)
    tri_mask = torch.triu(torch.ones((x_len//win_size+1,x_len//win_size+1), device=device),diagonal=k)
    window_mask = tri_mask.repeat_interleave(win_size,dim=0).repeat_interleave(win_size,dim=1)[:x_len,:x_len]
    window_mask[...,0] = 0 # Always allowing first index to see. Otherwise you'll get NaN loss
    mask = torch.cat((mem_mask, window_mask), dim=1).byte()[None,None]
#     if m_len == 0: mask[...,0] = 0 # attention needs to see at least first column otherwise NaN
    return mask
    
def rand_window_mask(x_len,m_len,device,max_size=3,p=0.2,is_eval=False):
    if is_eval or np.random.rand() >= p: 
        win_size,k = (1,1)
    else: win_size,k = (np.random.randint(0,max_size)+1,0)
    return window_mask(x_len, device, m_len, size=(win_size,k))

def lm_mask(x_len, device):
    return torch.triu(torch.ones((x_len, x_len), device=device), diagonal=1)[None,None].byte()

# import inspect
# argspec = inspect.getfullargspec(TransformerXL)
# config_params = { k:config[k] for k in argspec.args if k in config }
def music_model_learner(data:DataBunch, config:dict=None, drop_mult:float=1., pretrained:bool=False,
                        pretrained_fnames:OptStrTuple=None, **learn_kwargs) -> 'LanguageLearner':
    "Create a `Learner` with a language model from `data` and `arch`."
    _model_meta[MusicTransformerXL] = _model_meta[TransformerXL]
    model = get_language_model(MusicTransformerXL, len(data.vocab.itos), config=config, drop_mult=drop_mult)
    
    meta = _model_meta[TransformerXL]
    learn = MusicLearner(data, model, split_func=meta['split_lm'], **learn_kwargs)
    return learn

class MusicTransformerXL(TransformerXL):
    def __init__(self, *args, **kwargs):
        import inspect
        sig = inspect.signature(TransformerXL)
        arg_params = { k:kwargs[k] for k in sig.parameters if k in kwargs }
        super().__init__(*args, **arg_params)
        
    def forward(self, x):
        #The hidden state has to be initiliazed in the forward pass for nn.DataParallel
        if self.mem_len > 0 and not self.init: 
            self.reset()
            self.init = True
        bs,x_len = x.size()
        inp = self.drop_emb(self.encoder(x)) #.mul_(self.d_model ** 0.5)
        m_len = self.hidden[0].size(1) if hasattr(self, 'hidden') and len(self.hidden[0].size()) > 1 else 0
        seq_len = m_len + x_len
        
        mask = rand_window_mask(x_len, m_len, inp.device, is_eval=not self.training) if self.mask else None
        if m_len == 0: mask[...,0,0] = 0
        #[None,:,:None] for einsum implementation of attention
        hids = []
        pos = torch.arange(seq_len-1, -1, -1, device=inp.device, dtype=inp.dtype)
        pos_enc = self.pos_enc(pos)
        hids.append(inp)
        for i, layer in enumerate(self.layers):
            mem = self.hidden[i] if self.mem_len > 0 else None
            inp = layer(inp, r=pos_enc, u=self.u, v=self.v, mask=mask, mem=mem)
            hids.append(inp)
        core_out = inp[:,-x_len:]
        if self.mem_len > 0 : self._update_mems(hids)
        return (self.hidden if self.mem_len > 0 else [core_out]),[core_out]

    
# Predictions
from fastai import basic_train # for predictions
class MusicLearner(LanguageLearner):
    def beam_search(self, xb:Tensor, n_words:int, top_k:int=10, beam_sz:int=10, temperature:float=1.,
                    ):
        "Return the `n_words` that come after `text` using beam search."
        self.model.reset()
        self.model.eval()
        xb_length = xb.shape[-1]
        if xb.shape[0] > 1: xb = xb[0][None]
        yb = torch.ones_like(xb)

        nodes = None
        xb = xb.repeat(top_k, 1)
        nodes = xb.clone()
        scores = xb.new_zeros(1).float()
        with torch.no_grad():
            for k in progress_bar(range(n_words), leave=False):
                out = F.log_softmax(self.model(xb)[0][:,-1], dim=-1)
    #             if no_unk: out[:,self.data.vocab.stoi[UNK]] = -float('Inf')
                values, indices = out.topk(top_k, dim=-1)
                scores = (-values + scores[:,None]).view(-1)
                indices_idx = torch.arange(0,nodes.size(0))[:,None].expand(nodes.size(0), top_k).contiguous().view(-1)
                sort_idx = scores.argsort()[:beam_sz]
                scores = scores[sort_idx]
                nodes = torch.cat([nodes[:,None].expand(nodes.size(0),top_k,nodes.size(1)),
                                indices[:,:,None].expand(nodes.size(0),top_k,1),], dim=2)
                nodes = nodes.view(-1, nodes.size(2))[sort_idx]
                self.model[0].select_hidden(indices_idx[sort_idx])
                xb = nodes[:,-1][:,None]
        if temperature != 1.: scores.div_(temperature)
        node_idx = torch.multinomial(torch.exp(-scores), 1).item()
        return [i.item() for i in nodes[node_idx][xb_length:] ]

    def predict(self, xb:Tensor, n_words:int=128,
                     temperatures:float=(1.0,1.0), min_bars=4,
                     top_k=40, top_p=0.9):
        "Return the `n_words` that come after `text`."
        self.model.reset()
        if xb.shape[0] > 1: xb = xb[0][None]
        seed = xb.cpu().numpy().squeeze()
        yb = torch.ones_like(xb)
        new_idx = []

        sep_count = 0

        bar_len = SAMPLE_FREQ * 4 # assuming 4/4 time
        vocab = self.data.vocab

        with torch.no_grad():
            for i in progress_bar(range(n_words), leave=True):

                res = self.pred_batch(batch=(xb,yb))[0][-1]

                # bar = 16 beats
                if (sep_count // 16) <= min_bars: res[vocab.bos_idx] = 0.

                # Use first temperatures value if last prediction was duration
                temperature = temperatures[0] if (len(new_idx)==0 or self.data.vocab.is_duration(new_idx[-1])) else temperatures[1]
                if temperature != 1.: res.pow_(1 / temperature)

                res = top_k_top_p(res, top_k=top_k, top_p=top_p, filter_value=0)
                idx = torch.multinomial(res, 1).item()

                if new_idx and new_idx[-1]==vocab.sep_idx: 
                    duration = idx - vocab.dur_range[0]
                    sep_count += duration
                    # print('Bars', duration, sep_count // 16)

                if idx==vocab.bos_idx: 
                    print('Predicted BOS token. Returning prediction...')
                    break


                new_idx.append(idx)
                xb = xb.new_tensor([idx])[None]
        return np.array(new_idx), seed
    
