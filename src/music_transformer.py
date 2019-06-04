from enum import Enum

import torch.nn as nn
import torch

from fastai.text import *
from fastai.text.models.transformer import *
from fastai.text.models.transformer import init_transformer
from fastai.text.learner import language_model_learner, get_language_model, _model_meta
from fastai.callbacks.tracker import *

def window_mask(x_len, device, m_len=0, size=(1,1)):
    win_size,k = size
    mem_mask = np.zeros((x_len,m_len))
    tri_mask = np.triu(np.ones((x_len//win_size+1,x_len//win_size+1)),k=k)
    window_mask = tri_mask.repeat(win_size,axis=0).repeat(win_size,axis=1)[:x_len,:x_len]
    np_mask = np.concatenate((mem_mask, window_mask), axis=1)
    mask = torch.tensor(np_mask, device=device).byte()[None,None]
    return mask
    
def rand_window_mask(x_len,m_len,device,max_size=3,p=0.2,is_eval=False):
    if is_eval or m_len == 0 or np.random.rand() >= p: 
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
    model = get_language_model(MusicTransformerXL, config['vocab_size'], config=config, drop_mult=drop_mult)
    
    meta = _model_meta[TransformerXL]
    learn = MusicLearner(data, model, split_func=meta['split_lm'], 
                         bos_idx=config['bos_idx'], sep_idx=config['sep_idx'],
                        **learn_kwargs)
    return learn

class MusicTransformerXL(TransformerXL):
    def __init__(self, *args, **kwargs):
        import inspect
        argspec = inspect.getfullargspec(TransformerXL)
        arg_params = { k:kwargs[k] for k in argspec.args if k in kwargs }
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
        
        # mask = torch.triu(x.new_ones(x_len, seq_len), diagonal=m_len).byte()[None,None] if self.mask else None # bert
        # mask = torch.triu(x.new_ones(x_len, seq_len), diagonal=1+m_len).byte()[None,None] if self.mask else None # lm
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
    def __init__(self, *args, bos_idx=None, sep_idx=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bos_idx = bos_idx
        self.sep_idx = sep_idx
        print('Sep_idx:', self.sep_idx)

    def beam_search(self, xb:Tensor, n_words:int, top_k:int=10, beam_sz:int=10, temperature:float=1.,
                    ):
        "Return the `n_words` that come after `text` using beam search."
        ds = self.data.single_dl.dataset
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
            temperatures:float=(1.0,1.0), min_p:float=None, min_bars=4):
        "Return the `n_words` that come after `text`."
        ds = self.data.single_dl.dataset
        self.model.reset()
        self.model.mask = False
        if xb.shape[0] > 1: xb = xb[0][None]
        seed = xb.cpu().numpy().squeeze()
        yb = torch.ones_like(xb)
        new_idx = []

        running_ps = 1.0
        timesteps = []
        sep_count = 0

        for i in progress_bar(range(n_words), leave=True):

            running_ps = (n_words * 2 - i) / (n_words * 2)

            res = self.pred_batch(batch=(xb,yb))[0][-1]
            #if len(new_idx) == 0: self.model[0].select_hidden([0])
            if min_p is not None: 
                if (res >= min_p).float().sum() == 0:
                    warn(f"There is no item with probability >= {min_p}, try a lower value.")
                else: res[res < min_p] = 0.

            # bar = 16 beats
            if (sep_count // 16) <= min_bars: res[self.bos_idx] = 0.

            # Use first temperatures value if last prediction was duration
            temperature = temperatures[0] if (len(new_idx)==0 or self.data.vocab.is_duration(new_idx[-1])) else temperatures[1]
            if temperature != 1.: res.pow_(1 / (temperature * running_ps))

            idx = torch.multinomial(res, 1).item()


            if new_idx and new_idx[-1]==self.sep_idx: 
                duration = (idx - 3 - 130) + 1
                sep_count += duration
    #                 print('Bars', duration, sep_count // 16)

            if idx==self.bos_idx: 
                print('Predicted BOS token. Returning prediction...')
                break


            new_idx.append(idx)
            xb = xb.new_tensor([idx])[None]
        return np.array(new_idx), seed
