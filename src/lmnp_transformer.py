from enum import Enum

import torch.nn as nn
import torch

from fastai.text import *
from fastai.text.models.transformer import *
from fastai.callbacks.tracker import *

MaskType = Enum('MaskType', 'NoMask Sequential RandomWindow Bert')

class TransformerEmbed(nn.Module):
    def __init__(self, emb_map, idx_map, embed_p:float=0.1, pad_idx=None, **kwargs):
        super().__init__()
        # note, octave, duration, instrument
        self.idx_map = idx_map
        self.emb_map = emb_map
        embeddings = []
        for idx,in_d,out_d in emb_map:
            embeddings.append(nn.Embedding(in_d, out_d, padding_idx=pad_idx))
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

def rand_window_mask(x_len,m_len,device,p=0.2,is_eval=False):
    if is_eval or m_len == 0 or np.random.rand() >= p: 
        win_size,k = (1,1)
    else: win_size,k = (np.random.randint(0,3)+1,0)
        
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
        elif self.mask_type.value == MaskType.Sequential.value: 
            self.mask = torch.triu(x.new_ones(x_len, seq_len), diagonal=1+m_len).byte()[None,None]
        elif self.mask_type.value == MaskType.RandomWindow.value: 
            self.mask = rand_window_mask(x_len,m_len,x.device,is_eval=not self.training)
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
#     init = config.pop('init') if 'init' in config else None
    init = init_transformer # removing from config so we can pickle it
    
    embed = TransformerEmbed(**config)
    txl = LMNPTransformerXL(embed, **config)
    decoder = TransformerDec(embed, **config)
    model = SequentialRNN(txl, decoder)
    
    return model if init is None else model.apply(init)


def language_model_learner(data:DataBunch, config:dict=None, drop_mult:float=1., pretrained:bool=True,
                           **learn_kwargs) -> 'LanguageLearner':
    "Create a `Learner` with a language model from `data` and `arch`."
    model = get_language_model(config=config, drop_mult=drop_mult)
    
    acc = partial(lmnp_accuracy, pad_idx=config['pad_idx'])
    loss_func = LMNPLoss(**config)
    learn = LMNPLearner(data, model, split_func=tfmer_lm_split, 
                        loss_func=loss_func, metrics=[acc], bos_idx=config['bos_idx'],
                        **learn_kwargs)
    return learn

class LMNPLoss(nn.Module):
    "Same as `func`, but flattens input and target."
    def __init__(self, pad_idx=None, loss_weights=None, **kwargs):
        super().__init__()
        self.fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.loss_weights = loss_weights
        # not using func otherwise _loss_func_name2activ uses this attribute to get cross entropy loss

    def __repr__(self): return f"numpyenc loss of {self.fn}"

    def forward(self, inputs:Tensor, target:Tensor, **kwargs)->Rank0Tensor:
        losses = []
        for idx,input in enumerate(inputs):
#             if idx in CIDX_ALL: continue
            t = target[...,idx]
            input = input.view(-1,input.shape[-1])
            loss_weight = 1 if not self.loss_weights else self.loss_weights[idx]
            losses.append(self.fn(input, t.view(-1))*loss_weight)
        return sum(losses)

def lmnp_accuracy(inputs:Tensor, target:Tensor, pad_idx=None, **kwargs)->Rank0Tensor:
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    inputs = [i.argmax(dim=-1).unsqueeze(dim=-1) for i in inputs]
    input = torch.cat(inputs, dim=-1)
    target = target.view(input.shape)
    
    acc = (input==target).float().cpu().numpy()
    acc[target==pad_idx] = np.nan
    return torch.tensor(np.nanmean(acc), device=target.device)


# Predictions
from fastai import basic_train # for predictions

def predict_func(parts): return [F.softmax(p, dim=-1) for p in parts]
# Need to monkey patch pred_batch activation function for 2d array
loss_func_name = camel2snake(LMNPLoss.__name__)
basic_train.loss_func_name2activ[loss_func_name] = predict_func

class LMNPLearner(LanguageLearner):
    def __init__(self, *args, bos_idx=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bos_idx = bos_idx
    
    def predict(self, xb, n_words:int=1, temperatures=(1.2,0.7), min_ps=(1/128,0.0), min_length=50):
        "Return the `n_words` that come after `text`."
        ds = self.data.single_dl.dataset
        self.model.reset()
        self.model[0].mask = False
        if xb.shape[0] > 1: xb = xb[0][None]
        seed = xb.cpu().numpy()
        yb = torch.ones_like(xb)
        timesteps = []
        for _ in progress_bar(range(n_words), leave=True):
            bar = []
            outputs = self.pred_batch(batch=(xb,yb))
            for idx,item in enumerate(outputs): #progress_bar(range(n_words), leave=False):
                res = item[0][-1]
                min_p,temperature = min_ps[idx], temperatures[idx]
#                 if idx == 0: print('Items over p:', (res >= min_p).float().sum(), res.shape)
                if (res >= min_p).float().sum() == 0:
                    warn(f"There is no item with probability >= {min_p}, try a lower value.")
                else: res[res < min_p] = 0.

                if len(timesteps) < min_length: res[self.bos_idx] = 0.

                res.pow_(1 / temperature)
                idx = torch.multinomial(res, 1)
    #             val,idx = torch.topk(res, 1)
                bar.append(idx.squeeze().to(xb.device))
            bar = torch.stack(bar, dim=-1)
            if self.bos_idx is not None and (bar==self.bos_idx).any(): 
                print('Predicted BOS token. Returning prediction...')
                break
            timesteps.append(bar.cpu().numpy())
            xb = bar.clone().detach()[None,None] # don't use timesteps. use it's own memory instead

        self.model[0].mask = True
        return timesteps, seed.squeeze()
