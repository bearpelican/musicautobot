from .fastai_data import *
from .music_transformer import MusicLearner, lm_mask
from fastai.basics import *
from fastai.text.models.transformer import _line_shift, init_transformer
from fastai.text.models.awd_lstm import *
from fastai.text.models.transformer import *

# DATALOADING AND TRANSFORMATIONS

TaskType = Enum('TaskType', 'MaskOnly, NextSent, Translate, NextWord')

# BERT Transform

def next_sentence_ranges(x, y, max_cls=4):
    bs,bptt = x.shape
    s = min(random.randint(1, max_cls), bs-2)
    
    min_seq_len = bptt // s

    bs_shift = [0]+(np.random.choice(bs-1, s, replace=False)+1).tolist()    
    row_shift = [int(min_seq_len + random.randint(-min_seq_len, min_seq_len)//s) for i in range(s)]
    
    accum = 0
    ranges = []
    for i in range(s):
        end = accum + row_shift[i] if i < (s-1) else bptt
        ranges.append((i, bs_shift[i], accum, end))
        accum = end
    return ranges

def next_sentence_tfm(b, max_cls=4):
    x, y = b
    x_new = x.clone()
    y_new = y.clone()
    z = torch.zeros_like(x)
    ranges = next_sentence_ranges(x, y, max_cls)
    for i,shift,s,e in ranges:
        if i == 0: continue
        x_new[:, s:e] = torch.roll(x, shifts=shift, dims=0)[:, s:e]
        y_new[:, s:e] = torch.roll(y, shifts=shift, dims=0)[:, s:e]
        z[:, s:e] = i
    return (x_new, torch.tensor([TaskType.NextSent.value])), (y_new, z)

def mask_tfm(b, word_range=vocab.npenc_range, pad_idx=vocab.pad_idx, 
             mask_idx=vocab.mask_idx, p=0.2, mask_last=False):
    # p = replacement probability
    x,y = b
    rand = torch.rand(x.shape, device=x.device)
    rand[x < word_range[0]] = 1.0
    if mask_last: rand[-1] = 0.0
    y[rand > p] = pad_idx
    x[rand <= (p*.8)] = mask_idx # 80% = mask
    wrong_word = (rand > (p*.8)) & (rand <= (p*.9)) # 10% = wrong word
    x[wrong_word] = torch.randint(*word_range, [wrong_word.sum().item()], device=x.device)
    return x, y

# Sequence 2 Sequence Translate

class S2SFileProcessor(PreProcessor):
    "`PreProcessor` that opens the filenames and read the texts."
    def process_one(self,item):
        out = np.load(item, allow_pickle=True)
        if out.shape != (2,): return None
        if len(out[0]) > 2048: return None
        if len(out[1]) > 2048: return None
        if len(out[0]) < 16: return None
        if len(out[1]) < 16: return None
#         return np.array([out[0].reshape(-1), out[1].reshape(-1)])
        return out
    
    def process(self, ds:Collection):
        ds.items = [self.process_one(item) for item in ds.items]
        ds.items = [i for i in ds.items if i is not None]
#         ds.items = array([self.process_one(item) for item in ds.items], dtype=np.object)

def avg_tempo(t, sep_idx=0):
    avg = t[t[:, 0] == sep_idx][:, 1].sum()/t.shape[0]
    return 'mt'+str(int(max(round(avg), 4)))

class S2SPreloader(Callback):
    def __init__(self, dataset:LabelList, bptt:int=512, y_offset=1, **kwargs):
        # y_offset = extra padding for translation
        self.dataset,self.bptt = dataset,bptt
        self.vocab = vocab
        self.y_offset = y_offset
        self.single_tfm = partial(to_single_stream, vocab=vocab)
        self.transpose_tfm = partial(rand_transpose, note_range=vocab.note_range, rand_range=(0,12))
    
    def __getitem__(self, k:int):
        item,_ = self.dataset[k]
        x,y = item
        
        melody_meta = np.array([self.vocab.stoi[MSEQ], self.vocab.stoi[avg_tempo(x)]]) # pad should be average notes - tempo
        chord_meta = np.array([self.vocab.stoi[CSEQ], self.vocab.stoi[avg_tempo(y)]])
        x = self.single_tfm(x, start_seq=melody_meta)
        y = self.single_tfm(y, start_seq=chord_meta)
        
        if random.randint(0,1) == 1: x,y = y,x # switch translation order around
        x,y = self.transpose_tfm((x,y))
        
        x = np.pad(x, (0,max(0,self.bptt-len(x))), 'constant', constant_values=vocab.pad_idx)[:self.bptt]
        y = np.pad(y, (self.y_offset,max(0,self.bptt-len(y))), 'constant', constant_values=vocab.pad_idx)[:self.bptt+1]
        return x, y
    
    def __len__(self):
        return len(self.dataset)
    
# preloader itself contains all the transforms
def mask_s2s_tfm(b, word_range=vocab.npenc_range, pad_idx=vocab.pad_idx, 
             mask_idx=vocab.mask_idx, p=0.1, double=False, mask_last=False):
    x,y_s2s = b
    x_mask,y_mask = mask_tfm((x,x.clone()))
    return (x,torch.tensor([TaskType.Translate.value]),y_s2s[:,:-1]),(y_mask,y_s2s[:,1:])

# Next Word transform
def nw_tfm(b):
    x,y_nw = b
    x_mask,y_mask = mask_tfm((x,x.clone()), mask_last=True)
    return (x_mask,torch.tensor([TaskType.NextWord.value]),x),(y_mask,y_nw) 
    
    
# DataLoading

class BertTrainer(LearnerCallback):
    "`Callback` that regroups lr adjustment to seq_len, AR and TAR."
    def __init__(self, learn:Learner, ns_data, s2s_data, nw_data):
        super().__init__(learn)
        self.ns_data = ns_data
        self.s2s_data = s2s_data
        self.nw_data = nw_data
        self.count = 0
    
    def on_epoch_end(self, last_metrics, **kwargs):
        "Finish the computation and sends the result to the Recorder."
        if self.count % 3 == 0:
            print('Switching to next sentence data')
            self.learn.data = self.ns_data
        elif self.count % 3 == 1:
            print('Switching to translate data')
            self.learn.data = self.s2s_data
        elif self.count % 3 == 2:
            print('Switching to next word data')
            self.learn.data = self.nw_data
        self.count += 1
        
        
        
# MODEL LOADING

def get_bert_model(vocab_sz:int, config:dict=None, drop_mult:float=1.):
    "Create a language model from `arch` and its `config`, maybe `pretrained`."
    for k in config.keys(): 
        if k.endswith('_p'): config[k] *= drop_mult
#     tie_weights,output_p,out_bias = map(config.pop, ['tie_weights', 'output_p', 'out_bias'])
    tie_weights,output_p,out_bias = map(config.get, ['tie_weights', 'output_p', 'out_bias'])
    n_hid = config['d_model']
    embed = TransformerEmbedding(vocab_sz, n_hid, inp_p=config['embed_p'])
    encoder = BertEncoder(embed=embed, **config)
    mask_decoder = BertLinearDecoder(n_hid, vocab_sz, output_p, tie_encoder=embed.embed, bias=out_bias)
    ns_decoder = BertLinearDecoder(n_hid, 16, output_p, tie_encoder=None, bias=out_bias) # hardcoded max number of next sentence shifts
    s2s_decoder = S2SDecoder(embed, n_hid, vocab_sz, **config)
    model = BertHead(encoder, mask_decoder, ns_decoder, s2s_decoder)
    return model.apply(init_transformer)


def bert_model_learner(data:DataBunch, config:dict=None, drop_mult:float=1., pretrained:bool=False,
                        pretrained_fnames:OptStrTuple=None, **learn_kwargs) -> 'LanguageLearner':
    "Create a `Learner` with a language model from `data` and `arch`."
    model = get_bert_model(config['vocab_size'], config=config, drop_mult=drop_mult)
    learn = MusicLearner(data, model, split_func=tfmerXL_lm_split, 
                         bos_idx=config['bos_idx'], sep_idx=config['sep_idx'],
                        **learn_kwargs)
    
    if pretrained:
        if 'url' not in meta: 
            warn("There are no pretrained weights for that architecture yet!")
            return learn
        model_path = untar_data(meta['url'], data=False)
        fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
        learn.load_pretrained(*fnames)
        learn.freeze()
    if pretrained_fnames is not None:
        fnames = [learn.path/learn.model_dir/f'{fn}.{ext}' for fn,ext in zip(pretrained_fnames, ['pth', 'pkl'])]
        learn.load_pretrained(*fnames)
        learn.freeze()
    return learn

class BertHead(nn.Module):
    def __init__(self, encoder, mask_decoder, ns_decoder, s2s_decoder):
        super().__init__()
        self.encoder = encoder
        self.mask_decoder = mask_decoder
        self.ns_decoder = ns_decoder
        self.s2s_decoder = s2s_decoder
        
    def forward(self, x, task_type=None, y=None):
#         x_emb = self.embed(x)
        task_value = task_type.item() if task_type is not None else task_type
        self.encoder.mask = task_value == TaskType.NextWord.value # mask encoder for next word (so decoder can't cheat)
        x_enc = self.encoder(x)
        x_mask = self.mask_decoder(x_enc) # all tasks include mask decoding
        
        requires_grad(self.ns_decoder, task_value == TaskType.NextSent.value)
        requires_grad(self.s2s_decoder, task_value != TaskType.NextSent.value)
        
        if task_value == TaskType.NextSent.value: # mask, and next sentence task
            dummy_task = self.s2s_decoder(x_enc, torch.zeros_like(x))*0
            return x_mask+dummy_task.sum(), task_type, self.ns_decoder(x_enc)
        if task_value in [TaskType.Translate.value, TaskType.NextWord.value]: 
            # use same translation decoder
            dummy_task = self.ns_decoder(x_enc)*0
            return x_mask+dummy_task.sum(), task_type, self.s2s_decoder(x_enc, y)
        return x_mask, task_type
    
    def __getitem__(self, idx):
        return [self.encoder, self.mask_decoder, self.ns_decoder, self.s2s_decoder][idx]
        
    "A sequential module that passes the reset call to its children."
    def reset(self): pass

# COMPONENTS

class TransformerEmbedding(nn.Module):
    "Embedding + positional encoding + dropout"
    def __init__(self, vocab_sz:int, emb_sz:int, inp_p:float=0.):
        super().__init__()
        self.emb_sz = emb_sz
        self.embed = embedding(vocab_sz, emb_sz)
        self.pos_enc = PositionalEncoding(emb_sz)
        self.drop = nn.Dropout(inp_p)
    
    def forward(self, inp):
        emb = self.drop(self.embed(inp))
        pos = torch.arange(0, inp.size(1), device=inp.device, dtype=emb.dtype)
        return emb, self.pos_enc(pos)

class BertEncoder(nn.Module):
    "TransformerXL model: https://arxiv.org/abs/1901.02860."
    def __init__(self, embed:nn.Module, n_layers:int, n_heads:int, d_model:int, d_head:int, d_inner:int, 
                 resid_p:float=0., attn_p:float=0., ff_p:float=0., bias:bool=False, scale:bool=True,
                 act:Activation=Activation.ReLU, double_drop:bool=True, attn_cls:Callable=MultiHeadRelativeAttention,
                 learned_pos_enc:bool=False, mask:bool=True, **kwargs):
        super().__init__()
        self.encoder = embed
        self.u = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.v = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.n_layers,self.d_model,self.mask = n_layers,d_model,mask
        self.layers = nn.ModuleList([DecoderLayer(n_heads, d_model, d_head, d_inner, resid_p=resid_p, attn_p=attn_p,
                      ff_p=ff_p, bias=bias, scale=scale, act=act, double_drop=double_drop, 
                      attn_cls=attn_cls) for k in range(n_layers)])
        
        nn.init.normal_(self.u, 0., 0.02)
        nn.init.normal_(self.v, 0., 0.02)
    
    def forward(self, x):
        bs,x_len = x.size()
        inp, pos_enc = self.encoder(x)

        mask = lm_mask(x_len, x.device) if self.mask else None
        
        for i, layer in enumerate(self.layers):
            inp = layer(inp, r=pos_enc, u=self.u, v=self.v, mask=mask, mem=None)
        core_out = inp[:,-x_len:]
        return core_out
    
class BertLinearDecoder(nn.Module):
    "To go on top of a RNNCore module and create a Language Model."
    initrange=0.1

    def __init__(self, n_hid:int, n_out:int, output_p:float, tie_encoder:nn.Module=None, bias:bool=True, **kwargs):
        super().__init__()
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.output_dp = RNNDropout(output_p)
        if bias: self.decoder.bias.data.zero_()
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input:Tuple[Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        output = self.output_dp(input)
        decoded = self.decoder(output)
        return decoded

    
# DECODER TRANSLATE BLOCK
class S2SDecoder(nn.Module):
    def __init__(self, embed:nn.Module, n_hid:int, vocab_sz:int, dec_layers:int, n_heads:int, d_model:int, d_head:int, d_inner:int, 
                 resid_p:float=0., attn_p:float=0., ff_p:float=0., bias:bool=False, scale:bool=True,
                 act:Activation=Activation.ReLU, double_drop:bool=True, attn_cls:Callable=MultiHeadRelativeAttention,
                 learned_pos_enc:bool=False, mask:bool=True, **kwargs):
        super().__init__()
        self.encoder = embed
        self.u = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.v = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.dec_layers,self.d_model,self.mask = dec_layers,d_model,mask
        self.layers = nn.ModuleList([S2SDecoderBlock(n_heads, d_model, d_head, d_inner, resid_p=resid_p, attn_p=attn_p,
                      ff_p=ff_p, bias=bias, scale=scale, act=act, double_drop=double_drop, 
                      attn_cls=attn_cls) for k in range(dec_layers)])
        self.head = BertLinearDecoder(d_model, vocab_sz, tie_encoder=embed.embed, **kwargs)
    
        nn.init.normal_(self.u, 0., 0.02)
        nn.init.normal_(self.v, 0., 0.02)
        
    def forward(self, enc, targ):
        # x = encoder, y = target
        bs,targ_len = targ.size()
        
        targ_emb, pos_enc = self.encoder(targ)

#         mask = window_mask(x_len, x.device) if self.mask else None
        mask_out = lm_mask(targ_len, targ.device)
        
        for i, layer in enumerate(self.layers):
            targ_emb = layer(targ_emb, enc, mask_out=mask_out,
                        r=pos_enc, u=self.u, v=self.v)
        return self.head(targ_emb)

class S2SDecoderBlock(nn.Module):
    "Decoder block of a Transformer model."
    #Can't use Sequential directly cause more than one input...
    def __init__(self, n_heads:int, d_model:int, d_head:int, d_inner:int, resid_p:float=0., attn_p:float=0., ff_p:float=0.,
                 bias:bool=True, scale:bool=True, double_drop:bool=True, **kwargs):
        super().__init__()
        self.mha1 = KVMultiHeadRelativeAttention(n_heads, d_model, d_head, resid_p=resid_p, attn_p=attn_p, bias=bias, scale=scale)
        self.mha2 = KVMultiHeadRelativeAttention(n_heads, d_model, d_head, resid_p=resid_p, attn_p=attn_p, bias=bias, scale=scale)
        self.ff   = feed_forward(d_model, d_inner, ff_p=ff_p, double_drop=double_drop)
    
    def forward(self, targ:Tensor, enc:Tensor, 
                r=None, u=None, v=None,
                mask_in:Tensor=None, mask_out:Tensor=None): 
        y = self.mha1(targ, targ, targ, r, u, v, mask=mask_out)
        return self.ff(self.mha2(y, enc, enc, r, u, v, mask=mask_in))
    
class KVMultiHeadRelativeAttention(nn.Module):
    "MutiHeadAttention with relative positional encoding."
    def __init__(self, n_heads:int, d_model:int, d_head:int=None, resid_p:float=0., attn_p:float=0., bias:bool=True,
                 scale:bool=True):
        super().__init__()
        d_head = ifnone(d_head, d_model//n_heads)
        self.n_heads,self.d_head,self.scale = n_heads,d_head,scale
        
        self.q_wgt = nn.Linear(d_model, n_heads * d_head, bias=bias)
        self.k_wgt = nn.Linear(d_model, n_heads * d_head, bias=bias)
        self.v_wgt = nn.Linear(d_model, n_heads * d_head, bias=bias)
        
        self.out = nn.Linear(n_heads * d_head, d_model, bias=bias)
        self.drop_att,self.drop_res = nn.Dropout(attn_p),nn.Dropout(resid_p)
        self.ln = nn.LayerNorm(d_model)
        self.r_attn = nn.Linear(d_model, n_heads * d_head, bias=bias)
        
    def forward(self, q:Tensor, k:Tensor, v:Tensor, 
                r:Tensor=None, g_u:Tensor=None, g_v:Tensor=None, 
                mask:Tensor=None, **kwargs):
        return self.ln(q + self.drop_res(self.out(self._apply_attention(q, k, v, r, g_u, g_v, mask=mask, **kwargs))))
    
    def _apply_attention(self, q:Tensor, k:Tensor, v:Tensor, 
                         r:Tensor=None, g_u:Tensor=None, g_v:Tensor=None, 
                         mask:Tensor=None):
        #Notations from the paper: x input, r vector of relative distance between two elements, u et v learnable
        #parameters of the model common between all layers, mask to avoid cheating and mem the previous hidden states.
        bs,x_len,seq_len = q.size(0),q.size(1),r.size(0)
        wq,wk,wv = self.q_wgt(q),self.k_wgt(k),self.v_wgt(v)
        wq = wq[:,-x_len:]
        wq,wk,wv = map(lambda x:x.view(bs, x.size(1), self.n_heads, self.d_head), (wq,wk,wv))
        wq,wk,wv = wq.permute(0, 2, 1, 3),wk.permute(0, 2, 3, 1),wv.permute(0, 2, 1, 3)
        wkr = self.r_attn(r)
        wkr = wkr.view(seq_len, self.n_heads, self.d_head)
        wkr = wkr.permute(1,2,0)
        #### compute attention score (AC is (a) + (c) and BS is (b) + (d) in the paper)
        AC = torch.matmul(wq+g_u,wk)
        BD = _line_shift(torch.matmul(wq+g_v, wkr))
        if self.scale: attn_score = (AC + BD).mul_(1/(self.d_head ** 0.5))
        if mask is not None: 
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=-1))
        attn_vec = torch.matmul(attn_prob, wv)
        return attn_vec.permute(0, 2, 1, 3).contiguous().view(bs, x_len, -1)
    
# LOSS AND METRICS

class BertLoss():
    def __init__(self):
        self.index_loss = CrossEntropyFlat(ignore_index=vocab.pad_idx)
        self.class_loss = CrossEntropyFlat()
        
    def __call__(self, input:Tensor, target:Tensor, target_2:Tensor, **kwargs)->Rank0Tensor:
        x_mask, task_type, x_task = input
        if task_type is not None: task_type = task_type.item()
        m = self.index_loss.__call__(x_mask, target, **kwargs)
        
        if task_type == TaskType.NextSent.value: s = self.class_loss.__call__(x_task, target_2, **kwargs)
        elif task_type == TaskType.Translate.value: s = self.index_loss.__call__(x_task, target_2, **kwargs)
        elif task_type == TaskType.NextWord.value: s = self.index_loss.__call__(x_task, target_2, **kwargs)
        else: return m

        return m + s
    
def acc_ignore_pad(input:Tensor, targ:Tensor, pad_idx)->Rank0Tensor:
    n = targ.shape[0]
    input = input.argmax(dim=-1).view(n,-1)
    targ = targ.view(n,-1)
    mask = targ != pad_idx
    return (input[mask]==targ[mask]).float().mean()

def mask_acc(input:Tensor, t1:Tensor, t2:Tensor)->Rank0Tensor:
    return acc_ignore_pad(input[0], t1, vocab.pad_idx)

def s2s_acc(input:Tensor, t1:Tensor, t2:Tensor)->Rank0Tensor:
    x_mask, task_type, x_task = input
    if task_type.item() != TaskType.Translate.value: return torch.tensor(0, device=x_mask.device)
    return acc_ignore_pad(x_task, t2, vocab.pad_idx)

def nw_acc(input:Tensor, t1:Tensor, t2:Tensor)->Rank0Tensor:
    x_mask, task_type, x_task = input
    if task_type.item() != TaskType.NextWord.value: return torch.tensor(0, device=x_mask.device)
    return acc_ignore_pad(x_task, t2, vocab.pad_idx)

def ns_acc(input:Tensor, t1:Tensor, t2:Tensor)->Rank0Tensor:
    x_mask, task_type, x_task = input
    if task_type.item() != TaskType.NextSent.value: return torch.tensor(0, device=x_mask.device)
    return accuracy(input[-1], t2)
