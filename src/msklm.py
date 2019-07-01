from .fastai_data import *
from .encode_data import *
from .music_transformer import *
from fastai.basics import *
from fastai.text.models.transformer import _line_shift, init_transformer
from fastai.text.models.awd_lstm import *
from fastai.text.models.transformer import *

from .encode_data import VALTSEP, SAMPLE_FREQ

# DATALOADING AND TRANSFORMATIONS

MLMType = Enum('MLMType', 'Mask, NextWord, M2C, C2M')

# MLM Transform - DEPRECATED
def msklm_mask(shape, p, tile):
    p = p / tile # scale probability
    rand_mask = torch.rand(*shape) < p
    if tile > 1:
        rand_mask = torch.repeat_interleave(rand_mask, tile, dim=1)[:rand_mask.shape[0], :rand_mask.shape[1]]
        
    lm_mask = torch.roll(rand_mask, 1, dims=1)
    lm_mask[:, 0] = 0
    lm_mask = rand_mask & lm_mask
    return rand_mask, lm_mask

def msklm_tfm(b, word_range=vocab.npenc_range, mask_idx=vocab.mask_idx, pad_idx=vocab.pad_idx, p=0.2, tile=1):
    x,y = b
    
    rand_mask, lm_mask = msklm_mask(y.shape, p, tile)
    
    x_msk = y.clone()
    x_msk[rand_mask] = mask_idx
    
    new_rand = torch.rand(*y.shape)
    unchanged_mask = (new_rand < 0.1) & rand_mask # 10%
    wrong_mask = (new_rand > 0.9) & rand_mask # 10% = wrong word
    x_msk[unchanged_mask] = y[unchanged_mask]
    x_msk[wrong_mask] = torch.randint(*word_range, [wrong_mask.sum().item()], device=x.device)

    y_msk = y.clone()
    y_msk[~rand_mask] = pad_idx
    
    x_lm = torch.zeros_like(x)
    x_lm[lm_mask] = x[lm_mask]
    
    return (x_msk, x_lm), y_msk

def random_msklm_tfm(b, mask_idx=vocab.mask_idx, pad_idx=vocab.pad_idx, 
                     p_lm=0.5, p_mask=0.2, p_tile=0.2, 
                     tile_range=(1, 5)):
    if random.random() < p_lm:
        x,y = b
#         x_msk = torch.full_like(y, pad_idx)
        return (None, x), (y, MLMType.NextWord)
    tile = random.randrange(*tile_range) if random.random() < p_tile else 1
    x, y = msklm_tfm(b, p=p_mask, tile=tile)
    return x, (y, MLMType.Mask)

def mask_tfm(b, word_range=vocab.npenc_range, pad_idx=vocab.pad_idx, 
             mask_idx=vocab.mask_idx, p=0.2):
    # p = replacement probability
    x,y = b
    x,y = x.clone(),y.clone()
    rand = torch.rand(x.shape, device=x.device)
    rand[x < word_range[0]] = 1.0
    rand[x >= word_range[1]] = 1.0
    y[rand > p] = pad_idx
    x[rand <= (p*.8)] = mask_idx # 80% = mask
    wrong_word = (rand > (p*.8)) & (rand <= (p*.9)) # 10% = wrong word
    x[wrong_word] = torch.randint(*word_range, [wrong_word.sum().item()], device=x.device)
    return x, y

def mask_or_lm_tfm(b, mask_idx=vocab.mask_idx, pad_idx=vocab.pad_idx, 
                     p_lm=0.5, p_mask=0.2):
    x,y = b
    x,x_pos = x[...,0], x[...,1]
    y,y_pos = y[...,0], y[...,1]
    if random.random() < p_lm:
        return (None, x, None, x_pos), (y, MLMType.NextWord)
    x, y = mask_tfm((y, y), p=p_mask) # masking instead of x. Not that it matters
    return (x, None, y_pos, None), (y, MLMType.Mask)

# Utility for predictions
def mask_note_or_dur(b):
    x, y = b
    x,x_pos = x[...,0], x[...,1]
    y,y_pos = y[...,0], y[...,1]
    x = x.clone()
    rand = torch.rand(x.shape, device=x.device) < 0.9
    mask_range = vocab.dur_range if random.randint(0, 1) == 0 else vocab.note_range
    x[(x >= mask_range[0]) & (x < mask_range[1]) & rand] = vocab.mask_idx
    return (x, None, y_pos, None), (y, MLMType.Mask)

# Utility for predictions
def mask_input(xb, mask_range=vocab.note_range, mask_idx=vocab.mask_idx, clone=True):
    if clone: xb = xb.clone()
    xb[(xb >= mask_range[0]) & (xb < mask_range[1])] = mask_idx
    return xb

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

def avg_tempo(t, sep_idx=VALTSEP):
    avg = t[t[:, 0] == sep_idx][:, 1].sum()/t.shape[0]
    avg = int(round(avg/SAMPLE_FREQ))
    return 'mt'+str(min(avg, MTEMPO_SIZE-1))

def avg_pitch(t, sep_idx=VALTSEP):
    return t[t[:, 0] > sep_idx][:, 0].mean()

class S2SPreloader(Callback):
    def __init__(self, dataset:LabelList, bptt:int=512, 
                 transpose_range=(0,12), **kwargs):
        self.dataset,self.bptt = dataset,bptt
        self.np = vocab
        self.transpose_tfm = partial(rand_transpose_tfm, note_range=vocab.note_range, rand_range=transpose_range)
        
    def __getitem__(self, k:int):
        item,_ = self.dataset[k]
        x,y = item
        x,y = self.transpose_tfm([x,y])
        # WARNING: we are padding position encodings too. However, pos is negative, so should be fine
        return pad_seq(x, self.bptt+1), pad_seq(y, self.bptt+1) # offset bptt for decoder shift
    
    def __len__(self):
        return len(self.dataset)
    
def part_tfm(b):
    x,y = b
    x = partenc2seq2seq(x, part_type=MSEQ)
    y = partenc2seq2seq(y, part_type=CSEQ)
    return x, y
    
def partenc2seq2seq(part_np, part_type, vocab=vocab, add_eos=True):
    part_meta = np.array([vocab.stoi[part_type], vocab.pad_idx])
    s2s_out = to_single_stream(part_np, start_seq=part_meta)
    if add_eos: s2s_out = np.pad(s2s_out, (0,1), 'constant', constant_values=vocab.stoi[EOS])
#     s2s_out = position_tfm(s2s_out)
    return s2s_out

def pad_seq(seq, bptt, pad_idx=vocab.pad_idx):
    pad_len = max(bptt-seq.shape[0], 0)
    return np.pad(seq, [(0, pad_len),(0,0)], 'constant', constant_values=pad_idx)[:bptt]

def combined_npenc2chordarr(np1, np2):
    if len(np1.shape) == 1: np1 = to_double_stream(np1)
    if len(np2.shape) == 1: np1 = to_double_stream(np2)
    p1 = npenc2chordarr(np1)
    p2 = npenc2chordarr(np2)
    max_ts = max(p1.shape[0], p2.shape[0])
    p1w = ((0,max_ts-p1.shape[0]),(0,0),(0,0))
    p1_pad = np.pad(p1, p1w, 'constant')
    p2w = ((0,max_ts-p2.shape[0]),(0,0),(0,0))
    p2_pad = np.pad(p2, p2w, 'constant')
    chordarr_comb = np.concatenate((p1_pad, p2_pad), axis=1)
    return chordarr_comb

# preloader itself contains all the transforms
def s2s_tfm(b, mlm_type=MLMType.M2C):
    x,y = b if mlm_type == MLMType.M2C else reversed(b)
    
    x,x_pos = x[...,0], x[...,1]
    y,y_pos = y[...,0], y[...,1]
    
    return (x[:,:-1], y[:,:-1], x_pos[:,:-1], y_pos[:,:-1]),(y[:,1:], mlm_type)

# DataLoading
class CombinedDS(Callback):
    def __init__(self, dss):
        self.dss = self.dss
    def __getattr__(self, attr):
        def redirected(self, *args, **kwargs):
            for ds in self.dss:
                if hasattr(ds, attr):
                    getattr(ds, attr)(*args, **kwargs)
        return redirected

class CombinedDL():
    def __init__(self, dls, num_it=100):
        self.dls = dls
        self.dataset = CombinedDS([dl.dataset for dl in dls if hasattr(dl, 'dataset')])
        self.num_it = num_it
        self.dl_idx = -1
        
    def __len__(self)->int: return sum([len(dl) for dl in self.dls])
        
    def __iter__(self):
        "Process and returns items from `DataLoader`."
        iters = [iter(dl) for dl in self.dls]
        self.dl_idx = -1
        while len(iters):
            self.dl_idx = (self.dl_idx+1) % len(iters)
            for b in range(self.num_it):
                try:
                    yield next(iters[self.dl_idx])
                except StopIteration as e:
                    iters.remove(iters[self.dl_idx])
                    break
#         raise StopIteration

class CombinedData():
    def __init__(self, dbs, num_it=100):
        self.dbs = dbs
        self.train_dl = CombinedDL([db.train_dl for db in self.dbs], num_it)
        self.valid_dl = CombinedDL([db.valid_dl for db in self.dbs], num_it)
        self.vocab = vocab
        self.train_ds = None
        self.path = dbs[0].path
        self.device = dbs[0].device
        self.empty_val = False

    def add_tfm(self,tfm:Callable)->None:
        for dl in self.dbs: dl.add_tfm(tfm)

    def remove_tfm(self,tfm:Callable)->None:
        for dl in self.dbs: dl.remove_tfm(tfm)
        
class MLMLearner(MusicLearner):


    def predict_nw(self, xb:Tensor, n_words:int=128,
                     temperatures:float=(1.0,1.0), min_bars=4,
                     top_k=40, top_p=0.9):
        "Return the `n_words` that come after `text`."
        self.model.reset()

        seed = xb.cpu().numpy().squeeze()
        new_idx = []
        pos = torch.tensor(-position_enc(xb.cpu().numpy()), device=xb.device)
        last_pos = pos[-1]

        sep_count = 0
        bar_len = SAMPLE_FREQ * 4 # assuming 4/4 time
        vocab = self.data.vocab

        with torch.no_grad():
            for i in progress_bar(range(n_words), leave=True):
                res = self.pred_batch(batch=((None, xb[None], None, pos[None]),xb[None]))[-1][-1]
                res = F.softmax(res, dim=-1)

                # bar = 16 beats
                if (sep_count // 16) <= min_bars: res[vocab.bos_idx] = 0.

                # Use first temperatures value if last prediction was duration
                temperature = temperatures[0] if (len(new_idx)==0 or self.data.vocab.is_duration(new_idx[-1])) else temperatures[1]
                if temperature != 1.: res.pow_(1 / temperature)

                res = top_k_top_p_filtering(res, top_k=top_k, top_p=top_p, filter_value=0)
                idx = torch.multinomial(res, 1).item()

                if new_idx and new_idx[-1]==vocab.sep_idx: 
                    duration = idx - vocab.dur_range[0]
                    sep_count += duration
    #                 print('Updating positional encoding:', last_pos, last_pos - duration)
                    last_pos = last_pos - duration # position is negative
    #                 print('Bars', duration, sep_count // 16)

                if idx==vocab.bos_idx: 
                    print('Predicted BOS token. Returning prediction...')
                    break

                new_idx.append(idx)
                xb = xb.new_tensor([idx])
                pos = pos.new_tensor([last_pos])
        return np.array(new_idx), seed


    def predict_mask(self, xb:Tensor,
                    temperatures:float=(1.0,1.0),
                    top_k=20, top_p=0.8):
        xb = xb.clone().squeeze()[None]
        self.model.reset()
        mask_idxs = (xb == vocab.mask_idx).nonzero()
        for midx in progress_bar(mask_idxs, leave=True):

            pos = torch.tensor(-position_enc(xb[0].cpu().numpy()), device=xb.device)[None]
    #         print(pos)

            # Next Word
            res = self.pred_batch(batch=((xb, None, pos, None),xb))
            res = F.softmax(res[tuple(midx)], dim=-1) # task1, task2 - (bs x ts x vocab)

            # Don't allow any special tokens (as we are only removing notes and durations)
            res[vocab.bos_idx] = 0.
            res[vocab.sep_idx] = 0.
            res[vocab.stoi[EOS]] = 0

            # Use first temperatures value if last prediction was duration
            temperature = temperatures[0]
            if temperature != 1.: res.pow_(1 / temperature)

            res = top_k_top_p_filtering(res, top_k=top_k, top_p=top_p, filter_value=0)
            idx = torch.multinomial(res, 1).item()
            #         idx = res.argmax()

            xb[tuple(midx)] = idx

        return xb.cpu().numpy()


    def predict_s2s(self, xb_msk:Tensor, xb_lm:Tensor, n_words:int=128,
                    temperatures:float=(1.0,1.0),
                    top_k=40, top_p=0.9):
        self.model.reset()


        x_lm = xb_lm.tolist()
        lm_pos = (-position_enc(xb_lm.cpu().numpy())).tolist()
        last_pos = lm_pos[-1]

        msk_pos = torch.tensor(-position_enc(xb_msk.cpu().numpy()), device=xb_msk.device)
        x_enc = self.model.encoder(xb_msk.view(1, -1), msk_pos.view(1, -1))

        max_pos = msk_pos[-1] + SAMPLE_FREQ * 4

        for i in progress_bar(range(n_words), leave=True):

            # Next Word
            x, pos = torch.tensor(x_lm, device=xb_lm.device)[None], torch.tensor(lm_pos, device=xb_lm.device)[None]
            dec = self.model.decoder(x, pos, x_enc) # all tasks include mask decoding
            res = F.softmax(self.model.head(dec), dim=-1)[-1, -1]

            # Use first temperatures value if last prediction was duration
            temperature = temperatures[0] if (len(x_lm)==0 or self.data.vocab.is_duration(x_lm[-1])) else temperatures[1]
            if temperature != 1.: res.pow_(1 / temperature)

            res = top_k_top_p_filtering(res, top_k=top_k, top_p=top_p, filter_value=0)
            idx = torch.multinomial(res, 1).item()
            #         idx = res.argmax()

            if idx == vocab.bos_idx | idx == vocab.stoi[EOS]: 
                print('Predicting BOS/EOS')
                break

            if x_lm and x_lm[-1]==vocab.sep_idx: 
                duration = idx - vocab.dur_range[0]
    #             sep_count += duration
                last_pos = last_pos - duration # position is negative
                if last_pos < max_pos+SAMPLE_FREQ * 4:
                    print('Predicted past counter-part length. Returning early')
                    break

            lm_pos.append(last_pos)
            x_lm.append(idx)

        return np.array(x_lm)

# High level serve api
def part_enc(chordarr, part):
    partarr = chordarr[:,part:part+1,:]
    npenc = chordarr2npenc(partarr)
    return npenc
    

def s2s_predict_from_midi(learn, midi=None, n_words=200, 
                      temperatures=(1.0,1.0), top_k=24, top_p=0.7, pred_melody=True, **kwargs):

    stream = file2stream(midi) # 1.
    chordarr = stream2chordarr(stream) # 2.
    _,num_parts,_ = chordarr.shape
    p1, p2 = [part_enc(chordarr, i) for i in range(num_parts)]
    
    part_order = (MSEQ, CSEQ) if avg_pitch(p1) > avg_pitch(p2) else (CSEQ, MSEQ)
    
    mpart = torch.tensor(partenc2seq2seq(p1, part_type=MSEQ, add_eos=False))
    cpart = torch.tensor(partenc2seq2seq(p2, part_type=CSEQ, add_eos=False))
    
    xb, yb = (cpart, mpart) if pred_melody else (mpart, cpart)
    if torch.cuda.is_available(): xb, yb = xb.cuda(), yb.cuda()
    
    pred = learn.predict_s2s(xb, yb, n_words=n_words, temperatures=temperatures, top_k=top_k, top_p=top_p)
    # pred = yb

    seed_npenc = to_double_stream(xb.cpu().numpy()) # chord
    yb_npenc = to_double_stream(pred) # melody
    npenc_order = [yb_npenc, seed_npenc] if pred_melody else [seed_npenc, yb_npenc]
    chordarr_comb = combined_npenc2chordarr(*npenc_order)

    return chordarr_comb


def nw_predict_from_midi(learn, midi=None, n_words=600, 
                      temperatures=(1.0,1.0), top_k=24, top_p=0.7, **kwargs):
    seed_np = to_single_stream(midi2npenc(midi)) # music21 can handle bytes directly
    xb = torch.tensor(seed_np)
    if torch.cuda.is_available(): xb = xb.cuda()
    pred, seed = learn.predict_nw(xb, n_words=n_words, temperatures=temperatures, top_k=top_k, top_p=top_p)
    seed = to_double_stream(seed)
    pred = to_double_stream(pred)
    full = np.concatenate((seed,pred), axis=0)
    return full

def mask_predict_from_midi(learn, midi=None,
                           temperatures=(1.0,1.0), top_k=20, top_p=0.8, 
                           predict_notes=True,
                           **kwargs):
    seed_np = midi2npenc(midi) # music21 can handle bytes directly
    xb = torch.tensor(to_single_stream(seed_np))[None]
    mask_range = vocab.note_range if predict_notes else vocab.dur_range
    if predict_notes:
        mask_input(xb, mask_range=mask_range, clone=False)
    else:
        mask_input(xb[10:], mask_range=mask_range, clone=False)
    if torch.cuda.is_available(): xb = xb.cuda()
    pred = learn.predict_mask(xb, temperatures=temperatures, top_k=top_k, top_p=top_p)
    pred = to_double_stream(pred)
    return pred
        
# MODEL LOADING

class MLMTrainer(LearnerCallback):
    "`Callback` that regroups lr adjustment to seq_len, AR and TAR."
    def __init__(self, learn:Learner, dataloaders=None, starting_mask_window=1):
        super().__init__(learn)
        self.count = 1
        self.mw_start = starting_mask_window
        self.dataloaders = dataloaders

    def on_epoch_begin(self, **kwargs):
        "Reset the hidden state of the model."
        model = get_model(self.learn.model)
        model.reset()
#         model.encoder.mask_size = max(self.count+self.mw_start, 100)
        
    def on_epoch_end(self, last_metrics, **kwargs):
        "Finish the computation and sends the result to the Recorder."
        if self.dataloaders is not None: self.learn.data = self.dataloaders[self.count % len(self.dataloaders)]
        self.count += 1


def get_mlm_model(vocab_sz:int, config:dict=None, drop_mult:float=1.):
    "Create a language model from `arch` and its `config`, maybe `pretrained`."
    for k in config.keys(): 
        if k.endswith('_p'): config[k] *= drop_mult
#     tie_weights,output_p,out_bias = map(config.pop, ['tie_weights', 'output_p', 'out_bias'])
    tie_weights,output_p,out_bias = map(config.get, ['tie_weights', 'output_p', 'out_bias'])
    n_hid = config['d_model']
    embed = TransformerEmbedding(vocab_sz, n_hid, embed_p=config['embed_p'], mem_len=config['mem_len'])
    encoder = MLMEncoder(embed, n_hid, **config)
    decoder = MLMEncoder(embed, n_hid, is_decoder=True, **config)
    head = MLMLinearDecoder(n_hid, vocab_sz, tie_encoder=embed.embed, **config)
    model = MLMTransformer(encoder, decoder, head, mem_len=config['mem_len'])
    return model.apply(init_transformer)


def mlm_model_learner(data:DataBunch, config:dict=None, drop_mult:float=1., pretrained:bool=False,
                        pretrained_fnames:OptStrTuple=None, **learn_kwargs) -> 'LanguageLearner':
    "Create a `Learner` with a language model from `data` and `arch`."
    model = get_mlm_model(config['vocab_size'], config=config, drop_mult=drop_mult)
#     learn = UnilmLearner(data, model, config=config, split_func=tfmerXL_lm_split,
#     learn = UnilmLearner(data, model, config=config, split_func=None,
#                         **learn_kwargs)
    learn = MLMLearner(data, model, config=config, split_func=None,
                        **learn_kwargs)
    return learn

# Attn

class MemMultiHeadRelativeAttentionKV(nn.Module):
    "MutiHeadAttention with relative positional encoding."
    def __init__(self, n_heads:int, d_model:int, d_head:int=None, resid_p:float=0., attn_p:float=0., bias:bool=True,
                 scale:bool=True, mem_len:int=512, r_mask=True):
        super().__init__()
        d_head = ifnone(d_head, d_model//n_heads)
        self.n_heads,self.d_head,self.scale = n_heads,d_head,scale
        
        assert(d_model == d_head * n_heads)
#         self.out = nn.Linear(n_heads * d_head, d_model, bias=bias)
        self.q_wgt = nn.Linear(d_model, n_heads * d_head, bias=bias)
        self.k_wgt = nn.Linear(d_model, n_heads * d_head, bias=bias)
        self.v_wgt = nn.Linear(d_model, n_heads * d_head, bias=bias)
        
        self.drop_att,self.drop_res = nn.Dropout(attn_p),nn.Dropout(resid_p)
        self.ln = nn.LayerNorm(d_model)
        self.r_attn = nn.Linear(d_model, n_heads * d_head, bias=bias)
        self.r_mask = r_mask

        self.mem_len = mem_len
        self.prev_k = None
        self.prev_v = None
        
    def forward(self, q:Tensor, k:Tensor=None, v:Tensor=None, 
                r:Tensor=None, g_u:Tensor=None, g_v:Tensor=None, 
                mask:Tensor=None, **kwargs):
        if k is None: k = q
        if v is None: v = q
#         return self.ln(q + self.drop_res(self.out(self._apply_attention(q, k, v, r, g_u, g_v, mask=mask, **kwargs))))
        return self.ln(q + self.drop_res(self._apply_attention(q, k, v, r, g_u, g_v, mask=mask, **kwargs)))

    def mem_k(self, k):
        if self.mem_len == 0: return k
        if self.prev_k is None or (self.prev_k.shape[0] != k.shape[0]): # reset if wrong batch size
            self.prev_k = k[:, -self.mem_len:]
            return k
        with torch.no_grad():
            k_ext = torch.cat([self.prev_k, k], dim=1)
            self.prev_k = k_ext[:, -self.mem_len:]
        return k_ext.detach()
    
    def mem_v(self, v):
        if self.mem_len == 0: return v
        if self.prev_v is None or (self.prev_v.shape[0] != v.shape[0]): # reset if wrong batch size
            self.prev_v = v[:, -self.mem_len:]
            return v
        with torch.no_grad():
            v_ext = torch.cat([self.prev_v, v], dim=1)
            self.prev_v = v_ext[:, -self.mem_len:]
        return v_ext.detach()
        
    def reset(self):
        self.prev_v = None
        self.prev_k = None
        
    def _apply_attention(self, q:Tensor, k:Tensor, v:Tensor, 
                         r:Tensor=None, g_u:Tensor=None, g_v:Tensor=None, 
                         mask:Tensor=None, **kwargs):
        #Notations from the paper: x input, r vector of relative distance between two elements, u et v learnable
        #parameters of the model common between all layers, mask to avoid cheating and mem the previous hidden states.
#         bs,x_len,seq_len = q.size(0),q.size(1),r.size(0)
        k = self.mem_k(k)
        v = self.mem_v(v)
        bs,x_len,seq_len = q.size(0),q.size(1),k.size(1)
        wq,wk,wv = self.q_wgt(q),self.k_wgt(k),self.v_wgt(v)
        wq = wq[:,-x_len:]
        wq,wk,wv = map(lambda x:x.view(bs, x.size(1), self.n_heads, self.d_head), (wq,wk,wv))
        wq,wk,wv = wq.permute(0, 2, 1, 3),wk.permute(0, 2, 3, 1),wv.permute(0, 2, 1, 3)
        wkr = self.r_attn(r[-seq_len:])
        wkr = wkr.view(seq_len, self.n_heads, self.d_head)
        wkr = wkr.permute(1,2,0)
        #### compute attention score (AC is (a) + (c) and BS is (b) + (d) in the paper)
        AC = torch.matmul(wq+g_u,wk)
        BD = _line_shift(torch.matmul(wq+g_v, wkr), mask=self.r_mask)
        if self.scale: attn_score = (AC + BD).mul_(1/(self.d_head ** 0.5))
        if mask is not None: 
            mask = mask[...,-seq_len:]
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=-1))
        attn_vec = torch.matmul(attn_prob, wv)
        return attn_vec.permute(0, 2, 1, 3).contiguous().view(bs, x_len, -1)
    
    
class MLMTransformer(nn.Module):
    def __init__(self, encoder, decoder, head, mem_len):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        self.default_mem_len = mem_len
        self.current_mem_len = None
    
    def forward(self, x_msk, x_lm, msk_pos, lm_pos):
        if x_msk is None:
            reset_children(self.encoder)
            return self.head(self.decoder(x_lm, lm_pos))
        if x_lm is None:
            reset_children(self.decoder)
            return self.head(self.encoder(x_msk, msk_pos))
        self.reset()
        x_msk = self.encoder(x_msk, msk_pos)
        dec = self.decoder(x_lm, lm_pos, x_msk) # all tasks include mask decoding
        return self.head(dec)
    
    "A sequential module that passes the reset call to its children."
    def reset(self):
        for module in self.children(): 
            reset_children(module)
            
    def update_mem_len(self, use_mem):
        # Only Next word predictions should have memory
        next_mem_len = self.default_mem_len if use_mem else 0
        if self.current_mem_len == next_mem_len: return
        # print('Updating mem length to:', next_mem_len)
        for module in self.children(): 
            update_mem_len(module, next_mem_len)
        self.current_mem_len = next_mem_len
        self.reset()
        
def reset_children(mod):
    if hasattr(mod, 'reset'): mod.reset()
    for module in mod.children(): 
        reset_children(module)
        
def update_mem_len(mod, mem_len):
    if hasattr(mod, 'mem_len'): mod.mem_len = mem_len
    for module in mod.children(): 
        update_mem_len(module, mem_len)
 # COMPONENTS

class TransformerEmbedding(nn.Module):
    "Embedding + positional encoding + dropout"
    def __init__(self, vocab_sz:int, emb_sz:int, embed_p:float=0., mem_len=512, beat_len=32):
        super().__init__()
        self.emb_sz = emb_sz
        
        self.embed = nn.Embedding(vocab_sz, emb_sz, padding_idx=vocab.pad_idx)
        # See https://arxiv.org/abs/1711.09160
        with torch.no_grad(): trunc_normal_(self.embed.weight, std=0.01)
#         self.embed = embedding(vocab_sz, emb_sz)
        self.pos_enc = PositionalEncoding(emb_sz)
        self.initrange = 0.05
        self.beat_len = beat_len
        self.beat_enc = nn.Embedding(beat_len, emb_sz, padding_idx=0) # negative pad
        self.bar_enc = nn.Embedding(1024, emb_sz, padding_idx=0) # negative pad
#         self.bar_enc = PositionalEncoding(emb_sz) # positional encoding doesn't work for multi dimensions right now

        self.beat_enc.weight.data.uniform_(-self.initrange, self.initrange)
        self.bar_enc.weight.data.uniform_(-self.initrange, self.initrange)
        
        
        self.drop = nn.Dropout(embed_p)
        self.mem_len = mem_len
    
    def forward(self, inp, pos_enc):
#         pdb.set_trace()
#        return self.drop(self.embed(inp))
        pe = -pos_enc.clone()
        pe[pe==-vocab.pad_idx] = 0
        
        beat_enc = self.beat_enc(pe % self.beat_len)
        bar_pos = pe // self.beat_len % 1024
#        bar_pos[bar_pos > 4096] = 4095
        bar_enc = self.bar_enc((bar_pos))
#        bar_enc = self.bar_enc((pe // self.beat_len).type(beat_enc.dtype))
#        assert((pe//self.beat_len < 1024).all())
        emb = self.drop(self.embed(inp) + beat_enc + bar_enc)
        return emb
    
    def relative_pos_enc(self, emb):
        seq_len = emb.shape[1] + self.mem_len
        pos = torch.arange(seq_len-1, -1, -1, device=emb.device, dtype=emb.dtype) # backwards (txl pos encoding)
        return self.pos_enc(pos)

class MLMLinearDecoder(nn.Module):
    "To go on top of a RNNCore module and create a Language Model."
    initrange=0.1

    def __init__(self, n_hid:int, n_out:int, output_p:float, tie_encoder:nn.Module=None, out_bias:bool=True, **kwargs):
        super().__init__()
        self.decoder = nn.Linear(n_hid, n_out, bias=out_bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.output_dp = RNNDropout(output_p)
        if out_bias: self.decoder.bias.data.zero_()
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input:Tuple[Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        output = self.output_dp(input)
        decoded = self.decoder(output)
        return decoded

    
# DECODER TRANSLATE BLOCK
class MLMEncoder(nn.Module):
    def __init__(self, embed:nn.Module, n_hid:int, n_layers:int, n_heads:int, d_model:int, d_head:int, d_inner:int, 
                 resid_p:float=0., attn_p:float=0., ff_p:float=0., bias:bool=True, scale:bool=True,
                 act:Activation=Activation.ReLU, double_drop:bool=True,
                 mask:bool=True, mem_len:int=512, is_decoder=False, **kwargs):
        super().__init__()
        self.embed = embed
        self.u = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.v = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.n_layers,self.d_model,self.mask = n_layers,d_model,mask
        self.layers = nn.ModuleList([MLMEncoderBlock(n_heads, d_model, d_head, d_inner, resid_p=resid_p, attn_p=attn_p,
                      ff_p=ff_p, bias=bias, scale=scale, act=act, double_drop=double_drop, mem_len=mem_len,
                      is_decoder=is_decoder) for k in range(n_layers)])
        self.mask_size = 1
    
        nn.init.normal_(self.u, 0., 0.02)
        nn.init.normal_(self.v, 0., 0.02)
        
    def forward(self, x_lm, lm_pos, msk_emb=None):
        bs,lm_len = x_lm.size()
        
        lm_emb = self.embed(x_lm, lm_pos)
        if msk_emb is not None and msk_emb.shape[1] > lm_emb.shape[1]:
            pos_enc = self.embed.relative_pos_enc(msk_emb)
        else:
            pos_enc = self.embed.relative_pos_enc(lm_emb)
    
        # Masks
        if self.mask:
            lm_mask = rand_window_mask(lm_len, self.embed.mem_len, x_lm.device,
                                       max_size=self.mask_size, p=0.3, is_eval=not self.train)
        else:
            lm_mask = None
        
        for i, layer in enumerate(self.layers):
            lm_emb = layer(lm_emb, msk_emb, lm_mask=lm_mask,
                        r=pos_enc, g_u=self.u, g_v=self.v)
        return lm_emb

class MLMEncoderBlock(nn.Module):
    "Decoder block of a Transformer model."
    #Can't use Sequential directly cause more than one input...
    def __init__(self, n_heads:int, d_model:int, d_head:int, d_inner:int, resid_p:float=0., attn_p:float=0., ff_p:float=0.,
                 bias:bool=True, scale:bool=True, double_drop:bool=True, mem_len:int=512,
                 is_decoder=False, **kwargs):
        super().__init__()
        attn_cls = MemMultiHeadRelativeAttentionKV
        self.mha1 = attn_cls(n_heads, d_model, d_head, resid_p=resid_p, attn_p=attn_p, bias=bias, scale=scale, mem_len=mem_len, r_mask=False)
        self.mha2 = attn_cls(n_heads, d_model, d_head, resid_p=resid_p, attn_p=attn_p, bias=bias, scale=scale, mem_len=mem_len, r_mask=True)
#         self.mha2 = attn_cls(n_heads, d_model, d_head, resid_p=resid_p, attn_p=attn_p, bias=bias, scale=scale, mem_len=mem_len, r_mask=True)
        self.ff   = feed_forward(d_model, d_inner, ff_p=ff_p, double_drop=double_drop)
    
    def forward(self, enc_lm:Tensor, enc_msk:Tensor,
                r=None, g_u=None, g_v=None,
                msk_mask:Tensor=None, lm_mask:Tensor=None): 
        
        y_lm = self.mha1(enc_lm, enc_lm, enc_lm, r, g_u, g_v, mask=lm_mask)
        if enc_msk is None: return y_lm
        return self.ff(self.mha2(y_lm, enc_msk, enc_msk, r, g_u, g_v, mask=msk_mask))
    
# LOSS AND METRICS
class MLMLoss():
    def __init__(self):
        "Loss mult - Mask, NextWord, Seq2Seq"
        self.loss = CrossEntropyFlat(ignore_index=vocab.pad_idx)
        
    def __call__(self, input:Tensor, target:Tensor, mlm_type:Tensor=None, **kwargs)->Rank0Tensor:
        return self.loss(input, target)
    
def acc_ignore_pad(input:Tensor, targ:Tensor, mlm_type=None, acc_type=None, pad_idx=vocab.pad_idx)->Rank0Tensor:
    if acc_type is not None and mlm_type != acc_type: return None
    n = targ.shape[0]
    input = input.argmax(dim=-1).view(n,-1)
    targ = targ.view(n,-1)
    mask = targ != pad_idx
    return (input[mask]==targ[mask]).float().mean()

def mask_acc(*args): return acc_ignore_pad(*args, acc_type=MLMType.Mask)
def lm_acc(*args): return acc_ignore_pad(*args, acc_type=MLMType.NextWord)
def c2m_acc(*args): return acc_ignore_pad(*args, acc_type=MLMType.C2M)
def m2c_acc(*args): return acc_ignore_pad(*args, acc_type=MLMType.M2C)
