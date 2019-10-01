from fastai.basics import *
from fastai.text.models.transformer import Activation, PositionalEncoding, feed_forward, init_transformer, _line_shift
from fastai.text.models.awd_lstm import RNNDropout
from ..utils.attention_mask import *

def get_multitask_model(vocab_size:int, config:dict=None, drop_mult:float=1., pad_idx=None):
    "Create a language model from `arch` and its `config`, maybe `pretrained`."
    for k in config.keys(): 
        if k.endswith('_p'): config[k] *= drop_mult
    n_hid = config['d_model']
    mem_len = config.pop('mem_len')
    embed = TransformerEmbedding(vocab_size, n_hid, embed_p=config['embed_p'], mem_len=mem_len, pad_idx=pad_idx)
    encoder = MTEncoder(embed, n_hid, n_layers=config['enc_layers'], mem_len=0, **config) # encoder doesn't need memory
    decoder = MTEncoder(embed, n_hid, is_decoder=True, n_layers=config['dec_layers'], mem_len=mem_len, **config)
    head = MTLinearDecoder(n_hid, vocab_size, tie_encoder=embed.embed, **config)
    model = MultiTransformer(encoder, decoder, head, mem_len=mem_len)
    return model.apply(init_transformer)

class MultiTransformer(nn.Module):
    "Multitask Transformer for training mask, next word, and sequence 2 sequence"
    def __init__(self, encoder, decoder, head, mem_len):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        self.default_mem_len = mem_len
        self.current_mem_len = None
    
    def forward(self, inp):
        # data order: mask, next word, melody, chord
        outputs = {}
        msk, lm, c2m, m2c = [inp.get(key) for key in ['msk', 'lm', 'c2m', 'm2c']]
        
        if msk is not None:
            outputs['msk'] = self.head(self.encoder(msk['x'], msk['pos']))
        if lm is not None:
            outputs['lm'] = self.head(self.decoder(lm['x'], lm['pos']))
        
        if c2m is not None:
            self.reset()
            c2m_enc = self.encoder(c2m['enc'], c2m['enc_pos'])
            c2m_dec = self.decoder(c2m['dec'], c2m['dec_pos'], c2m_enc)
            outputs['c2m'] = self.head(c2m_dec)
            
        if m2c is not None:
            self.reset()
            m2c_enc = self.encoder(m2c['enc'], m2c['enc_pos'])
            m2c_dec = self.decoder(m2c['dec'], m2c['dec_pos'], m2c_enc)
            outputs['m2c'] = self.head(m2c_dec)
            
        return outputs
    
    "A sequential module that passes the reset call to its children."
    def reset(self):
        for module in self.children(): 
            reset_children(module)
        
def reset_children(mod):
    if hasattr(mod, 'reset'): mod.reset()
    for module in mod.children(): 
        reset_children(module)

 # COMPONENTS
class TransformerEmbedding(nn.Module):
    "Embedding + positional encoding + dropout"
    def __init__(self, vocab_size:int, emb_sz:int, embed_p:float=0., mem_len=512, beat_len=32, max_bar_len=1024, pad_idx=None):
        super().__init__()
        self.emb_sz = emb_sz
        self.pad_idx = pad_idx
        
        self.embed = nn.Embedding(vocab_size, emb_sz, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(emb_sz)
        self.beat_len, self.max_bar_len = beat_len, max_bar_len
        self.beat_enc = nn.Embedding(beat_len, emb_sz, padding_idx=0)
        self.bar_enc = nn.Embedding(max_bar_len, emb_sz, padding_idx=0)
        
        self.drop = nn.Dropout(embed_p)
        self.mem_len = mem_len
    
    def forward(self, inp, pos):
        beat_enc = self.beat_enc(pos % self.beat_len)
        bar_pos = pos // self.beat_len % self.max_bar_len
        bar_pos[bar_pos >= self.max_bar_len] = self.max_bar_len - 1
        bar_enc = self.bar_enc((bar_pos))
        emb = self.drop(self.embed(inp) + beat_enc + bar_enc)
        return emb
    
    def relative_pos_enc(self, emb):
#         return torch.arange(640-1, -1, -1).float().cuda()
        seq_len = emb.shape[1] + self.mem_len
        pos = torch.arange(seq_len-1, -1, -1, device=emb.device, dtype=emb.dtype) # backwards (txl pos encoding)
        return self.pos_enc(pos)

class MTLinearDecoder(nn.Module):
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
class MTEncoder(nn.Module):
    def __init__(self, embed:nn.Module, n_hid:int, n_layers:int, n_heads:int, d_model:int, d_head:int, d_inner:int, 
                 resid_p:float=0., attn_p:float=0., ff_p:float=0., bias:bool=True, scale:bool=True,
                 act:Activation=Activation.ReLU, double_drop:bool=True, mem_len:int=512, is_decoder=False,
                 mask_steps=1, mask_p=0.3, **kwargs):
        super().__init__()
        self.embed = embed
        self.u = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.v = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.n_layers,self.d_model = n_layers,d_model
        self.layers = nn.ModuleList([MTEncoderBlock(n_heads, d_model, d_head, d_inner, resid_p=resid_p, attn_p=attn_p,
                      ff_p=ff_p, bias=bias, scale=scale, act=act, double_drop=double_drop, mem_len=mem_len,
                      ) for k in range(n_layers)])

        self.mask_steps, self.mask_p = mask_steps, mask_p
        self.is_decoder = is_decoder
    
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
        if self.is_decoder:
            lm_mask = rand_window_mask(lm_len, self.embed.mem_len, x_lm.device,
                                       max_size=self.mask_steps, p=self.mask_p, is_eval=not self.training)
        else:
            lm_mask = None
        
        for i, layer in enumerate(self.layers):
            lm_emb = layer(lm_emb, msk_emb, lm_mask=lm_mask,
                        r=pos_enc, g_u=self.u, g_v=self.v)
        return lm_emb

class MTEncoderBlock(nn.Module):
    "Decoder block of a Transformer model."
    #Can't use Sequential directly cause more than one input...
    def __init__(self, n_heads:int, d_model:int, d_head:int, d_inner:int, resid_p:float=0., attn_p:float=0., ff_p:float=0.,
                 bias:bool=True, scale:bool=True, double_drop:bool=True, mem_len:int=512, mha2_mem_len=0, **kwargs):
        super().__init__()
        attn_cls = MemMultiHeadRelativeAttentionKV
        self.mha1 = attn_cls(n_heads, d_model, d_head, resid_p=resid_p, attn_p=attn_p, bias=bias, scale=scale, mem_len=mem_len, r_mask=False)
        self.mha2 = attn_cls(n_heads, d_model, d_head, resid_p=resid_p, attn_p=attn_p, bias=bias, scale=scale, mem_len=mha2_mem_len, r_mask=True)
        self.ff   = feed_forward(d_model, d_inner, ff_p=ff_p, double_drop=double_drop)
    
    def forward(self, enc_lm:Tensor, enc_msk:Tensor,
                r=None, g_u=None, g_v=None,
                msk_mask:Tensor=None, lm_mask:Tensor=None): 
        
        y_lm = self.mha1(enc_lm, enc_lm, enc_lm, r, g_u, g_v, mask=lm_mask)
        if enc_msk is None: return y_lm
        return self.ff(self.mha2(y_lm, enc_msk, enc_msk, r, g_u, g_v, mask=msk_mask))
    

    # Attention Layer


# Attn

class MemMultiHeadRelativeAttentionKV(nn.Module):
    "Attention Layer monster - relative positioning, keeps track of own memory, separate kv weights to support sequence2sequence decoding."
    def __init__(self, n_heads:int, d_model:int, d_head:int=None, resid_p:float=0., attn_p:float=0., bias:bool=True,
                 scale:bool=True, mem_len:int=512, r_mask=True):
        super().__init__()
        d_head = ifnone(d_head, d_model//n_heads)
        self.n_heads,self.d_head,self.scale = n_heads,d_head,scale
        
        assert(d_model == d_head * n_heads)
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
            if hasattr(mask, 'bool'): mask = mask.bool()
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=-1))
        attn_vec = torch.matmul(attn_prob, wv)
        return attn_vec.permute(0, 2, 1, 3).contiguous().view(bs, x_len, -1)
