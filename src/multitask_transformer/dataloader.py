from fastai.basics import *
from ..music_transformer.dataloader import MusicVocab
# Sequence 2 Sequence Translate

class S2SFileProcessor(PreProcessor):
    "`PreProcessor` that opens the filenames and read the texts."
    def process_one(self,item):
        out = np.load(item, allow_pickle=True)
        if out.shape != (2,): return None
        if not 16 < len(out[0]) < 2048: return None
        if not 16 < len(out[1]) < 2048: return None
        return out
    
    def process(self, ds:Collection):
        ds.items = [self.process_one(item) for item in ds.items]
        ds.items = [i for i in ds.items if i is not None] # filter out None
#         ds.items = array([self.process_one(item) for item in ds.items], dtype=np.object)

class S2SPartEncProcessor(PreProcessor):
    "Encodes midi file into 2 separate parts - melody and chords."
    def __init__(self, ds:ItemList=None, vocab:MusicVocab=None):
        self.vocab = ifnone(vocab, ds.vocab if ds is not None else None)
        
    def process_one(self,item):
        m, c = item
        m = position_tfm(partenc2seq2seq(m, MSEQ, vocab=self.vocab))
        c = position_tfm(partenc2seq2seq(c, CSEQ, vocab=self.vocab))
        return np.array((m, c))
    
    def process(self, ds):
        if self.vocab is None: self.vocab = MusicVocab.create()
        ds.vocab = self.vocab
        ds.items = [self.process_one(item) for item in ds.items]
        
class S2SPreloader(Callback):
    def __init__(self, dataset:LabelList, bptt:int=512, 
                 transpose_range=(0,12), **kwargs):
        self.dataset,self.bptt = dataset,bptt
        self.vocab = self.dataset.vocab
        self.transpose_tfm = partial(rand_transpose_tfm, note_range=vocab.note_range, rand_range=transpose_range) if transpose_range is not None else None
        
    def __getitem__(self, k:int):
        item,_ = self.dataset[k]
        x,y = item
        if self.transpose_tfm is not None:
            x,y = self.transpose_tfm([x,y])
        # WARNING: we are padding position encodings too. However, pos is negative, so should be fine
        return pad_seq(x, self.bptt+1, self.vocab.pad_idx), pad_seq(y, self.bptt+1, self.vocab.pad_idx) # offset bptt for decoder shift
    
    def __len__(self):
        return len(self.dataset)
        
def pad_seq(seq, bptt, value):
    pad_len = max(bptt-seq.shape[0], 0)
    return np.pad(seq, [(0, pad_len),(0,0)], 'constant', constant_values=value)[:bptt]
    
def partenc2seq2seq(part_np, part_type, vocab, add_eos=True):
    part_meta = np.array([vocab.stoi[part_type], vocab.pad_idx])
    s2s_out = npenc2idxenc(part_np, vocab=vocab, start_seq=part_meta)
    if add_eos: s2s_out = np.pad(s2s_out, (0,1), 'constant', constant_values=vocab.stoi[EOS])
    return s2s_out

def s2s_combine2chordarr(np1, np2, vocab):
    if len(np1.shape) == 1: np1 = idxenc2npenc(np1, vocab)
    if len(np2.shape) == 1: np2 = idxenc2npenc(np2, vocab)
    p1 = npenc2chordarr(np1)
    p2 = npenc2chordarr(np2)
    return chordarr_combine_parts(p1, p2)

def midi_extract_melody_chords(midi, vocab):
    stream = file2stream(midi) # 1.
    chordarr = stream2chordarr(stream) # 2.
    _,num_parts,_ = chordarr.shape

    if num_parts == 1:
        # if predicting melody, assume only track is chord track
        p1, p2 = part_enc(chordarr, 0), np.zeros((0,2), dtype=int)
        p1, p2 = (p2, p1) if pred_melody else (p1, p2)
    elif num_parts == 2:
        p1, p2 = [part_enc(chordarr, i) for i in range(num_parts)]
        p1, p2 = (p1, p2) if avg_pitch(p1) > avg_pitch(p2) else (p2, p1)
    else:
        raise ValueError('Could not extract melody and chords from midi file. Please make sure file contains exactly 2 tracks')
        
    mpart = partenc2seq2seq(p1, part_type=MSEQ, vocab=vocab)
    cpart = partenc2seq2seq(p2, part_type=CSEQ, vocab=vocab)
    return mpart, cpart


# DATALOADING AND TRANSFORMATIONS

# MLMType = Enum('MLMType', 'Mask, NextWord, M2C, C2M')

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

# Utility for predictions
# def mask_note_or_dur(b):
#     x, y = b
#     x,x_pos = x[...,0], x[...,1]
#     y,y_pos = y[...,0], y[...,1]
#     x = x.clone()
#     rand = torch.rand(x.shape, device=x.device) < 0.9
#     mask_range = vocab.dur_range if random.randint(0, 1) == 0 else vocab.note_range
#     x[(x >= mask_range[0]) & (x < mask_range[1]) & rand] = vocab.mask_idx
#     return (x, None, y_pos, None), (y, MLMType.Mask)


def mask_lm_tfm(b, mask_idx=vocab.mask_idx, pad_idx=vocab.pad_idx, p_mask=0.2):
    x,y = b
    x_lm,x_pos = x[...,0], x[...,1]
    y_lm,y_pos = y[...,0], y[...,1]
    
    x_msk, y_msk = mask_tfm((y_lm, y_lm), p=p_mask) # masking instead of x. Just in case we ever do sequential s2s training
    msk_pos = y_pos
    
    x_dict = { 
        'msk': { 'x': x_msk, 'pos': msk_pos },
        'lm': { 'x': x_lm, 'pos': msk_pos }
    }
    y_dict = { 'msk': y_msk, 'lm': y_lm }
    return x_dict, y_dict

def melody_chord_tfm(b):
    m,c = b
    
    m,m_pos = m[...,0], m[...,1]
    c,c_pos = c[...,0], c[...,1]
    
    # offset x and y for next word prediction
    y_m = m[:,1:]
    x_m, m_pos = m[:,:-1], m_pos[:,:-1]
    
    y_c = c[:,1:]
    x_c, c_pos = c[:,:-1], c_pos[:,:-1]
    
    x_dict = { 
        'c2m': {
            'enc': x_c,
            'enc_pos': c_pos,
            'dec': x_m,
            'dec_pos': m_pos
        },
        'm2c': {
            'enc': x_m,
            'enc_pos': m_pos,
            'dec': x_c,
            'dec_pos': c_pos
        }
    }
    y_dict = {
        'c2m': y_m, 'm2c': y_c
    }
    return x_dict, y_dict
