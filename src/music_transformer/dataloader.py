"Fastai Language Model Databunch modified to work with music"
from fastai.basics import *
# from fastai.basic_data import DataBunch
from fastai.text.data import LMLabelList
from ..data_encode import *
# Additional encoding

BOS = 'xxbos'
PAD = 'xxpad'
EOS = 'xxeos'
S2SCLS = 'xxs2scls' # used for sequence2sequence start of translation
MASK = 'xxmask'
CSEQ = 'xxcseq'
MSEQ = 'xxmseq'
NSCLS = 'xxnscls'

SEP = 'xxsep' # separator idx = -1 (part of notes)

SPECIAL_TOKS = [BOS, PAD, EOS, S2SCLS, MASK, CSEQ, MSEQ, NSCLS, SEP] # Important: SEP token must be last

NOTE_TOKS = [f'n{i}' for i in range(NOTE_SIZE)] 
DUR_TOKS = [f'd{i}' for i in range(DUR_SIZE)]
NOTE_START, NOTE_END = NOTE_TOKS[0], NOTE_TOKS[-1]
DUR_START, DUR_END = DUR_TOKS[0], DUR_TOKS[-1]

MTEMPO_SIZE = 10
MTEMPO_OFF = 'mt0'
MTEMPO_TOKS = [f'mt{i}' for i in range(MTEMPO_SIZE)]

# Vocab - token to index mapping
class MusicVocab():
    "Contain the correspondence between numbers and tokens and numericalize."
    def __init__(self, itos:Collection[str]):
        self.itos = itos
        self.stoi = {v:k for k,v in enumerate(self.itos)}

    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return [self.stoi[w] for w in t]

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        return sep.join([self.itos[i] for i in nums]) if sep is not None else [self.itos[i] for i in nums]
    @property 
    def mask_idx(self): return self.stoi[MASK]
    @property 
    def pad_idx(self): return self.stoi[PAD]
    @property
    def bos_idx(self): return self.stoi[BOS]
    @property
    def sep_idx(self): return self.stoi[SEP]
    @property
    def npenc_range(self): return (vocab.stoi[SEP], vocab.stoi[DUR_END]+1)
    @property
    def note_range(self): return vocab.stoi[NOTE_START], vocab.stoi[NOTE_END]+1
    @property
    def dur_range(self): return vocab.stoi[DUR_START], vocab.stoi[DUR_END]+1

    def is_duration(self, idx): 
        return idx >= self.dur_range[0] and idx < self.dur_range[1]
        
    def __getstate__(self):
        return {'itos':self.itos}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.stoi = {v:k for k,v in enumerate(self.itos)}

    def save(self, path):
        "Save `self.itos` in `path`"
        pickle.dump(self.itos, open(path, 'wb'))

    @classmethod
    def create(cls) -> 'Vocab':
        "Create a vocabulary from a set of `tokens`."
        itos = SPECIAL_TOKS + NOTE_TOKS + DUR_TOKS + MTEMPO_TOKS
        if len(itos)%8 != 0:
            itos = itos + [f'dummy{i}' for i in range(len(itos)%8)]
        return cls(itos)
    
    @classmethod
    def load(cls, path):
        "Load the `Vocab` contained in `path`"
        itos = pickle.load(open(path, 'rb'))
        return cls(itos)

vocab = MusicVocab.create()

class MusicItem(ItemBase):
    def __init__(self, item, vocab):
        self.data = item
        self.vocab = vocab
        self.stream = None
    def __repr__(self): return self.data[:10]

    @classmethod
    def from_file(cls, midi_file, vocab):
        return MusicItem(midi2idxenc(midi_file, vocab), vocab)

    def to_stream(self, bpm=120):
        if self.stream is None: 
            self.stream = idxenc2stream(self.data, bpm=bpm)
        return self.stream

    def show_score(self):
        self.to_stream().show()

    def play_file(self):
        self.to_stream().show('midi')

    def trim_to_beat(self):
        self.

def seed_tfm(idxenc, seed_len=None, sample_freq=SAMPLE_FREQ):
    if seed_len is None: return idxenc
    pos = -neg_position_enc(idxenc)
    cutoff = np.searchsorted(pos, seed_len * sample_freq) + 1
    return idxenc[:cutoff]
    
def midi2idxenc(midi_file, vocab):
    "Converts midi file to index encoding for training"
    npenc = midi2npenc(midi_file) # 3.
    return npenc2idxenc(npenc, vocab)

def idxenc2stream(arr, vocab, bpm=120):
    "Converts index encoding to music21 stream"
    npenc = idxenc2npenc(arr, vocab)
    return npenc2stream(npenc, bpm=bpm)

# single stream instead of note,dur
def npenc2idxenc(t, vocab=vocab, start_seq=None):
    "Transforms numpy array from 2 column (note, duration) matrix to a single column"
    "[[n1, d1], [n2, d2], ...] -> [n1, d1, n2, d2]"
    if isinstance(t, (list, tuple)) and len(t) == 2: 
        return [npenc2idxenc(x, vocab, start_seq) for x in t]
    t = t.copy()
    
    t[:, 0] = t[:, 0] + vocab.note_range[0]
    t[:, 1] = t[:, 1] + vocab.dur_range[0]
    if start_seq is None: start_seq = np.array([vocab.bos_idx, vocab.pad_idx])
    return np.concatenate([start_seq, t.reshape(-1)])

def idxenc2npenc(t, vocab=vocab, validate=True):
    if validate: t = to_valid_npenc(t, vocab.npenc_range)
    t = t.copy().reshape(-1, 2)
    if t.shape[0] == 0: return
        
    t[:, 0] = t[:, 0] - vocab.note_range[0]
    t[:, 1] = t[:, 1] - vocab.dur_range[0]
    
    is_note = (t[:, 0] < VALTSEP) | (t[:, 0] >= NOTE_SIZE)
    invalid_note_idx = is_note.argmax()
    if invalid_note_idx > 0: 
        print('Non midi note detected. Only returning valid portion. Index, seed', invalid_note_idx, t.shape)
        return t[:invalid_note_idx]
    return t

def to_valid_npenc(t, valid_range):
    r = valid_range
    t = t[np.where((t >= r[0]) & (t < r[1]))]
    if t.shape[-1] % 2 == 1: t = t[..., :-1]
    return t

def neg_position_enc(idxenc, vocab=vocab):
    "Calculates positional beat encoding."
    "Note: returns negative position to prevent index collision with vocab"
    sep_idxs = (idxenc == vocab.sep_idx).nonzero()[0]
    sep_idxs = sep_idxs[sep_idxs+2 < idxenc.shape[0]] # remove any indexes right before out of bounds (sep_idx+2)
    dur_vals = idxenc[sep_idxs+1]
    dur_vals[dur_vals == vocab.mask_idx] = vocab.dur_range[0] # make sure masked durations are 0
    dur_vals -= vocab.dur_range[0]
    
    posenc = np.zeros_like(idxenc)
    posenc[sep_idxs+2] = dur_vals
    return -posenc.cumsum()

def position_tfm(idxenc, vocab=vocab):
    posenc = neg_position_enc(idxenc, vocab) # using negative values so we don't interfere with indexes
    return np.stack([idxenc, posenc], axis=1)

def tfm_transpose(x, value, note_range=vocab.note_range):
    x = x.copy()
    x[(x >= note_range[0]) & (x < note_range[1])] += value
    return x

def rand_transpose_tfm(t, note_range=vocab.note_range, rand_range=(0,24), p=0.5):
    if np.random.rand() < p:
        transpose_value = np.random.randint(*rand_range)-rand_range[1]//2
        if isinstance(t, (list, tuple)) and len(t) == 2: 
            return [tfm_transpose(x, transpose_value, note_range) for x in t]
        return tfm_transpose(t, transpose_value, note_range)
    return t

## For npenc dataset
class MusicPreloader(Callback):
    "Transforms the tokens in `dataset` to a stream of contiguous batches for language modelling."
    
    class CircularIndex():
        "Handles shuffle, direction of indexing, wraps around to head tail in the ragged array as needed"
        def __init__(self, length:int, forward:bool): self.idx, self.forward = np.arange(length), forward
        def __getitem__(self, i): 
            return self.idx[ i%len(self.idx) if self.forward else len(self.idx)-1-i%len(self.idx)]
        def __len__(self) -> int: return len(self.idx)
        def shuffle(self): np.random.shuffle(self.idx)

    def __init__(self, dataset:LabelList, lengths:Collection[int]=None, bs:int=32, bptt:int=70, backwards:bool=False, 
                 shuffle:bool=False, y_offset:int=1, 
                 transpose_range=(0,24), transpose_p=0.5,
                 **kwargs):
        self.dataset,self.bs,self.bptt,self.shuffle,self.backwards,self.lengths = dataset,bs,bptt,shuffle,backwards,lengths
        self.bs *= num_distrib() or 1
        self.totalToks,self.ite_len,self.idx = int(0),None,None
        self.y_offset = y_offset
        
        self.transpose_range,self.transpose_p = transpose_range,transpose_p
        self.bptt_len = self.bptt
        
        self.allocate_buffers() # needed for valid_dl on distributed training - otherwise doesn't get initialized on first epoch

    def __len__(self): 
        if self.ite_len is None:
            if self.lengths is None: self.lengths = np.array([len(item) for item in self.dataset.x])
            self.totalToks = self.lengths.sum()
            self.ite_len   = self.bs*int( math.ceil( self.totalToks/(self.bptt*self.bs) )) if self.item is None else 1
        return self.ite_len

    def __getattr__(self,k:str)->Any: return getattr(self.dataset, k)
   
    def allocate_buffers(self):
        "Create the ragged array that will be filled when we ask for items."
        if self.ite_len is None: len(self)
        self.idx   = MusicPreloader.CircularIndex(len(self.dataset.x), not self.backwards)
        self.batch = np.zeros((self.bs, self.bptt+self.y_offset) + self.dataset.x[0].shape[1:], dtype=np.int64)
        self.batch_x, self.batch_y = self.batch[:,0:self.bptt], self.batch[:,self.y_offset:self.bptt+self.y_offset] 
        #ro: index of the text we're at inside our datasets for the various batches
        self.ro    = np.zeros(self.bs, dtype=np.int64)
        #ri: index of the token we're at inside our current text for the various batches
        self.ri    = np.zeros(self.bs, dtype=np.int)
        
        # allocate random transpose values. Need to allocate this before hand.
        self.transpose_values = self.get_random_transpose_values()
        
    def get_random_transpose_values(self):
        if self.transpose_range is None: return None
        n = len(self.dataset)
        rt_arr = torch.randint(*self.transpose_range, (n,))-self.transpose_range[1]//2
        mask = torch.rand(rt_arr.shape) > self.transpose_p
        rt_arr[mask] = 0
        return rt_arr

    def on_epoch_begin(self, **kwargs):
        if self.idx is None: self.allocate_buffers()
        elif self.shuffle:   
            self.ite_len = None
            len(self)
            self.idx.shuffle()
            self.transpose_values = self.get_random_transpose_values()
            self.bptt_len = self.bptt
        self.idx.forward = not self.backwards 

        step = self.totalToks / self.bs
        ln_rag, countTokens, i_rag = 0, 0, -1
        for i in range(0,self.bs):
            #Compute the initial values for ro and ri 
            while ln_rag + countTokens <= int(step * i):
                countTokens += ln_rag
                i_rag       += 1
                ln_rag       = self.lengths[self.idx[i_rag]]
            self.ro[i] = i_rag
            self.ri[i] = ( ln_rag - int(step * i - countTokens) ) if self.backwards else int(step * i - countTokens)
        
    #Training dl gets on_epoch_begin called, val_dl, on_epoch_end
    def on_epoch_end(self, **kwargs): self.on_epoch_begin()

    def __getitem__(self, k:int):
        j = k % self.bs
        if j==0:
            if self.item is not None: return self.dataset[0]
            if self.idx is None: self.on_epoch_begin()
                
        self.ro[j],self.ri[j] = self.fill_row(not self.backwards, self.dataset.x, self.idx, self.batch[j][:self.bptt_len+self.y_offset], 
                                              self.ro[j], self.ri[j], overlap=1, lengths=self.lengths)
        return self.batch_x[j][:self.bptt_len], self.batch_y[j][:self.bptt_len]

    def fill_row(self, forward, items, idx, row, ro, ri, overlap, lengths):
        "Fill the row with tokens from the ragged array. --OBS-- overlap != 1 has not been implemented"
        ibuf = n = 0 
        ro  -= 1
        while ibuf < row.shape[0]:  
            ro   += 1 
            ix    = idx[ro]
            
            rag   = items[ix]
            if self.transpose_values is not None: 
                rag = tfm_transpose(rag, self.transpose_values[ix].item())
                
            if forward:
                ri = 0 if ibuf else ri
                n  = min(lengths[ix] - ri, row.shape[0] - ibuf)
                row[ibuf:ibuf+n] = rag[ri:ri+n]
            else:    
                ri = lengths[ix] if ibuf else ri
                n  = min(ri, row.size - ibuf) 
                row[ibuf:ibuf+n] = rag[ri-n:ri][::-1]
            ibuf += n
        return ro, ri + ((n-overlap) if forward else -(n-overlap))

class IndexEncodeProcessor(PreProcessor):
    "`PreProcessor` that transforms numpy files to indexes for training"
    def __init__(self, ds:ItemList=None, vocab:MusicVocab=None):
        self.vocab = ifnone(vocab, ds.vocab if ds is not None else None)

    def process_one(self,item):
        return npenc2idxenc(item, vocab=self.vocab)
    
    def process(self, ds):
        if self.vocab is None: self.vocab = MusicVocab.create()
        ds.vocab = self.vocab
        super().process(ds)
        
class PositionProcessor(IndexEncodeProcessor):
    "`PreProcessor` that opens the filenames and read the texts."
    def process_one(self,item):
        item = position_tfm(item, vocab=self.vocab)
        return item

class OpenNPFileProcessor(PreProcessor):
    "`PreProcessor` that opens the filenames and read the texts."
    def process_one(self,item):
        return np.load(item, allow_pickle=True) if isinstance(item, Path) else item

class MusicDataBunch(DataBunch):
    "Create a `TextDataBunch` suitable for training a language model."
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', no_check:bool=False, bs=64, val_bs:int=None, 
               num_workers:int=0, device:torch.device=None, collate_fn:Callable=data_collate, 
               dl_tfms:Optional[Collection[Callable]]=None, bptt:int=70,
               preloader_cls=MusicPreloader, shuffle_dl=False, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` in `path` from the `datasets` for language modelling."
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        datasets = [preloader_cls(ds, shuffle=(i==0), bs=(bs if i==0 else val_bs), bptt=bptt, **kwargs) 
                    for i,ds in enumerate(datasets)]
        val_bs = bs
        dls = [DataLoader(d, b, shuffle=shuffle_dl) for d,b in zip(datasets, (bs,val_bs,val_bs,val_bs)) if d is not None]
        return cls(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)
    
    @classmethod    
    def from_folder(cls, path:PathOrStr, processors=None, extensions='.npy', split_pct=0.1, **kwargs):
        files = get_files(path, extensions=extensions, recurse=True);
        src = (MusicItemList(items=files, path=path, processor=processors)
                .split_by_rand_pct(split_pct, seed=6)
                .label_const(label_cls=LMLabelList))
        return src.databunch(**kwargs)
    
class MusicItemList(ItemList):
    _bunch = MusicDataBunch
