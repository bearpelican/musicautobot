"Fast parallel databunch creation and special npencoding DataBunch"
# from fastai.text import *
from numbers import Integral
from .encode_data import npenc2seq

from fastai.basics import *
from fastai.text.data import LMLabelList

# Additional encoding

BOS = 'xxbos'
PAD = 'xxpad'
EOS = 'xxeos'
CLS = 'xxcls'
MASK = 'xxmask'
CSEQ = 'xxcseq'
MSEQ = 'xxmseq'
FSEQ = 'xxfseq'

SEP = 'xxsep' # separator idx = -1 (part of notes)

SPECIAL_TOKS = [BOS, PAD, EOS, CLS, MASK, CSEQ, MSEQ, FSEQ, SEP] # Important: SEP token must be last


NOTE_RANGE = 130
DUR_RANGE = 130

NOTE_TOKS = [f'n{i}' for i in range(NOTE_RANGE)] 
DUR_TOKS = [f'd{i}' for i in range(DUR_RANGE)]
NOTE_START, NOTE_END = NOTE_TOKS[0], NOTE_TOKS[-1]
DUR_START, DUR_END = DUR_TOKS[0], DUR_TOKS[-1]

MTEMPO_OFF = 'mt0'
MTEMPO_TOKS = [f'mt{i}' for i in range(5)]

# Vocab - token to index mapping
class MusicVocab():
    "Contain the correspondence between numbers and tokens and numericalize."
    def __init__(self, itos:Collection[str]):
        self.itos = itos
#         self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})
        self.stoi = {v:k for k,v in enumerate(self.itos)}

#     def numericalize(self, t:Collection[str]) -> List[int]:
#         "Convert a list of tokens `t` to their ids."
#         return [self.stoi[w] for w in t]

#     def textify(self, nums:Collection[int], sep=' ') -> List[str]:
#         "Convert a list of `nums` to their tokens."
#         return sep.join([self.itos[i] for i in nums]) if sep is not None else [self.itos[i] for i in nums]
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
#         self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})
        self.stoi = {v:k for k,v in enumerate(self.itos)}

    def save(self, path):
        "Save `self.itos` in `path`"
        pickle.dump(self.itos, open(path, 'wb'))

    @classmethod
    def create(cls) -> 'Vocab':
        "Create a vocabulary from a set of `tokens`."
        itos = SPECIAL_TOKS + NOTE_TOKS + DUR_TOKS + MTEMPO_TOKS
        return cls(itos)
    
    @classmethod
    def load(cls, path):
        "Load the `Vocab` contained in `path`"
        itos = pickle.load(open(path, 'rb'))
        return cls(itos)

vocab = MusicVocab.create()

# single stream instead of note,dur
def to_single_stream(t, vocab=vocab, start_seq=None):
    if isinstance(t, (list, tuple)) and len(t) == 2: 
        return [to_single_stream(x, vocab, start_seq) for x in t]
    t = t.copy()
    t[:, 0] = t[:, 0] + vocab.note_range[0]
    t[:, 1] = t[:, 1] + vocab.dur_range[0]
    if start_seq is None: start_seq = np.array([vocab.stoi[BOS], vocab.stoi[PAD]])
    return np.concatenate([start_seq, t.reshape(-1)])

def to_double_stream(t, vocab=vocab):
    t = t.copy().reshape(-1, 2)
    t[:, 0] = t[:, 0] - vocab.note_range[0]
    t[:, 1] = t[:, 1] - vocab.dur_range[0]
    return t

def tfm_transpose(x, value, note_range):
    x = x.copy()
    x[(x >= note_range[0]) & (x < note_range[1])] += value
    return x

def rand_transpose(t, note_range=None, rand_range=(0,24), p=0.5):
    if np.random.rand() < p:
        transpose_value = np.random.randint(*rand_range)-rand_range[1]//2
        if isinstance(t, (list, tuple)) and len(t) == 2: 
            return [tfm_transpose(x, transpose_value, note_range) for x in t]
        return tfm_transpose(t, transpose_value, note_range)
    return t

def rand_transpose_double(t, rand_range=(0,24), p=0.5):
    "For transposing double column encoded midi"
    if np.random.rand() < p:
        t = t.copy()
        notes = t[...,0]
        notes[notes > VALTSEP] += np.random.randint(*rand_range)-rand_range[1]//2
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
                 shuffle:bool=False, y_offset:int=1, **kwargs):
        self.dataset,self.bs,self.bptt,self.shuffle,self.backwards,self.lengths = dataset,bs,bptt,shuffle,backwards,lengths
        self.bs *= num_distrib() or 1
        self.totalToks,self.ite_len,self.idx = int(0),None,None
        self.y_offset = y_offset
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

    def on_epoch_begin(self, **kwargs):
        if self.idx is None: self.allocate_buffers()
        elif self.shuffle:   self.idx.shuffle()
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
        self.ro[j],self.ri[j] = self.fill_row(not self.backwards, self.dataset.x, self.idx, self.batch[j], 
                                              self.ro[j], self.ri[j], overlap=1, lengths=self.lengths)
        return self.batch_x[j], self.batch_y[j]

    def fill_row(self, forward, items, idx, row, ro, ri, overlap, lengths):
        "Fill the row with tokens from the ragged array. --OBS-- overlap != 1 has not been implemented"
        ibuf = n = 0 
        ro  -= 1
        while ibuf < row.shape[0]:  
            ro   += 1 
            ix    = idx[ro]
            rag   = items[ix]
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

    
class OpenNPFileProcessor(PreProcessor):
    "`PreProcessor` that opens the filenames and read the texts."
    def process_one(self,item):
        return np.load(item, allow_pickle=True) if isinstance(item, Path) else item
    
class MusicDataBunch(DataBunch):
    "Create a `TextDataBunch` suitable for training a language model."
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', no_check:bool=False, bs=64, val_bs:int=None, 
               num_workers:int=0, device:torch.device=None, collate_fn:Callable=data_collate, 
               dl_tfms:Optional[Collection[Callable]]=None, bptt:int=70, backwards:bool=False, y_offset:int=1,
               preloader_cls=MusicPreloader, shuffle_dl=False, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` in `path` from the `datasets` for language modelling."
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        datasets = [preloader_cls(ds, shuffle=(i==0), bs=(bs if i==0 else val_bs), bptt=bptt, backwards=backwards, y_offset=y_offset) 
                    for i,ds in enumerate(datasets)]
        val_bs = bs
        dls = [DataLoader(d, b, shuffle=shuffle_dl) for d,b in zip(datasets, (bs,val_bs,val_bs,val_bs)) if d is not None]
        return cls(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)
    
    @classmethod    
    def from_ids(cls, path:PathOrStr, train_ids:Collection[Collection[int]], valid_ids:Collection[Collection[int]],
                 test_ids:Collection[Collection[int]]=None, train_lbls:Collection[Union[int,float]]=None,
                 valid_lbls:Collection[Union[int,float]]=None, classes:Collection[Any]=None,
                 processor:PreProcessor=None,
                 train_tfms=None, valid_tfms=None,
                 **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from ids, labels and a `vocab`. `kwargs` are passed to the dataloader creation."
        src = ItemLists(path, MusicItemList(train_ids, path=path, processor=[], tfms=train_tfms),
                        MusicItemList(valid_ids, path=path, processor=[], tfms=valid_tfms))
        src = src.label_const(label_cls=LMLabelList)
        if not is1d(train_lbls): src.train.y.one_hot,src.valid.y.one_hot = True,True
        return src.databunch(**kwargs)
    
    def save(self, cache_name:PathOrStr='tmp'):
        "Save the `DataBunch` in `self.path/cache_name` folder."
        os.makedirs(self.path/cache_name, exist_ok=True)
        cache_path = self.path/cache_name
        np.save(cache_path/f'train_ids.npy', self.train_ds.x.items)
        np.save(cache_path/f'train_lbl.npy', self.train_ds.y.items)
        np.save(cache_path/f'valid_ids.npy', self.valid_ds.x.items)
        np.save(cache_path/f'valid_lbl.npy', self.valid_ds.y.items)
        if self.test_dl is not None: np.save(cache_path/f'test_ids.npy', self.test_ds.x.items)
        if hasattr(self.train_ds, 'classes'): save_texts(cache_path/'classes.txt', self.train_ds.classes)

    @classmethod
    def load(cls, path:PathOrStr, cache_name:PathOrStr='tmp', processor:PreProcessor=None, **kwargs):
        "Load a `TextDataBunch` from `path/cache_name`. `kwargs` are passed to the dataloader creation."
        cache_path = Path(path)/cache_name
        train_ids,train_lbls = np.load(cache_path/f'train_ids.npy', allow_pickle=True), np.load(cache_path/f'train_lbl.npy', allow_pickle=True)
        valid_ids,valid_lbls = np.load(cache_path/f'valid_ids.npy', allow_pickle=True), np.load(cache_path/f'valid_lbl.npy', allow_pickle=True)
        test_ids = np.load(cache_path/f'test_ids.npy', allow_pickle=True) if os.path.isfile(cache_path/f'test_ids.npy') else None
        classes = loadtxt_str(cache_path/'classes.txt') if os.path.isfile(cache_path/'classes.txt') else None
        return cls.from_ids(path, train_ids, valid_ids, test_ids, train_lbls, valid_lbls, classes, processor, **kwargs)

class MusicItemList(ItemList):
    _bunch = MusicDataBunch
    
    def __init__(self, *args, tfms=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tfms = tfms or []
        
    def get(self, i)->Any:
        item = self.items[i]
        if self.tfms is None: return item
        for tfm in self.tfms: item = tfm(item)
        return item
