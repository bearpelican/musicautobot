from fastai.text import *
from numbers import Integral
from encode_data import npenc2seq

TO_SEQ = False
NO_INST = True
Y_OFFSET=4
VAL_OFFSET=0
    
class MusicTokenizer():
    def __init__(self):
        super().__init__()
        self.n_cpus = num_cpus()
    def process_text(self, t:str) -> List[str]: return t.split(" ")
    def _process_all_1(self, texts:Collection[str]) -> List[List[str]]:
        return [self.process_text(t) for t in texts]
    def process_all(self, texts:Collection[str]) -> List[List[str]]:
        "Process a list of `texts`."
        if self.n_cpus <= 1: return self._process_all_1(texts)
        with ProcessPoolExecutor(self.n_cpus) as e:
            return sum(e.map(self._process_all_1, partition_by_cores(texts, self.n_cpus)), [])


def lm_join_texts(texts:Collection[str]):
    return [f'{BOS} {t}' for t in texts]

class LMOpenFileProcessor(OpenFileProcessor):
    # Removing numpy array conversion to fix OOM error
    def process(self, ds:Collection): ds.items = [self.process_one(item) for item in ds.items] 

class LMTokenizeProcessor(PreProcessor):
    "`PreProcessor` that tokenizes the texts in `ds`."
    def __init__(self, ds:ItemList=None, tokenizer:Tokenizer=None, chunksize:int=10000):
        self.tokenizer,self.chunksize = ifnone(tokenizer, Tokenizer()),chunksize
    def process_one(self, item):  return self.tokenizer._process_all_1([item])[0]
    def process(self, ds):
        ds.items = lm_join_texts(ds.items)
        tokens = []
        for i in progress_bar(range(0,len(ds),self.chunksize), leave=False):
            tokens += self.tokenizer.process_all(ds.items[i:i+self.chunksize])
        ds.items = tokens

def numericalize(data):
    stoi, items = data
    return [[stoi[w] for w in item] for item in items]

def count_tokens(tokens): return Counter(p for o in tokens for p in o)    
def vocab_create_parallel(tokens:Tokens, max_vocab:int, min_freq:int) -> 'Vocab':
    "Create a vocabulary from a set of `tokens`."
    n_cpus = num_cpus()
    print("Creating vocabulary")
    gc.collect()
    with ProcessPoolExecutor(n_cpus) as e:
        freq = sum(e.map(count_tokens, partition_by_cores(tokens, n_cpus*10)), Counter())
        
    print("Counting done")
    itos = [o for o,c in freq.most_common(max_vocab) if c > min_freq]
    for o in reversed(defaults.text_spec_tok):
        if o in itos: itos.remove(o)
        itos.insert(0, o)
    return Vocab(itos)
Vocab.create = vocab_create_parallel

class LMNumericalizeProcessor(PreProcessor):
    "`PreProcessor` that numericalizes the tokens in `ds`."
    def __init__(self, ds:ItemList=None, vocab:Vocab=None, max_vocab:int=60000, min_freq:int=2):
        vocab = ifnone(vocab, ds.vocab if ds is not None else None)
        self.vocab,self.max_vocab,self.min_freq = vocab,max_vocab,min_freq

    def process_one(self,item): return np.array(self.vocab.numericalize(item), dtype=np.int64)
    def process(self, ds):
        if self.vocab is None: self.vocab = vocab_create_parallel(ds.items, self.max_vocab, self.min_freq)
#         if self.vocab is None: self.vocab = Vocab.create(ds.items, self.max_vocab, self.min_freq)
        ds.vocab = self.vocab
        print("Numericalizing")
        n_cpus = num_cpus()
        parts = partition_by_cores(ds.items, n_cpus*4)
        vocabs = [ds.vocab.stoi.copy() for i in range(len(parts))]
        with ProcessPoolExecutor(n_cpus) as e:
            items = sum(e.map(numericalize, zip(vocabs, parts)), [])
        gc.collect()
        ds.items = array(items)
        
        
        
## For npenc dataset

class OpenNPFileProcessor(PreProcessor):
    "`PreProcessor` that opens the filenames and read the texts."
    def process_one(self,item):
        return np.load(item) if isinstance(item, Path) else item
    
class LMNPDataBunch(DataBunch):
    "Create a `TextDataBunch` suitable for training a language model."
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', no_check:bool=False, bs=64, val_bs:int=None, 
               num_workers:int=0, device:torch.device=None, collate_fn:Callable=data_collate, 
               dl_tfms:Optional[Collection[Callable]]=None, bptt:int=70, backwards:bool=False) -> DataBunch:
        "Create a `TextDataBunch` in `path` from the `datasets` for language modelling."
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        datasets = [LMNPPreloader(ds, shuffle=(i==0), bs=(bs if i==0 else val_bs), bptt=bptt, backwards=backwards) 
                    for i,ds in enumerate(datasets)]
        val_bs = bs
        dls = [DataLoader(d, b, shuffle=False) for d,b in zip(datasets, (bs,val_bs,val_bs,val_bs)) if d is not None]
        return cls(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)
    
    @classmethod    
    def from_ids(cls, path:PathOrStr, train_ids:Collection[Collection[int]], valid_ids:Collection[Collection[int]],
                 test_ids:Collection[Collection[int]]=None, train_lbls:Collection[Union[int,float]]=None,
                 valid_lbls:Collection[Union[int,float]]=None, classes:Collection[Any]=None,
                 processor:PreProcessor=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from ids, labels and a `vocab`. `kwargs` are passed to the dataloader creation."
        src = ItemLists(path, LMNPItemList(train_ids, path=path, processor=[]),
                        LMNPItemList(valid_ids, path=path, processor=[]))
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
        train_ids,train_lbls = np.load(cache_path/f'train_ids.npy'), np.load(cache_path/f'train_lbl.npy')
        valid_ids,valid_lbls = np.load(cache_path/f'valid_ids.npy'), np.load(cache_path/f'valid_lbl.npy')
        test_ids = np.load(cache_path/f'test_ids.npy') if os.path.isfile(cache_path/f'test_ids.npy') else None
        classes = loadtxt_str(cache_path/'classes.txt') if os.path.isfile(cache_path/'classes.txt') else None
        return cls.from_ids(path, train_ids, valid_ids, test_ids, train_lbls, valid_lbls, classes, processor, **kwargs)

class LMNPItemList(ItemList):
    _bunch = LMNPDataBunch
    def get(self, i)->Any:
        tfmd = self.items[i] + VAL_OFFSET
#         if NO_INST: return tfmd[:,:3]
        return tfmd
    
    def reconstruct(self, t:Tensor):
#         if TO_SEQ: return npenc2seq((t-1))
        return t-VAL_OFFSET
    
    def __getitem__(self,idxs:int)->Any:
        idxs = try_int(idxs)
        if isinstance(idxs, Integral): return self.get(idxs)
        else: return self.new(self.items[idxs], xtra=index_row(self.xtra, idxs))
        

class LMNPPreloader(Callback):
    "Transforms the tokens in `dataset` to a stream of contiguous batches for language modelling."
    
    class CircularIndex():
        "Handles shuffle, direction of indexing, wraps around to head tail in the ragged array as needed"
        def __init__(self, length:int, forward:bool): self.idx, self.forward = np.arange(length), forward
        def __getitem__(self, i): 
            return self.idx[ i%len(self.idx) if self.forward else len(self.idx)-1-i%len(self.idx)]
        def __len__(self) -> int: return len(self.idx)
        def shuffle(self): np.random.shuffle(self.idx)

    def __init__(self, dataset:LabelList, lengths:Collection[int]=None, bs:int=32, bptt:int=70, backwards:bool=False, 
                 shuffle:bool=False):
        self.dataset,self.bs,self.bptt,self.shuffle,self.backwards,self.lengths = dataset,bs,bptt,shuffle,backwards,lengths
        self.totalToks,self.ite_len,self.idx = int(0),None,None

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
        self.idx   = LMNPPreloader.CircularIndex(len(self.dataset.x), not self.backwards)
        self.batch = np.zeros((self.bs, self.bptt+Y_OFFSET, self.dataset.x[0].shape[1]), dtype=np.int64)
        self.batch_x, self.batch_y = self.batch[:,0:self.bptt], self.batch[:,Y_OFFSET:self.bptt+Y_OFFSET] 
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

    def fill_row(self, forward, items, idx, row, ro, ri, overlap,lengths):
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
