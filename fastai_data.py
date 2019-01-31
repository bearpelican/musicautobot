from fastai.text import *

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
        freq = sum(e.map(count_tokens, partition_by_cores(tokens, n_cpus)), Counter())
        
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
#         if self.vocab is None: self.vocab = vocab_create_parallel(ds.items, self.max_vocab, self.min_freq)
        if self.vocab is None: self.vocab = Vocab.create(ds.items, self.max_vocab, self.min_freq)
        ds.vocab = self.vocab
        print("Numericalizing")
        n_cpus = num_cpus()
        parts = partition_by_cores(ds.items, n_cpus)
        vocabs = [ds.vocab.stoi.copy() for i in range(len(parts))]
        with ProcessPoolExecutor(n_cpus) as e:
            items = sum(e.map(numericalize, zip(vocabs, parts)), [])
        gc.collect()
        ds.items = array(items)