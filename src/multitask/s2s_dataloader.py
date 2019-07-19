
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
        m = position_tfm(partenc2seq2seq(m, MSEQ))
        c = position_tfm(partenc2seq2seq(c, CSEQ))
        return np.array((m, c))
    
    def process(self, ds):
        if self.vocab is None: self.vocab = MusicVocab.create()
        ds.vocab = self.vocab
        ds.items = [self.process_one(item) for item in ds.items]
        
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
        self.transpose_range = transpose_range
        self.transpose_tfm = partial(rand_transpose_tfm, note_range=vocab.note_range, rand_range=transpose_range) if transpose_range is not None else None
        
    def __getitem__(self, k:int):
        item,_ = self.dataset[k]
        x,y = item
        if self.transpose_tfm is not None:
            x,y = self.transpose_tfm([x,y])
        # WARNING: we are padding position encodings too. However, pos is negative, so should be fine
        return pad_seq(x, self.bptt+1), pad_seq(y, self.bptt+1) # offset bptt for decoder shift
    
    def __len__(self):
        return len(self.dataset)
        
def pad_seq(seq, bptt, pad_idx=vocab.pad_idx):
    pad_len = max(bptt-seq.shape[0], 0)
    return np.pad(seq, [(0, pad_len),(0,0)], 'constant', constant_values=pad_idx)[:bptt]
    
def partenc2seq2seq(part_np, part_type, vocab=vocab, add_eos=True):
    part_meta = np.array([vocab.stoi[part_type], vocab.pad_idx])
    s2s_out = to_single_stream(part_np, start_seq=part_meta)
    if add_eos: s2s_out = np.pad(s2s_out, (0,1), 'constant', constant_values=vocab.stoi[EOS])
#     s2s_out = position_tfm(s2s_out)
    return s2s_out


def s2s_combine2chordarr(np1, np2):
    if len(np1.shape) == 1: np1 = to_double_stream(np1)
    if len(np2.shape) == 1: np2 = to_double_stream(np2)
    p1 = npenc2chordarr(np1)
    p2 = npenc2chordarr(np2)
    max_ts = max(p1.shape[0], p2.shape[0])
    p1w = ((0,max_ts-p1.shape[0]),(0,0),(0,0))
    p1_pad = np.pad(p1, p1w, 'constant')
    p2w = ((0,max_ts-p2.shape[0]),(0,0),(0,0))
    p2_pad = np.pad(p2, p2w, 'constant')
    chordarr_comb = np.concatenate((p1_pad, p2_pad), axis=1)
    return chordarr_comb

        