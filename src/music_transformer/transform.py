
class MusicItem(ItemBase):
    def __init__(self, item, vocab):
        self.data = item
        self.vocab = vocab
        self.stream = None
    def __repr__(self): return self.data[:10]

    @classmethod
    def from_file(cls, midi_file, vocab):
        return MusicItem(midi2idxenc(midi_file, vocab), vocab)
        
    @classmethod
    def from_npenc(cls, npenc, vocab):
        return MusicItem(npenc2idxenc(npenc, vocab), vocab)

    def to_stream(self, bpm=120):
        if self.stream is None: 
            self.stream = idxenc2stream(self.data, bpm=bpm)
        return self.stream

    def to_tensor(self, device=None):
        t = torch.tensor(self.data)
        if device is None and torch.cuda.is_available(): t = t.cuda()
        else: t.to(device)
        return t

    def get_pos(self):
        return neg_position_enc(self.data, self.vocab)

    def to_npenc(self):
        return idxenc2npenc(self.data)

    def show_score(self):
        self.to_stream().show()

    def play_file(self):
        self.to_stream().show('midi')

    def trim_to_beat(self, beat):
        self.data = self.seed_tfm(self.data, beat)

def trim_tfm(idxenc, to_beat=None, sample_freq=SAMPLE_FREQ):
    if to_beat is None: return idxenc
    pos = -neg_position_enc(idxenc)
    cutoff = np.searchsorted(pos, to_beat * sample_freq) + 1
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
def npenc2idxenc(t, vocab, start_seq=None):
    "Transforms numpy array from 2 column (note, duration) matrix to a single column"
    "[[n1, d1], [n2, d2], ...] -> [n1, d1, n2, d2]"
    if isinstance(t, (list, tuple)) and len(t) == 2: 
        return [npenc2idxenc(x, vocab, start_seq) for x in t]
    t = t.copy()
    
    t[:, 0] = t[:, 0] + vocab.note_range[0]
    t[:, 1] = t[:, 1] + vocab.dur_range[0]
    if start_seq is None: start_seq = np.array([vocab.bos_idx, vocab.pad_idx])
    return np.concatenate([start_seq, t.reshape(-1)])

def idxenc2npenc(t, vocab, validate=True):
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

def neg_position_enc(idxenc, vocab):
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
