from ..numpy_encode import *
import numpy as np
import torch

# MLMType = Enum('MLMType', 'Mask, NextWord, M2C, C2M')

class MusicItem():
    def __init__(self, data, vocab, stream=None, position=None):
        self.data = data
        self.vocab = vocab
        self._stream = stream
    def __repr__(self): return vocab.textify(self.data)
    def __len__(self): return len(self.data)

    @classmethod
    def from_file(cls, midi_file, vocab):
        return MusicItem(midi2idxenc(midi_file, vocab), vocab, file2stream(midi_file))
        
    @classmethod
    def from_npenc(cls, npenc, vocab):
        return MusicItem(npenc2idxenc(npenc, vocab), vocab)
    
#     @classmethod
#     def empty(cls, vocab, seq_type:str=None):
#         return MusicItem(np.array([vocab.bos_idx, vocab.pad_idx])

    @property
    def stream(self, bpm=120):
        if self._stream is None: 
            self._stream = idxenc2stream(self.data, self.vocab, bpm=bpm)
        return self._stream

    def to_tensor(self, device=None):
        return to_tensor(self.data, device)
    
    @property
    def position(self): return neg_position_enc(self.data, self.vocab)
    def get_pos_tensor(self, device=None): return to_tensor(self.get_pos(), device)

    def to_npenc(self):
        return idxenc2npenc(self.data)

    def show(self, format:str=None):
        return self.stream.show(format)
    def show_score(self): self.stream.show()
    def show_midi(self): self.stream.show('midi')
    def play(self): self.stream.show('midi')

    def trim_to_beat(self, beat):
        return MusicItem(trim_tfm(self.data, self.vocab, beat), self.vocab)
    
    def transpose(self, interval):
        return MusicItem(tfm_transpose(self.data, interval, self.vocab), self.vocab)
        

def to_tensor(t, device=None):
    t = t if isinstance(t, torch.Tensor) else torch.tensor(t)
    if device is None and torch.cuda.is_available(): t = t.cuda()
    else: t.to(device)
    return t
            
def trim_tfm(idxenc, vocab, to_beat=None, sample_freq=SAMPLE_FREQ):
    if to_beat is None: return idxenc
    pos = -neg_position_enc(idxenc, vocab)
    cutoff = np.searchsorted(pos, to_beat * sample_freq)
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

def position_tfm(idxenc, vocab):
    posenc = neg_position_enc(idxenc, vocab) # using negative values so we don't interfere with indexes
    return np.stack([idxenc, posenc], axis=1)

def tfm_transpose(x, value, vocab):
    x = x.copy()
    x[(x >= vocab.note_range[0]) & (x < vocab.note_range[1])] += value
    return x

def rand_transpose_tfm(t, vocab, rand_range=(0,24), p=0.5):
    if np.random.rand() < p:
        transpose_value = np.random.randint(*rand_range)-rand_range[1]//2
        if isinstance(t, (list, tuple)) and len(t) == 2: 
            return [tfm_transpose(x, transpose_value, vocab) for x in t]
        return tfm_transpose(t, transpose_value, vocab)
    return t
