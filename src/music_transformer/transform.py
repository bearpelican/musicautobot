from __future__ import annotations
from ..numpy_encode import *
import numpy as np
from enum import Enum
import torch
from ..vocab import *
from functools import partial

SEQType = Enum('SEQType', 'Mask, Sentence, Melody, Chords, Empty')

class MusicItem():
    def __init__(self, data, vocab, stream=None, position=None):
        self.data = data
        self.vocab = vocab
        self._stream = stream
        self._position = position
    def __repr__(self): return self.vocab.textify(self.data)
    def __len__(self): return len(self.data)

    @classmethod
    def from_file(cls, midi_file, vocab):
        return MusicItem(midi2idxenc(midi_file, vocab), vocab, file2stream(midi_file))
        
    @classmethod
    def from_npenc(cls, npenc, vocab):
        return MusicItem(npenc2idxenc(npenc, vocab), vocab)
    
    @classmethod
    def from_idx(cls, item, vocab):
        idx,pos = item
        return MusicItem(idx, vocab=vocab, position=pos)
    def to_idx(self): return self.data, self.position
    
    @classmethod
    def empty(cls, vocab, seq_type=SEQType.Sentence):
        return MusicItem(seq_prefix(seq_type, vocab), vocab)

    @property
    def stream(self, bpm=120):
        self._stream = idxenc2stream(self.data, self.vocab, bpm=bpm) if self._stream is None else self._stream
        return self._stream

    def to_tensor(self, device=None):
        return to_tensor(self.data, device)
    
    @property
    def position(self): 
        self._position = position_enc(self.data, self.vocab) if self._position is None else self._position
        return self._position
    
    def get_pos_tensor(self, device=None): return to_tensor(self.position, device)

    def to_npenc(self):
        return idxenc2npenc(self.data, self.vocab)

    def show(self, format:str=None):
        return self.stream.show(format)
    def play(self): self.stream.show('midi')
        
    @property
    def new(self):
        return partial(type(self), vocab=self.vocab)

    def trim_to_beat(self, beat):
        return self.new(trim_tfm(self.data, self.vocab, beat))
    
    def transpose(self, interval):
        return self.new(tfm_transpose(self.data, interval, self.vocab), position=self._position)
    
    def append(self, item:MusicItem):
        return self.new(np.concatenate((self.data, item.data), axis=0))
    
    def mask_notes(self):
        masked_data = mask_input(self.data, self.vocab.note_range, self.vocab.mask_idx)
        return self.new(masked_data, position=self.position)
    
    def mask_duration(self, keep_position_enc=True):
        masked_data = mask_input(self.data, self.vocab.dur_range, self.vocab.mask_idx)
        if keep_position_enc: return self.new(masked_data, position=self.position)
        return self.new(masked_data)
    
    def pad_to(self, bptt):
        data = pad_seq(self.data, bptt, self.vocab.pad_idx)
        pos = pad_seq(self.position, bptt, 0)
        return self.new(data, stream=self._stream, position=pos)
        
def pad_seq(seq, bptt, value):
    pad_len = max(bptt-seq.shape[0], 0)
    return np.pad(seq, (0, pad_len), 'constant', constant_values=value)[:bptt]

# def partenc2seq2seq(part_np, part_type, vocab, add_eos=True):
#     part_meta = np.array([vocab.stoi[part_type], vocab.pad_idx])
#     s2s_out = npenc2idxenc(part_np, vocab=vocab, start_seq=part_meta)
#     if add_eos: s2s_out = np.pad(s2s_out, (0,1), 'constant', constant_values=vocab.stoi[EOS])
#     return s2s_out

# def s2s_combine2chordarr(np1, np2, vocab):
#     if len(np1.shape) == 1: np1 = idxenc2npenc(np1, vocab)
#     if len(np2.shape) == 1: np2 = idxenc2npenc(np2, vocab)
#     p1 = npenc2chordarr(np1)
#     p2 = npenc2chordarr(np2)
#     return chordarr_combine_parts(p1, p2)
     
# def stream2melody_chord(stream, vocab):
#     chordarr = stream2chordarr(stream) # 2.
#     _,num_parts,_ = chordarr.shape
#     if num_parts != 2: 
#         raise ValueError('Could not extract melody and chords from midi file. Please make sure file contains exactly 2 tracks')
    
#         p1, p2 = [part_enc(chordarr, i) for i in range(num_parts)]
#         sorted(avg_pitch
#         p1, p2 = (p1, p2) if avg_pitch(p1) > avg_pitch(p2) else (p2, p1)
#     mpart = partenc2seq2seq(p1, part_type=MSEQ, vocab=vocab)
#     cpart = partenc2seq2seq(p2, part_type=CSEQ, vocab=vocab)
#     return mpart, cpart

def to_tensor(t, device=None):
    t = t if isinstance(t, torch.Tensor) else torch.tensor(t)
    if device is None and torch.cuda.is_available(): t = t.cuda()
    else: t.to(device)
    return t
            
def trim_tfm(idxenc, vocab, to_beat=None, sample_freq=SAMPLE_FREQ):
    if to_beat is None: return idxenc
    pos = position_enc(idxenc, vocab)
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
def npenc2idxenc(t, vocab, seq_type=SEQType.Sentence, add_eos=False):
    "Transforms numpy array from 2 column (note, duration) matrix to a single column"
    "[[n1, d1], [n2, d2], ...] -> [n1, d1, n2, d2]"
    if isinstance(t, (list, tuple)) and len(t) == 2: 
        return [npenc2idxenc(x, vocab, start_seq) for x in t]
    t = t.copy()
    
    t[:, 0] = t[:, 0] + vocab.note_range[0]
    t[:, 1] = t[:, 1] + vocab.dur_range[0]
    
    prefix = seq_prefix(seq_type, vocab)
    suffix = np.array([vocab.stoi[EOS]]) if add_eos else np.empty(0, dtype=int)
    return np.concatenate([prefix, t.reshape(-1), suffix])

def seq_prefix(seq_type, vocab):
    if seq_type == SEQType.Empty: return np.empty(0, dtype=int)
    start_token = vocab.bos_idx
    if seq_type == SEQType.Chords: start_token = vocab.stoi[CSEQ]
    if seq_type == SEQType.Melody: start_token = vocab.stoi[MSEQ]
    return np.array([start_token, vocab.pad_idx])

def idxenc2npenc(t, vocab, validate=True):
    if validate: t = to_valid_idxenc(t, vocab.npenc_range)
    t = t.copy().reshape(-1, 2)
    if t.shape[0] == 0: return
        
    t[:, 0] = t[:, 0] - vocab.note_range[0]
    t[:, 1] = t[:, 1] - vocab.dur_range[0]
    
    if validate: return to_valid_npenc(t)
    return t

def to_valid_idxenc(t, valid_range):
    r = valid_range
    t = t[np.where((t >= r[0]) & (t < r[1]))]
    if t.shape[-1] % 2 == 1: t = t[..., :-1]
    return t

def to_valid_npenc(t):
    is_note = (t[:, 0] < VALTSEP) | (t[:, 0] >= NOTE_SIZE)
    invalid_note_idx = is_note.argmax()
    invalid_dur_idx = (t[:, 1] < 0).argmax()

    invalid_idx = max(invalid_dur_idx, invalid_note_idx)
    if invalid_idx > 0: 
        if invalid_note_idx > 0 and invalid_dur_idx > 0: invalid_idx = min(invalid_dur_idx, invalid_note_idx)
        print('Non midi note detected. Only returning valid portion. Index, seed', invalid_idx, t.shape)
        return t[:invalid_idx]
    return t

def position_enc(idxenc, vocab):
    "Calculates positional beat encoding."
    sep_idxs = (idxenc == vocab.sep_idx).nonzero()[0]
    sep_idxs = sep_idxs[sep_idxs+2 < idxenc.shape[0]] # remove any indexes right before out of bounds (sep_idx+2)
    dur_vals = idxenc[sep_idxs+1]
    dur_vals[dur_vals == vocab.mask_idx] = vocab.dur_range[0] # make sure masked durations are 0
    dur_vals -= vocab.dur_range[0]
    
    posenc = np.zeros_like(idxenc)
    posenc[sep_idxs+2] = dur_vals
    return posenc.cumsum()

def position_tfm(idxenc, vocab):
    posenc = position_enc(idxenc, vocab)
    return np.stack([idxenc, posenc], axis=1)

def tfm_transpose(x, value, vocab):
    x = x.copy()
    x[(x >= vocab.note_range[0]) & (x < vocab.note_range[1])] += value
    return x

# Utility for predictions
def mask_input(xb, mask_range, replacement_idx):
    xb = xb.copy()
    xb[(xb >= mask_range[0]) & (xb < mask_range[1])] = replacement_idx
    return xb

# def rand_transpose_tfm(t, vocab, rand_range=(0,24), p=0.5):
#     if np.random.rand() < p:
#         transpose_value = np.random.randint(*rand_range)-rand_range[1]//2
#         if isinstance(t, (list, tuple)) and len(t) == 2: 
#             return [tfm_transpose(x, transpose_value, vocab) for x in t]
#         return tfm_transpose(t, transpose_value, vocab)
#     return t