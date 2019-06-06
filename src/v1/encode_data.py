"Encoding music21 streams -> numpy array -> text"

import re
from pathlib import Path
import music21
import numpy as np
from .midi_data import file2stream
from fastai.text.data import BOS
import scipy.sparse
from collections import defaultdict
from math import ceil

TIMESIG = '4/4' # default time signature
PIANO_RANGE = (21, 108)
VALTSEP = -1 # separator value for numpy encoding
VALTSTART = -1 # numpy value for TSTART
VALTCONT = -2 # numpy value for TCONT


NPRE = 'n' # note value encoding prefix
OPRE = 'o' # octave encoding prefix
IPRE = 'i' # instrument encoding prefix
TPRE = 't' # note type encoding prefix - negative means duration encoded

# Encoding process
# 1. midi -> music21.Stream
# 2. Stream -> numpy chord array (timestep X instrument X noterange)
# 3. numpy array -> List[Timestep][NoteEnc]
# 4. NoteEnc -> string

# Decoding process
# 1. string -> NoteEnc
# 2. NoteEnc -> numpy array
# 3. numpy array -> music21.Stream
# 4. Stream -> midi

class NoteEnc():
    # dur = note start/continue, note = midi value, inst = instrument
    def __init__(self, note, dur, inst=None):
        assert(dur != 0)
        self.note,self.dur,self.inst = note,int(dur),inst
        if self.inst is not None: self.inst = str(self.inst)
            
    @property
    def pitch(self):
        return music21.pitch.Pitch(self.note)
    
    # continuous format is -1 for note strike, -2 for note continued
    def continuous_repr(self, short=True, instrument=False):
        dur = self.dur if self.dur == VALTCONT else VALTSTART
        tname = f'{TPRE}{dur}' # ts=note start, tc=note continue
        if short: 
            nname = NPRE + self.pitch.nameWithOctave
            return [nname, tname]
        nname = NPRE + self.pitch.name
        oname = OPRE + str(self.pitch.octave)
        if instrument:
            iname = IPRE+self.inst
            return [nname,oname,tname,iname]
        return [nname,oname,tname]
    
    # duration format is tX for note duration, Return nothing if continued note
    def duration_repr(self, short=True, instrument=False):
        if self.dur == VALTCONT: return []
        tname = f'{TPRE}{self.dur}'
        if short: 
            nname = NPRE + self.pitch.nameWithOctave
            return [nname, tname]
        nname = NPRE + self.pitch.name
        oname = OPRE + str(self.pitch.octave)
        if instrument:
            iname = IPRE+self.inst
            return [nname,oname,tname,iname]
        return [nname,oname,tname]
    
#     def joined_repr(self):
#         # returns something like 'nG:o2:ts:i1'
#         return NOTE_SEP.join(self.long_comp())
    
    def __repr__(self):
        kname = self.pitch.nameWithOctave
        tname = f'{TPRE}{self.dur}' # ts=note start, tc=note continue
        return kname+tname
    
    def ival(self): # instrument number value
        if self.inst is None: return 0
        return int(self.inst)
    
    def m21_note(self):
        return music21.note.Note(self.note)
        
    @classmethod
    def parse_arr(self, arr):
        kv = {s[0]:s[1:] for s in arr if s}
        if NPRE not in kv: return None
        note = kv[NPRE]
        if OPRE in kv: note += kv[OPRE]
        dur = int(kv[TPRE])
        assert(re.fullmatch(RENOTE, note))
        return NoteEnc(note=note, dur=dur, inst=kv.get(IPRE))

##### ENCODING ######

def midi2seq(midi_file):
    "Converts midi file to string representation for language model"
    stream = file2stream(midi_file) # 1.
    s_arr = stream2chordarr(stream) # 2.
    return chordarr2seq(s_arr) # 3.

# 2.
def stream2chordarr(s, note_range=128, sample_freq=4, max_dur=None):
    "Converts music21.Stream to 1-hot numpy array"
    # assuming 4/4 time
    # note x instrument x pitch
    # FYI: midi middle C value=60
    
    # (AS) TODO: need to order by instruments most played and filter out percussion or include the channel
    maxTimeStep = int(s.flat.highestTime * sample_freq)+1
    score_arr = np.zeros((maxTimeStep, len(s.parts), note_range))

    def note_data(pitch, note):
        return (pitch.midi, round(note.offset*sample_freq), round(note.duration.quarterLength*sample_freq))

    for idx,part in enumerate(s.parts):
        notes=[]
        for elem in part.flat:
            if isinstance(elem, music21.note.Note):
                notes.append(note_data(elem.pitch, elem))
            if isinstance(elem, music21.chord.Chord):
                for p in elem.pitches:
                    notes.append(note_data(p, elem))
                
        # sort notes by offset (1), duration (2) so that hits are not overwritten and longer notes have priority
        notes_sorted = sorted(notes, key=lambda x: (x[1], x[2])) 
        for n in notes_sorted:
            if n is None: continue
            pitch,offset,duration = n
            if max_dur is not None and duration > max_dur: duration = max_dur
            score_arr[offset, idx, pitch] = duration
            score_arr[offset+1:offset+duration, idx, pitch] = VALTCONT      # Continue holding note
    return score_arr

def compress_chordarr(chordarr):
    return shorten_chordarr_rests(trim_chordarr_rests(chordarr))

def trim_chordarr_rests(arr, max_rests=16):
    start_idx = 0
    for idx,t in enumerate(arr):
        if t.sum() != 0: break
        start_idx = idx+1
        
    end_idx = 0
    for idx,t in enumerate(reversed(arr)):
        if t.sum() != 0: break
        end_idx = idx+1
    start_idx = start_idx - start_idx % max_rests
    end_idx = end_idx - end_idx % max_rests
#     if start_idx > 0 or end_idx > 0: print('Trimming rests. Start, end:', start_idx, len(arr)-end_idx, end_idx)
    return arr[start_idx:(len(arr)-end_idx)]

def shorten_chordarr_rests(arr, max_rests=32):
    rest_count = 0
    result = []
    for timestep in arr:
        if timestep.sum() == 0: 
            rest_count += 1
        else:
            if rest_count > max_rests+4:
                old_count = rest_count
                rest_count = rest_count % 4 + max_rests
#                 print(f'Compressing rests: {old_count} -> {rest_count}')
            for i in range(rest_count): result.append(np.zeros(timestep.shape))
            rest_count = 0
            result.append(timestep)
    for i in range(rest_count): result.append(np.zeros(timestep.shape))
    return np.array(result)

# 3a.
def chordarr2seq(score_arr):
    # note x instrument x pitch
    return [timestep2seq(t) for t in score_arr]

# 3b.
def timestep2seq(timestep):
    # int x pitch
    notes = [NoteEnc(n,timestep[i,n],i) for i,n in zip(*timestep.nonzero())]
    sorted_keys = sorted(notes, key=lambda x: x.pitch, reverse=True)
    return sorted_keys

##### DECODING #####

# 2.
def seq2chordarr(seq, note_range=128): # 128 = default midi range
    num_parts = max([n.ival() for t in seq for n in t]) + 1
    score_arr = np.zeros((len(seq), num_parts, note_range))
    for idx,ts in enumerate(seq):
        for note in ts:
            score_arr[idx,note.ival(),note.pitch.midi] = note.dur
    return score_arr

# 3.
def chordarr2stream(arr, sample_freq=4, bpm=120):
    duration = music21.duration.Duration(1. / sample_freq)
    stream = music21.stream.Stream()
    stream.append(music21.meter.TimeSignature(TIMESIG))
    stream.append(music21.tempo.MetronomeMark(number=bpm))
    stream.append(music21.key.KeySignature(0))
    for inst in range(arr.shape[1]):
        p = partarr2stream(arr[:,inst,:], duration, stream=music21.stream.Part())
        stream.append(p)
    stream = stream.transpose(0)
    return stream

# 3b.
def partarr2stream(part, duration, stream=None):
    "convert instrument part to music21 chords"
    if stream is None: stream = music21.stream.Stream()
    stream.append(music21.instrument.Piano())
    if np.any(part > 0): part_append_duration_notes(part, duration, stream) # notes already have duration calculated
    else: part_append_continuous_notes(part, duration, stream) # notes are either start or continued 

    return stream

# 3b
def part_append_continuous_notes(part, duration, stream):
    starts = part == VALTSTART
    durations = calc_note_durations(part)
    for tidx,t in enumerate(starts):
        note_idxs = np.where(t < 0)[0]
        if len(note_idxs) == 0: continue
        notes = []
        for nidx in note_idxs:
            note = music21.note.Note(nidx)
            tnext = durations[tidx+1,nidx] if tidx+1 < len(part) else 0
            note.duration = music21.duration.Duration((tnext+1)*duration.quarterLength)
            notes.append(note)
        for g in group_notes_by_duration(notes):
            chord = music21.chord.Chord(g)
            stream.insert(tidx*duration.quarterLength, chord)
    return stream
        
# 3c.
def calc_note_durations(part):
    "calculate midi note durations from TCONT notes"
    cnotes = (part == VALTCONT).astype(int)
    for i in reversed(range(cnotes.shape[0]-1)):
        cnotes[i] += cnotes[i+1]*cnotes[i]
    return cnotes

# 3alt.
def part_append_duration_notes(part, duration, stream=None):
    "convert instrument part to music21 chords"
    for tidx,t in enumerate(part):
        note_idxs = np.where(t > 0)[0] # filter out any negative values (continuous mode)
        if len(note_idxs) == 0: continue
        notes = []
        for nidx in note_idxs:
            note = music21.note.Note(nidx)
            note.duration = music21.duration.Duration(part[tidx,nidx]*duration.quarterLength)
            notes.append(note)
        for g in group_notes_by_duration(notes):
            chord = music21.chord.Chord(g)
            stream.insert(tidx*duration.quarterLength, chord)
    return stream

from itertools import groupby
#  combining notes with different durations into a single chord may overwrite conflicting durations. Example: aylictal/still-waters-run-deep
def group_notes_by_duration(notes):
    "separate notes into chord groups"
    keyfunc = lambda n: n.duration.quarterLength
    notes = sorted(notes, key=keyfunc)
    return [list(g) for k,g in groupby(notes, keyfunc)]

# saving
def save_chordarr(out_file, chordarr):
    sparse_matrix = scipy.sparse.csc_matrix(chordarr.reshape(chordarr.shape[0], -1))
    scipy.sparse.save_npz(out_file, sparse_matrix)
    
def load_chordarr(file):
    sparse_matrix = scipy.sparse.load_npz(file)
    np_arr = np.array(sparse_matrix.todense())
    return np_arr.reshape((np_arr.shape[0], -1, 127))



# npenc functions

def npenc2stream(arr, bpm=120):
    "Converts numpy encoding to music21 stream"
    seq = npenc2seq(np.array(arr))
    chordarr = seq2chordarr(seq)
    return chordarr2stream(chordarr, bpm=bpm)

def midi2npenc(midi_file, midi_source=None):
    "Converts midi file to numpy encoding for language model"
    stream = file2stream(midi_file) # 1.
    s_arr = stream2chordarr(stream) # 2.
    seq = chordarr2seq(s_arr) # 3.
    return seq2npenc(seq)

# 4.
def npenc_func(n, num_comps=2):
    if num_comps == 2: return [n.pitch.midi, n.dur]
    raise ValueError('Unhandled number of components')

def seq2npenc(seq):
    "Note function returns a list of note components for separation"
    result = []
    wait_count = 1
    for idx,timestep in enumerate(seq):
        flat_time = [npenc_func(n) for n in timestep if n.pitch.octave and n.dur > 0]
        if len(flat_time) == 0:
            wait_count += 1
        else:
            # pitch, octave, duration, instrument
            result.append([VALTSEP, wait_count])
            result.extend(flat_time)
            wait_count = 1
    return np.array(result, dtype=int)

def npdec_func(step):
    n,d,o,i = pad_array(step, 0, final_length=4)
    return NoteEnc(n+o*12,d,i)
    
def npenc2seq(npenc, dec_func=npdec_func):
    seq = []
    tstep = []
    npenc = npenc.copy()
    for x in npenc:
        n,d = x[:2]
        if n < VALTSEP: continue # special tokens
        if n == VALTSEP: 
            if len(tstep) > 0: seq.append(tstep) # add notes if they exists
            tstep = []
            for i in range(1, d): seq.append([])
        else:
            if d == 0: 
                print('Note with 0 duration. continuing')
                continue
            tstep.append(dec_func(x))
    if len(tstep) > 0: seq.append(tstep)
    return seq

def pad_array(arr, fill_value, final_length):
    if isinstance(arr, np.ndarray): arr = arr.tolist()
    padding = [fill_value] * max(0, final_length - len(arr))
    return arr+padding
