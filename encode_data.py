import re
from pathlib import Path
import music21
import numpy as np
from midi_data import file2stream
from fastai.text.data import BOS
import scipy.sparse
from collections import defaultdict

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


# Functions inspired by:
# https://github.com/mcleavey/musical-neural-net/blob/master/data/midi-to-encoding.py
# https://github.com/tensorflow/magenta/tree/master/magenta/models/polyphony_rnn


TSEP = '||' # beat/timestep end encoding

MSTART = '|s|' # measure start encoding
MEND = '|e|' # measure end encoding
NPRE = 'n' # note value encoding prefix
OPRE = 'o' # octave encoding prefix
IPRE = 'i' # instrument encoding prefix

TPRE = 't' # note type encoding prefix - negative means duration encoded
VALTSTART = -1 # numpy value for TSTART
VALTCONT = -2 # numpy value for TCONT

NOTE_SEP = ':' # separator for note components. No longer using

TIMESIG = '4/4' # default time signature


RENOTE = re.compile('[A-Z][#-b]?\d')
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

# master encoder
def midi2str(midi_file, note_func, continuous=False):
    "Converts midi file to string representation for language model"
#     stream = file2stream(midi_file) # 1.
#     s_arr = stream2chordarr(stream, encode_duration) # 2.
#     seq = chordarr2seq(s_arr) # 3.
    seq = midi2seq(midi_file)
    return seq2str(seq, note_func=note_func, continuous=continuous)

# 2.
def stream2chordarr(s, note_range=127, sample_freq=4, max_dur=None):
    "Converts music21.Stream to 1-hot numpy array"
    # assuming 4/4 time
    # note x instrument x pitch
    # FYI: midi middle C value=60
    maxTimeStep = int(s.flat.duration.quarterLength * sample_freq)+1
    
    # (AS) TODO: need to order by instruments most played and filter out percussion or include the channel
    inst2idx = defaultdict(int, {inst.id:idx for idx,inst in enumerate(s.flat.getInstruments())})
    score_arr = np.zeros((maxTimeStep, len(inst2idx), note_range))

    notes=[]
    noteFilter=music21.stream.filters.ClassFilter('Note')
    chordFilter=music21.stream.filters.ClassFilter('Chord')
    
    def note_data(pitch, note):
        inst_id = note.activeSite.getInstrument().id
        iidx = inst2idx[inst_id]
        return (pitch.midi, round(note.offset*sample_freq), round(note.duration.quarterLength*sample_freq), iidx)

    for n in s.recurse().addFilter(noteFilter):
        notes.append(note_data(n.pitch, n))
        
    for c in s.recurse().addFilter(chordFilter):
        pitchesInChord=c.pitches
        for p in pitchesInChord:
            notes.append(note_data(p, c))
    notes_sorted = sorted(notes, key=lambda x:[1])
    for n in notes_sorted:
        if n is None: continue
        pitch,offset,duration,inst = n
        if max_dur is not None and duration > max_dur: duration = max_dur
        score_arr[offset, inst, pitch] = duration
        score_arr[offset+1:offset+duration, inst, pitch] = VALTCONT      # Continue holding note
    return score_arr

def compress_chordarr(chordarr):
    return _shorten_chordarr_rests(_trim_chordarr_rests(chordarr))

def _trim_chordarr_rests(arr, max_rests=16):
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

def _shorten_chordarr_rests(arr, max_rests=32):
    rest_count = 0
    result = []
    for timestep in arr:
        if timestep.sum() == 0: 
            rest_count += 1
        else:
            if rest_count > max_rests+4:
                old_count = rest_count
                rest_count = rest_count % 4 + max_rests
                print(f'Compressing rests: {old_count} -> {rest_count}')
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
    sorted_keys = sorted(notes, key=lambda x: x.pitch)
    return sorted_keys

# 4.
def seq2str(seq, note_func, continuous):
    if continuous: return seq2str_continuous(seq, note_func)
    else: return seq2str_duration(seq, note_func)
    
def seq2str_continuous(seq, note_func=None, separate_measures=False):
    "Note function returns a list of note components for spearation"
    result = []
    for idx,timestep in enumerate(seq):
        if separate_measures and idx and idx%4 == 0:
            result.append(MEND)
            if idx < len(seq)-1: result.append(MSTART)
        flat_time = [i for n in timestep for i in note_func(n)]
        result.extend(flat_time)
        result.append(TSEP)
    return ' '.join(result)

# 4.alt
def seq2str_duration(seq, note_func=None):
    "Note function returns a list of note components for spearation"
    result = []
    wait_count = 0
    for idx,timestep in enumerate(seq):
        flat_time = [i for n in timestep for i in note_func(n)]
        if len(flat_time) == 0:
            wait_count += 1
        else:
            result.append(TSEP)
            result.append(f'{TPRE}{wait_count}')
            result.extend(flat_time)
            wait_count = 0
    return ' '.join(result)
    
##### DECODING #####
# 
def str2stream(seq_str):
    seq = str2seq(seq_str)
    arr = seq2chordarr(seq)
    return chordarr2stream(arr)

# 1.
def str2seq(seq_str):
    seq_str = seq_str.replace(f'{MSTART} ', '').replace(f'{MEND} ', '').replace(f'{BOS} ', '')
    timesteps = seq_str.split(f'{TSEP} ')
    
    seq = []
    for t in timesteps:
        tsplit = t.split(' ')
        if tsplit and TPRE in tsplit[0]:
            duration = int(tsplit[0][1:])
            for i in range(duration): seq.append([])
            tsplit = tsplit[1:]
        seq.append(steps2seq(tsplit))
    return seq

# 1b.
def steps2seq(tarr):
    idxs = [idx for idx,s in enumerate(tarr) if s and s[0] == NPRE]
    notes = []
    for a in np.split(tarr, idxs):
        try: 
            note = NoteEnc.parse_arr(a) 
            if note: notes.append(note)
        except Exception as e:
            print(e)
    return notes

# 2.
def seq2chordarr(seq, note_range=127):
    num_instruments = max([n.ival() for t in seq for n in t]) + 1
    score_arr = np.zeros((len(seq), num_instruments, note_range))
    for idx,ts in enumerate(seq):
        for note in ts:
            score_arr[idx,note.ival(),note.pitch.midi] = note.dur
    return score_arr

# 3.
def chordarr2stream(arr, sample_freq=4):
    duration = music21.duration.Duration(1. / sample_freq)
    stream = music21.stream.Stream()
    stream.append(music21.meter.TimeSignature(TIMESIG))
    stream.append(music21.tempo.MetronomeMark(number=120))
    stream.append(music21.key.KeySignature(0))
    for inst in range(arr.shape[1]):
        p = partarr2stream(arr[:,inst,:], duration, stream=music21.stream.Part())
        stream.append(p)
    return stream

# 3b.
def partarr2stream(part, duration, stream=None):
    "convert instrument part to music21 chords"
    if stream is None: stream = music21.stream.Stream()
    stream.append(music21.instrument.Piano())
    if np.any(part > 0): part_append_duration_notes(part, duration, stream) # notes already have duration calcualted
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
        chord = music21.chord.Chord(notes)
        stream.insert(tidx*duration.quarterLength, chord)
        
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
        chord = music21.chord.Chord(notes)
        stream.insert(tidx*duration.quarterLength, chord)
    return stream

# saving
def save_chordarr(out_file, chordarr):
    sparse_matrix = scipy.sparse.csc_matrix(chordarr.reshape(chordarr.shape[0], -1))
    scipy.sparse.save_npz(out_file, sparse_matrix)
    
def load_chordarr(file):
    sparse_matrix = scipy.sparse.load_npz(file)
    np_arr = np.array(sparse_matrix.todense())
    return np_arr.reshape((np_arr.shape[0], -1, 127))



# npenc functions

VALTSEP = 12
VALTBOS = 13
PADDING_IDX = -1

# 4.
def npenc_func(n, inst=False):
    if inst: return [n.pitch.pitchClass, n.pitch.octave, n.dur, n.inst]
    return [n.pitch.pitchClass, n.pitch.octave, n.dur]

def seq2npenc(seq, enc_func=npenc_func):
    "Note function returns a list of note components for spearation"
    result = [[VALTBOS, PADDING_IDX, PADDING_IDX, PADDING_IDX]]
    wait_count = 0
    for idx,timestep in enumerate(seq):
        flat_time = [enc_func(n) for n in timestep if n.pitch.octave and n.dur > 0]
        if len(flat_time) == 0:
            wait_count += 1
        else:
            # pitch, octave, duration, instrument
            result.append([VALTSEP, PADDING_IDX, wait_count, PADDING_IDX])
            result.extend(flat_time)
            wait_count = 0
    return np.array(result, dtype=int)

def npdec_func(step):
    n,o,d = step[:3]
    i = step[3] if len(step) == 4 else None
    return NoteEnc(n+((o+1)*12),d,i)
    
def npenc2seq(npenc, dec_func=npdec_func):
    seq = []
    tstep = []
    for x in npenc:
        n = x[0]
        if n == VALTBOS: continue
        if n == VALTSEP: 
            if len(tstep) > 0: seq.append(tstep)
            tstep = []
            for i in range(d): seq.append([])
        else:
            tstep.append(dec_func(x))
    if len(tstep) > 0: seq.append(tstep)
    return seq