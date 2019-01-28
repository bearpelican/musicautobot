import re
from pathlib import Path
import music21
import numpy as np
from midi_data import file2stream

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

TPRE = 't' # note type encoding prefix
VALTSTART = 1 # numpy value for TSTART
VALTCONT = 2 # numpy value for TCONT
TSTART = f'{TPRE}{VALTSTART}' # note start/strike encoding
TCONT = f'{TPRE}{VALTCONT}' # note continue encoding

NOTE_SEP = ':' # separator for note components. No longer using

TIMESIG = '4/4' # default time signature


RENOTE = re.compile('[A-Z][#-b]?\d')
class NoteEnc():
    # tie = note start/continue, note = midi value, inst = instrument
    def __init__(self, note, tie, inst=None):
        assert(tie > 0)
        self.note,self.tie,self.inst = note,int(tie),str(inst)
        self.pitch = music21.pitch.Pitch(self.note)
        
    def long_comp(self):
        nname = NPRE + self.pitch.name
        oname = OPRE + str(self.pitch.octave)
        tname = TSTART if self.tie == VALTSTART else TCONT # ts=note start, tc=note continue
        iname = IPRE+self.inst
        return [nname,oname,tname,iname]
        
    def long_repr(self):
        # returns something like 'nG:o2:ts:i1'
        return NOTE_SEP.join(self.long_comp())
    
    def __repr__(self):
        kname = self.pitch.nameWithOctave
        tname = TSTART if self.tie == VALTSTART else TCONT # ts=note start, tc=note continue
        return kname+tname
    
    def ival(self): # instrument number value
        return int(self.inst or 0)
    
    def m21_note(self):
        return music21.note.Note(self.note)
        
    @classmethod
    def parse_arr(self, arr):
        kv = {s[0]:s[1:] for s in arr if s}
        if NPRE not in kv: return None
        note = kv[NPRE]
        if OPRE in kv: note += kv[OPRE]
        tie = int(kv[TPRE])
        assert(re.fullmatch(RENOTE, note))
        return NoteEnc(note=note, tie=tie, inst=kv.get(IPRE))

##### ENCODING ######
# master encoder
def midi2str(midi_file):
    "Converts midi file to string representation for language model"
    stream = file2stream(midi_file) # 1.
    s_arr = stream2chordarr(stream) # 2.
    seq = chordarr2seq(s_arr) # 3.
    return seq2str(seq) # 4.

# 2.
def stream2chordarr(s, note_range=127, sample_freq=4):
    "Converts music21.Stream to 1-hot numpy array"
    # assuming 4/4 time
    # note x instrument x pitch
    # FYI: midi middle C value=60
    maxTimeStep = int(s.duration.quarterLength * sample_freq)+1
    
    # (AS) TODO: need to order by instruments most played and filter out percussion or include the channel
    inst2idx = {inst.id:idx for idx,inst in enumerate(s.flat.getInstruments())}
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

    for n in notes:
        if n is None: continue
        pitch,offset,duration,inst = n
        score_arr[offset, inst, pitch] = VALTSTART                 # Strike note
        score_arr[offset+1:offset+duration, inst, pitch] = VALTCONT      # Continue holding note
    return score_arr

# 3a.
def chordarr2seq(score_arr):
    # note x instrument x pitch
    return [timestep2seq(t) for t in score_arr]

# 3b.
def timestep2seq(timestep):
    # int x pitch
    notes = [NoteEnc(n,timestep[i,n],i) for i,n in zip(*timestep.nonzero())]
    sorted_keys = [n for n in sorted(notes, key=lambda x: x.pitch)]
    return sorted_keys

# 4.
def seq2str(seq):
    result = []
    for idx,timestep in enumerate(seq):
        if idx and idx%4 == 0:
            result.append(MEND)
        if idx and idx < len(seq)-1:
            result.append(MSTART)
        flat_time = [i for n in timestep for i in n.long_comp()]
        result.extend(flat_time)
        result.append(TSEP)
    return ' '.join(result)
        
        
    
    
##### DECODING #####
# 
def str2stream(seq_str):
    seq = str2seq(seq_str)
    arr = seq2numpy(seq)
    return chordarr2stream(arr)

# 1.
def str2seq(seq_str):
    timesteps = seq_str.split(TSEP)
    return [steps2chordarr(t.split(' ')) for t in timesteps]

# 1b.
def steps2chordarr(tarr):
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
def seq2numpy(seq, note_range=127):
    num_instruments = max([n.ival() for t in seq for n in t]) + 1
    score_arr = np.zeros((len(seq), num_instruments, note_range))
    for idx,ts in enumerate(seq):
        for note in ts:
            score_arr[idx,note.ival(),note.pitch.midi] = note.tie
    return score_arr

# 3.
def chordarr2stream(arr, sample_freq=4):
    duration = music21.duration.Duration(1. / sample_freq)
    stream = music21.stream.Stream()
    for inst in range(arr.shape[1]):
        p = partarr2stream(arr[:,inst,:], duration, stream=music21.stream.Part())
        stream.append(p)
    return stream

# 3b.
def partarr2stream(part, duration, stream=None, inst=None):
    "convert instrument part to music21 chords"
    if stream is None: stream = music21.stream.Stream()
    stream.append(music21.instrument.Piano())
    stream.append(music21.meter.TimeSignature(TIMESIG))
    stream.append(music21.key.KeySignature(0))
    starts = part == 1
    durations = calc_note_durations(part)
    for tidx,t in enumerate(starts):
        note_idxs = t.nonzero()[0]
        if len(note_idxs) == 0: continue
        notes = []
        for nidx in note_idxs:
            note = music21.note.Note(nidx)
            tnext = durations[tidx+1,nidx] if tidx+1 < len(part) else 0
            note.duration = music21.duration.Duration((tnext+1)*duration.quarterLength)
            notes.append(note)
        chord = music21.chord.Chord(notes, offset=tidx*duration.quarterLength)
        stream.append(chord)
    return stream
    
# 3c.
def calc_note_durations(part):
    "calculate midi note durations from TCONT notes"
    cnotes = (part == VALTCONT).astype(int)
    for i in reversed(range(cnotes.shape[0]-1)):
        cnotes[i] += cnotes[i+1]*cnotes[i]
    return cnotes
