import music21
from src import roman_to_symbol
from src import to_pianoroll
from collections import defaultdict
from midi_data import keyc_offset
import numpy as np
from collections import Counter, defaultdict
from src.roman_to_symbol import *
from src.tab_parser import *

    
# ENC_IDXS = [0,1,2,3,4,5,6,7,8,9,10]
# iB,iM,iN,iNO,iND,iC1,iC2,iC3,iC4,iCD,iCI = ENC_IDXS
# BIDX_ALL = [iB,iM]
ENC_IDXS = [0,1,2,3,4,5,6,7,8]
iN,iNO,iND,iC1,iC2,iC3,iC4,iCD,iCI = ENC_IDXS
BIDX_ALL = []
CIDXS = [iC1,iC2,iC3,iC4]
CIDX_ALL = CIDXS + [iCD,iCI]
NIDX_ALL = [iN,iNO,iND]
DUR_IDXS = [iND,iCD]

# sustain=True, sep_octave=True, sus_idx=2, hit_idx=1
config_opts = dict(note_octave=5, chord_octave=3,
                   sort_invert=False, # sort pitches by inversion order
                   continuous=True,
                   throw_err=False,
                   sus_idx=1, hit_idx=0,
                   bim=4, key=0, bpm=120, freq=4, 
                   bos_idx=-1, pad_idx=-3, none_idx=-2, enc_offset=3)

# none_idx is a rest - checked by cross_entropy, pad_idx is not countd in loss
class EncConfig(object):
    def __init__(self, d): self.__dict__ = d
        
enc_config = EncConfig(config_opts)

MODE_TO_KEYOFFSET = {
    '1': 0,
    '2': 2,
    '3': 4,
    '4': 5,
    '5': 7,
    '6': 9,
    '7': 11
}

PITCH_TO_SD = {
    0: '1',
    1: '1#',
    2: '2',
    3: '2#',
    4: '3',
    5: '4',
    6: '4#',
    7: '5',
    8: '5#',
    9: '6',
    10:'6#',
    11:'7',
}

SD_TO_PITCH = {v:k for k,v in PITCH_TO_SD.items()}


from dataclasses import dataclass
import dataclasses
from typing import Dict, Any, AnyStr, List, Sequence, TypeVar, Tuple, Optional, Union

def parse(cls, d):
    cls_keys = cls.__dataclass_fields__.keys()
    kwargs = {key:d[key] for key in cls_keys}
    return cls(**kwargs)

@dataclass
class Base:
    @classmethod
    def from_dict(cls, d):
        cls_keys = cls.__dataclass_fields__.keys()
        kwargs = {key:d[key] for key in cls_keys}
        return cls(**kwargs)
    
    @classmethod
    def parse(cls, d):
        return cls.from_dict(d)

@dataclass
class HMetadata(Base):
    title:str
    BPM:str='120'
    beats_in_measure:str='4'
    key:str='C'
    mode:str='1'

@dataclass
class HBeat(Base):
    def __init__(self, abs:float, duration:float, measure:float, rel:float, **kwargs):
        self.abs = float(abs)
        self.duration = float(duration)
        self.measure = float(measure) if measure else None
        self.rel = float(rel) if rel else None
    @classmethod
    def parse_note(cls, d):
        return cls(abs=d['start_beat_abs'], 
                   duration=d['note_length'], 
                   measure=d['start_measure'],
                   rel=d['start_beat'])
    @classmethod
    def parse_chord(cls, d):
        return cls(abs=d['start_beat_abs'], 
                   duration=d['chord_duration'], 
                   measure=d['start_measure'],
                   rel=d['start_beat'])
    
    def end_time(self):
        return self.duration + self.abs
    
    def __repr__(self):
        return f'abs={self.abs}:dur={self.duration}'
        
@dataclass
class HPitch(Base):
    def __init__(self, pitch:int, octave:int, **kwargs):
        if enc_config.throw_err: assert(pitch >= 0)
        if enc_config.throw_err and octave: assert(octave >= 0)
        self.pitch = pitch
        self.octave = octave
    
    def sd(self):
        return PITCH_TO_SD[self.pitch]
    
    def abs(self, base_octave:int=0):
        if self.octave is None: return self.pitch
        return self.pitch + 12 * (self.octave+base_octave)
    
    def __repr__(self):
#         return f'sd={self.sd()}:oct={self.octave}'
        return f'{self.sd()}'
    
    @classmethod
    def parse_abs_pitch(cls, abs_pitch, base_octave=0):
        abs_pitch = int(abs_pitch)
        pitch = abs_pitch%12
        octave = abs_pitch//12+base_octave
        return cls(pitch=pitch, octave=octave)

@dataclass
class HNote(Base):
    def __init__(self, beat:HBeat, pitch:HPitch, **kwargs):
        self.beat = beat
        self.pitch = pitch
        
    def to_m21(self)->music21.note.Note:
        p = self.pitch.abs()
#         p = self.pitch.abs(base_octave=enc_config.note_octave)
        n = music21.note.Note(p, quarterLength=self.beat.duration)
        return n, self.beat.abs
    
    def end_time(self):
        return self.beat.end_time()
    
    def __repr__(self):
        return f'{self.pitch}'
    
    @classmethod
    def parse(cls, d, mode, key_offset):
        parsed = roman_to_symbol.hnote_parser(d, mode, key_offset)
        pitch = HPitch.parse_abs_pitch(parsed['pitch'], base_octave=enc_config.note_octave) # pitch can be negative so offset octave
        beat = HBeat.parse_note(d)
        return cls(beat=beat, pitch=pitch) 

def remove_embelishment(d):
    d = d.copy()
    d['sus'] = None
    d['emb'] = None
    return d

def remove_9_11_chords(d):
    d = d.copy()
    if isinstance(d.get('fb'), str) and int(d['fb'][0]) > 7: d['fb'] = '9'
    return d
    
def shift_octave(parsed):
    lowest = min(parsed['composition'])
    if lowest >= 12:
        parsed = parsed.copy()
        parsed = roman_to_symbol.chord_key_shifting(parsed, -(lowest//12 * 12))
        # reset chord name
        new_s = roman_to_symbol.chord_to_string(parsed)
        parsed['symbol'] = new_s
    return parsed

@dataclass
class HChord(Base):
    def __init__(self, beat:HBeat, pitches:List[HPitch], inv:int,
                 composition:List[int]=None, symbol:str=None, quality:str=None, metadata=None, **kwargs):
        self.beat = beat
        self.pitches = pitches
        min_octave = min([p.octave for p in pitches])
        if min_octave > 0:
            for p in self.pitches: p.octave = p.octave - min_octave
        self.inv = inv
        self.composition, self.symbol, self.quality = composition, symbol, quality
        self.metadata = None
        
    def end_time(self):
        return self.beat.end_time()
        
    def to_m21(self)->music21.chord.Chord:
        notes = [p.abs(base_octave=enc_config.chord_octave) for p in self.pitches]
        c = music21.chord.Chord(notes, quarterLength=self.beat.duration)
        c.volume = music21.volume.Volume(velocity=50)
        return c, float(self.beat.abs)
    
    def __repr__(self):
        return f'[{self.pitches}])'# + ' Comp:' + str(self.composition)

    @classmethod
    def parse(cls, d, mode, key_offset, reset_to_base=True, remove_emb=False, remove_large=True):
        if remove_emb: d = remove_embelishment(d)
        if remove_large: remove_9_11_chords(d)
        if d.get('emb') == 'add4': d['emb'] = None
        parsed = roman_to_symbol.hchord_parser(d, mode, key_offset)
        
        # After offset, let's reset the chord to be the lowest possible offset on new scale
        if reset_to_base: parsed = shift_octave(parsed)
        parsed['composition'] = parsed['composition'].astype(int).tolist()
        
        beat = HBeat.parse_chord(d)
        if enc_config.sort_invert: pitches = [HPitch.parse_abs_pitch(p) for p in sorted(parsed['composition'])] # first pitch = base note
        pitches = [HPitch.parse_abs_pitch(p) for p in parsed['composition']] # first pitch = base note
        
        return cls(beat=beat, pitches=pitches, metadata=parsed, **parsed)

def default_stream(cls=music21.stream.Score, ts='4/4', bpm=120, ks=0):
    # (AS) TODO: use config ts or metadata
    s = cls()
    s.append(music21.instrument.Piano())
    s.append(music21.meter.TimeSignature(ts))
    s.append(music21.tempo.MetronomeMark(number=bpm))
#     s.append(music21.key.KeySignature(ks))
    s.append(music21.key.Key('C'))
    return s

@dataclass
class HPart(Base):
    notes: List[HNote]
    chords: List[HChord]
    num_measure: float
        
    @classmethod
    def parse(cls, d, metadata):
        mode = metadata['mode'] or '1'
        key_offset = MODE_TO_KEYOFFSET.get(mode, 0)
        ns = [HNote.parse(n, mode, key_offset) for n in d.get('notes', []) if n['scale_degree'] != 'rest']
        cs = [HChord.parse(c, mode, key_offset) for c in d.get('chords', []) if c['sd'] != 'rest']
        ns = sorted(ns, key=lambda n: n.end_time())
        cs = sorted(cs, key=lambda c: c.end_time())
        return cls(notes=ns, chords=cs, num_measure=d['num_measure'])
    
    def __repr__(self):
        chords = '[Chords]:\n' + '\n'.join([str(c) for c in self.chords])
        notes = '[Notes]:\n' + '\n'.join([str(n) for n in self.notes])
        return chords + '\n\n' + notes
    
    def duration(self):
        return self.num_measure * enc_config.bim
#         c_last = self.chords[-1].end_time() if self.chords else 0
#         n_last = self.notes[-1].end_time() if self.notes else 0
#         return max(c_last, n_last)
    
    # Note: first chord not played - https://github.com/rism-ch/verovio/issues/995
    def to_m21(self)->music21.stream.Stream:
        mc = music21.stream.Part()
        mn = music21.stream.Part()
        
        cm21 = [c.to_m21() for c in self.chords]
        for c,d in cm21: mc.insert(d,c)
            
        nm21 = [n.to_m21() for n in self.notes]
        for n,d in nm21: mn.insert(d,n)
        return mn, mc
        
    def min_pitch(self):
        return min([n.pitch for n in self.notes])

@dataclass
class HSong(Base):
    metadata: HMetadata
    parts: List[HPart]
    
    @classmethod
    def parse(cls, metadata, segments):
        m = HMetadata.parse(metadata)
        ps = [HPart.parse(s, metadata) for s in segments]
        return cls(metadata=m, parts=ps)
    
    def duration(self):
        return sum(p.duration() for p in self.parts)
    
    
    def __repr__(self):
        parts = ''
        for idx,p in enumerate(self.parts):
            parts += f'Part[{idx}]:\n' + str(p) + '\n\n'
        return parts + '\n\n' + str(self.metadata)
    
    def to_stream(self):
        s = default_stream()
        pc = music21.stream.Part()
        pn = music21.stream.Part()
        
        offset = 0.0
        for p in self.parts:
            mn, mc = p.to_m21()
            pn.insert(offset, mn)
            pc.insert(offset, mc)
            offset = p.duration() # fixes mismatched chord/note lengths
            
        s.insert(0.0, pn)
        s.insert(0.0, pc)

#         s.flat.makeNotation(inPlace=True)
        s = s.transpose(0) # hack to get accidentals right. Above method does not work
        # music21 stream
        return s
    
    
    
    
def enc_beat(beat):
    start = round(beat.abs * enc_config.freq, 6) # float conversion error
    duration = round(beat.duration * enc_config.freq, 6)
    end = start+duration

    assert(start.is_integer() and duration.is_integer())
    return start, end, duration
    
def enc_part(part):
#     '(pitch x octave x sustain) x (chord_sd x base x suspend, sustain) x (bar_position x beat_pos)'
    '(pitch x octave x dur) x (c1,c2,c3,c4,dur) x (bar_position x beat_pos)'
    max_len = int(part.duration()*enc_config.freq)
    sequence = np.full((max_len, len(ENC_IDXS)), fill_value=enc_config.pad_idx, dtype=int)
    sequence[:,DUR_IDXS] = enc_config.none_idx
    
    if BIDX_ALL:
        # beat_pos
        bim = enc_config.bim # beats_in_measure = 4
        sequence[:, iB] = np.tile(np.arange(bim).repeat(enc_config.freq), int(part.duration())//bim+1)[:sequence.shape[0]]
        
        # bar_pos
        sequence[:, iM] = np.arange(part.duration()//bim+1).repeat(enc_config.freq*bim)[:sequence.shape[0]]
    
    continuous = enc_config.continuous
    for n in part.notes:
        start, end, duration = enc_beat(n.beat)
#         end = end if continuous else start
#         duration = enc_config.sus_idx if continuous else duration
        if continuous:
            sequence[int(start):int(end),NIDX_ALL] = n.pitch.pitch, n.pitch.octave, enc_config.sus_idx
            sequence[int(start),iND] = enc_config.hit_idx
        else:
            sequence[int(start),NIDX_ALL] = n.pitch.pitch, n.pitch.octave, duration
        
    for c in part.chords:
        start, end, duration = enc_beat(c.beat)
        if continuous:
            for idx, p in zip(CIDXS,c.pitches):
                sequence[int(start):int(end),idx] = p.pitch
            sequence[int(start):int(end),iCD] = enc_config.sus_idx
            sequence[int(start):int(end),iCI] = c.inv
            sequence[int(start),iCD] = enc_config.hit_idx
        else:
            for idx, p in zip(CIDXS,c.pitches):
                sequence[int(start),idx] = p.pitch
            sequence[int(start),iCI] = c.inv
            sequence[int(start),iCD] = duration
    return sequence

def enc_song(song, step_size=1):
    eps = [enc_part(p) for p in song.parts]
    cat = np.concatenate(eps)
    if step_size > 1: cat = cat.reshape(-1, step_size, cat.shape[-1])
        
    bos_row = np.full(((1,) + cat.shape[1:]), fill_value=enc_config.pad_idx)
    bos_row[...,DUR_IDXS] = enc_config.bos_idx
    enc_all = np.concatenate((bos_row, cat))
    
    enc_off = enc_all + enc_config.enc_offset
    assert((enc_off >= 0).all())
    return enc_off




#### DECODE

def dec_beat(duration, beat_abs, ts):
    if (ts.shape[0] == len(ENC_IDXS) and BIDX_ALL):
        rel,measure = ts[BIDX_ALL]+1 # +1 as hook is offset by 1
    else:
        rel,measure = None, None 
    return HBeat(abs=beat_abs/enc_config.freq, duration=duration/enc_config.freq, 
                 rel=rel, measure=measure)
    
def dec_note(ts, beat_abs):
    p,o,dur = ts[NIDX_ALL]
    
    beat = dec_beat(dur, beat_abs, ts)
    pitch = HPitch(octave=o, pitch=p)
    note = HNote(beat=beat, pitch=pitch)
    return note

def is_padding(val): 
    return val < 0
#     return val in [enc_config.pad_idx, enc_config.bos_idx, enc_config.none_idx]

def dec_chord(ts, beat_abs):
    
    beat = dec_beat(ts[iCD], beat_abs, ts)
    
    octave = 0
    pitches = []
    pvals = ts[CIDXS]
    inv = ts[iCI] if len(ts) > iCI else 0
    for idx,p in enumerate(pvals):
        if is_padding(p): continue
        if (idx > 0) and (p < pvals[idx-1]): 
            octave += 1
        pitches.append(HPitch(pitch=int(p), octave=octave))
    for p in pitches[:inv]: p.octave += 1 # invert pitches
    chord = HChord(pitches=pitches, beat=beat, inv=inv)
    return chord

def dec_part_durations(part):
    "calculate midi note durations from TCONT notes"
    plen = part.shape[0]
    part = part.copy()
    for i in range(plen):
        if part[i][iND]==enc_config.sus_idx:
            print('Broken encoding. Sustained note with out a hit. Removing all')
            part[i][NIDX_ALL] = enc_config.pad_idx
        if part[i][iND]==enc_config.hit_idx:
            duration = 1
            for j in range(i+1,plen):
                is_sus = part[j][iND]==enc_config.sus_idx
                same_note = (part[i][[iN,iNO]]==part[j][[iN,iNO]]).all()
                if is_sus and same_note:
                    duration+=1
                    part[j][NIDX_ALL] = enc_config.pad_idx
                else:
                    break
            part[i][iND] = duration
    
    for i in range(plen):
        if part[i][iCD]==enc_config.sus_idx:
            print('Broken encoding. Sustained chord with out a hit')
            part[i][iCD] = enc_config.pad_idx
            part[i][CIDXS] = enc_config.pad_idx
        if part[i][iCD]==enc_config.hit_idx:
            duration = 1
            for j in range(i+1,plen):
                is_sus = part[j][iCD]==enc_config.sus_idx
                same_chord = (part[i][CIDXS] == part[j][CIDXS]).all()
                if is_sus and same_chord:
                    duration+=1
                    part[j][CIDXS] = enc_config.pad_idx
                    part[j][iCD] = enc_config.pad_idx
                else:
                    break
            part[i][iCD] = duration
    return part

def dec_part(part):
    '(pitch x octave x dur) x (c1,c2,c3,c4,dur) x (bar_position x beat_pos)'
    if enc_config.continuous: part = dec_part_durations(part)
    notes = [dec_note(ts,idx) for idx,ts in enumerate(part) if not is_padding(ts[DUR_IDXS[0]])]
    chords = [dec_chord(ts,idx) for idx,ts in enumerate(part) if not is_padding(ts[DUR_IDXS[1]])]
    
    return HPart(notes=notes, chords=chords, num_measure=part.shape[0]/enc_config.bim)

def dec_arr(arr):
    arr = arr-enc_config.enc_offset
    if (arr[0] == enc_config.bos_idx).any():
        arr = arr[1:]
    arr = arr.reshape(-1, arr.shape[-1]) # reshape after bos - since timesteps could be blocks of 16
    dp = dec_part(arr)
    return HSong(parts=[dp], metadata=HMetadata('decoded'))

def parse_file(file_path):
    content = load_data(file_path)
    try: 
        root = xml_parser(content)
        metadata, version = get_metadata(root)
        segments, num_measures = get_lead_sheet(root, version)
    except Exception as e: 
        print('XML parse exception:', e)
        return None
    
    song = HSong.parse(metadata, segments)
    return song
