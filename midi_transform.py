from enum import Enum
import music21

class Track(Enum):
    INFO = 0
    MELODY = 1
    PIANO = 2 # discrete instruments - keyboard, woodwinds
    STRING = 3 # continuous instruments with pitch bend: violin, trombone, synths
    PERC = 4
    UNDEF = 5
    
type2inst = {
    # use print_music21_instruments() to see supported types
    Track.MELODY: 79, # Ocarina
    Track.STRING: 24, # Guitar
    Track.PIANO: 0 # Piano
}

INFO_TYPES = set(['TIME_SIGNATURE', 'KEY_SIGNATURE', 'SET_TEMPO'])

def transform_midi(midi_file, out_file, cutoff=6, offset=None):
    music_file = compress_midi_file(midi_file, cutoff=cutoff) # remove non note tracks and standardize instruments
    music_file = transpose_midi_file(music_file, offset) # transpose to keyc
    s_out = music21.midi.translate.midiFileToStream(music_file) # create music21 stream
    s_comb = music21.instrument.partitionByInstrument(s_out) # combine same track instruments to single part
    return s_comb.write('midi', fp=out_file)

def compress_midi_file(fp, cutoff=6, unsup_types=set([Track.UNDEF, Track.PERC])):
    music_file = file2mf(fp)
    supported_tracks = []
    for idx,t in enumerate(music_file.tracks):
        track_type = get_track_type(t,idx,fp)
        if len(supported_tracks) >= cutoff: continue
        if track_type in unsup_types: continue
        if track_type in type2inst:
            change_track_instrument(t, type2inst[track_type])
        supported_tracks.append(t)
    music_file.tracks = supported_tracks
    return music_file

def transpose_midi_file(mf, offset):
    for t in mf.tracks:
        if 10 in t.getChannels(): continue # skip percussion
        for e in t.events:
            if e.pitch is None: continue
            e.pitch += offset
    return mf

def get_track_type(t, idx, fp=''):
    if is_info_track(t, idx): return Track.INFO
    if not t.hasNotes(): return Track.UNDEF
    if is_melody(t, fp): return Track.MELODY
    if is_percussion(t): return Track.PERC
    if is_string(t): return Track.STRING
    return Track.PIANO

def change_track_instrument(t, value):
    for idx,e in enumerate(t.events):
        if e.type == 'PROGRAM_CHANGE':
            e.data = value



def is_channel(t, c_val):
    return any([c == c_val for c in t.getChannels()])

def has_event_type(t, etypes):
    if isinstance(etypes, list): etypes = set(etypes)
    elif isinstance(etypes, set): pass
    else: etypes = set([etypes])
    for e in t.events:
        if e.type in etypes:
            return True
    return False

def is_info_track(t, idx):
    is_ch0 = idx == 0
    is_info = has_event_type(t, INFO_TYPES)
    if is_info and t.hasNotes(): raise Exception('Error: found track with notes and track info')
    if is_ch0 ^ is_info: raise Exception('Error: info channel is not channel 0')
    return is_info

def is_melody(t, fp=''):
    if not t.hasNotes(): return False
    # special case for hooktheory
    if 'hooktheory' in str(fp):
        if is_channel(t, 1): return True
        return False
    if has_event_type(t, ['LYRIC']): return True # lyrics associated with vocals
    if is_channel(t, 1): return True # (AS) WARNING - assuming first track is always the melody track

def is_percussion(t):
    return is_channel(t, 10)

def is_string(t):
    if not t.hasNotes(): return False
    if has_event_type(t, ['PITCH_BEND']): return True
    return False

def is_piano(t, fp):
    if not t.hasNotes(): return False
    if is_melody(t, fp): return False
    if is_string(t): return False
    if is_percussion(t): return False
    return True

def print_music21_instruments():
    for i in range(200):
        try: print(i, music21.instrument.instrumentFromMidiProgram(i))
        except: pass