"Transform functions for raw midi files"
from enum import Enum
import music21

PIANO_TYPES = list(range(24)) + list(range(80, 96)) # Piano, Synths
PLUCK_TYPES = list(range(24, 40)) + list(range(104, 112)) # Guitar, Bass, Ethnic
BRIGHT_TYPES = list(range(40, 56)) + list(range(56, 80))

PIANO_RANGE = (21, 109) # https://en.wikipedia.org/wiki/Scientific_pitch_notation

class Track(Enum):
    PIANO = 0 # discrete instruments - keyboard, woodwinds
    PLUCK = 1 # continuous instruments with pitch bend: violin, trombone, synths
    BRIGHT = 2
    PERC = 3
    UNDEF = 4
    
type2inst = {
    # use print_music21_instruments() to see supported types
    Track.PIANO: 0, # Piano
    Track.PLUCK: 24, # Guitar
    Track.BRIGHT: 40, # Violin
    Track.PERC: 114, # Steel Drum
}

# INFO_TYPES = set(['TIME_SIGNATURE', 'KEY_SIGNATURE'])
INFO_TYPES = set(['TIME_SIGNATURE', 'KEY_SIGNATURE', 'SET_TEMPO'])

def file2mf(fp):
    mf = music21.midi.MidiFile()
    if isinstance(fp, bytes):
        mf.readstr(fp)
    else:
        mf.open(fp)
        mf.read()
        mf.close()
    return mf

def mf2stream(mf): return music21.midi.translate.midiFileToStream(mf)

def is_empty_midi(fp):
    if fp is None: return False
    mf = file2mf(fp)
    return not any([t.hasNotes() for t in mf.tracks])

def num_piano_tracks(fp):
    music_file = file2mf(fp)
    note_tracks = [t for t in music_file.tracks if t.hasNotes() and get_track_type(t) == Track.PIANO]
    return len(note_tracks)

def is_channel(t, c_val):
    return any([c == c_val for c in t.getChannels()])

def track_sort(t): # sort by 1. variation of pitch, 2. number of notes
    return len(unique_track_notes(t)), len(t.events)

def is_piano_note(pitch):
    return (pitch >= PIANO_RANGE[0]) and (pitch < PIANO_RANGE[1])

def unique_track_notes(t):
    return { e.pitch for e in t.events if e.pitch is not None }

def compress_midi_file(fp, cutoff=6, min_variation=3, supported_types=set([Track.PIANO, Track.PLUCK, Track.BRIGHT])):
    music_file = file2mf(fp)
    
    info_tracks = [t for t in music_file.tracks if not t.hasNotes()]
    note_tracks = [t for t in music_file.tracks if t.hasNotes()]
    
    if len(note_tracks) > cutoff:
        note_tracks = sorted(note_tracks, key=track_sort, reverse=True)
        
    supported_tracks = []
    for idx,t in enumerate(note_tracks):
        if len(supported_tracks) >= cutoff: break
        track_type = get_track_type(t)
        if track_type not in supported_types: continue
        pitch_set = unique_track_notes(t)
        if (len(pitch_set) < min_variation): continue # must have more than x unique notes
        if not all(map(is_piano_note, pitch_set)): continue # must not contain midi notes outside of piano range
#         if track_type == Track.UNDEF: print('Could not designate track:', fp, t)
        change_track_instrument(t, type2inst[track_type])
        supported_tracks.append(t)
    if not supported_tracks: return None
    music_file.tracks = info_tracks + supported_tracks
    return music_file

def get_track_type(t):
    if is_channel(t, 10): return Track.PERC
    i = get_track_instrument(t)
    if i in PIANO_TYPES: return Track.PIANO
    if i in PLUCK_TYPES: return Track.PLUCK
    if i in BRIGHT_TYPES: return Track.BRIGHT
    return Track.UNDEF

def get_track_instrument(t):
    for idx,e in enumerate(t.events):
        if e.type == 'PROGRAM_CHANGE': return e.data
    return None

def change_track_instrument(t, value):
    for idx,e in enumerate(t.events):
        if e.type == 'PROGRAM_CHANGE': e.data = value

def print_music21_instruments():
    for i in range(200):
        try: print(i, music21.instrument.instrumentFromMidiProgram(i))
        except: pass