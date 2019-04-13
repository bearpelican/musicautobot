# Functions for manipulating midi data/files

from pathlib import Path
import music21
import json

### Saving
def save_json(obj, path): json.dump(obj, open(path, 'w'))
    
def load_json(path): 
    if not Path(path).exists(): return None
    return json.load(open(path, 'r'))

### Display
if Path('/usr/bin/lilypond').exists():
    from IPython.display import Image
    import os
    # Might get an error server cannot use x11 display
    # https://github.com/ContinuumIO/anaconda-issues/issues/1806
    os.environ['QT_QPA_PLATFORM']='offscreen'
    os.environ['QT_QPA_FONTDIR']='/usr/share/fonts'
    music21.environment.set("lilypondPath", "/usr/bin/lilypond")

def display_score(stream):
    "Show music21.stream with lilypond. Alternatively use stream.show()"
    return Image(filename=str(stream.write('lily.png')))

### Playback
try:
    from midi2audio import FluidSynth
    from IPython.display import Audio
    import uuid
    sound_font = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
    fluidsynth_player = FluidSynth(sound_font)
except Exception as e:
    print('Failed to load FluidSynth. Must install if you want to convert to wav files.')

def midi2wav(midi_file, wav_file=None):
    if wav_file is None: wav_file = f'/tmp/{str(uuid.uuid4())}.wav'
    fluidsynth_player.midi_to_audio(str(midi_file), wav_file)
    return wav_file

def play_midi(midi_file, wav_file=None):
    wav_file = midi2wav(midi_file, wav_file)
    return Audio(str(wav_file))

def print_stream_durations(stream, threshold=None):
    for m in stream.flat:
        if isinstance(m, (music21.chord.Chord, music21.note.Note)):
            if threshold is None or m.duration.quarterLength > threshold:
                print(m.duration.quarterLength, m.offset, m)
            
### Midi transpose

def transpose_raw_midi(midi_file, out_file, offset):
    """
    Transpose midi key. Skips channel 10 percussion
    Alter midifile directly as music21 streams can't parse complex files/instruments
    """
    mf = music21.midi.MidiFile()
    mf.open(midi_file)
    mf.read()
    mf.close()
    new_mf = _transpose_raw_midi(mf, offset)
    new_mf.open(out_file, attrib='wb')
    new_mf.write()
    new_mf.close()
    
def _transpose_raw_midi(mf, offset):
    for t in mf.tracks:
        if 10 in t.getChannels(): continue # skip percussion
        for e in t.events:
            if e.pitch is None: continue
            e.pitch += offset
    return mf

def file2stream(fp, use_parser=True):
    if isinstance(fp, music21.midi.MidiFile): return music21.midi.translate.midiFileToStream(fp)
    if use_parser: return music21.converter.parse(fp)
    mf = file2mf(fp)
    return music21.midi.translate.midiFileToStream(mf)

def file2mf(fp):
    mf = music21.midi.MidiFile()
    mf.open(fp)
    mf.read()
    mf.close()
    return mf

def transpose_midi2c(file, score=None, out_file=None, overwrite=False, conversion_type='RAW', halfsteps=None):
    if out_file.exists() and not overwrite: 
#         print('Transposed file exists. Skipping: ', out_file)
        return out_file
    
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    needs_score = (conversion_type == 'music21') or (halfsteps is None)
    if score is None and needs_score:
        score = music21.converter.parse(file)
    if halfsteps is None:
        key = score.analyze('key')
        halfsteps = keyc_offset(key.tonic.name, key.mode)

    if conversion_type=='RAW':
        transpose_raw_midi(file, out_file, offset=halfsteps)
    elif conversion_type=='music21':
        transposed_score = score.transpose(halfSteps)
        transposed_score.write('midi',out_file)
    return out_file



### Midi transpose keys

# converting everything into the key of C major or A minor
# https://gist.github.com/aldous-rey/68c6c43450517aa47474
major2offset = {
    'G-': 6,'F#': 6,
    'G':  5,
    'A-': 4,'G#': 4,
    'A':  3,
    'B-': 2,'A#': 2,
    'B':  1,
    'C':  0,
    'D-':-1,'C#':-1,
    'D': -2,
    'E-':-3,'D#':-3,
    'E': -4,
    'F': -5,'E#':-5,
}
mode_key = ['C','D','E','F','G','A','B','C','A']
mode_name = ['ionian','dorian','phyrgian','lydian','mixolydian','aeolian','locrian','major','minor']
name_mode = {i[0]:i[1] for i in zip(mode_name, mode_key)}
num_mode = {str(idx+1):key for idx,key in enumerate(mode_key[:-2])}
mode2key = {**name_mode, **num_mode}

def keyc_offset(key, mode):
    if len(key) == 2 and key[-1] == 'b': key = key.replace('b', '-')
    mode_basekey = mode2key[str(mode).lower()] # find mode relative to C major - minor:A,dorian:D
    mode_offset = major2offset[mode_basekey] # get mode offset from C major
    key_offset = major2offset[key] # key offset when in major mode
    transpose_offset = key_offset - mode_offset # actual transpose offset
    if transpose_offset <= -6: transpose_offset = transpose_offset + 12
    return transpose_offset
