import os
import json
import copy
import pickle
import numpy as np
from .lookup_tables import VAL_TO_NAME, KEY_TO_SCALE, MODE_TO_KEY, NOTE_TO_OFFSET, ACCIDENTAL_DICT
from collections import OrderedDict


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


# chord
def compvec_to_comp(comp_vec):
    return comp_vec[comp_vec != np.array(None)]


def comp_to_compvec(comp):
    # comp_vec
    # 0 1 2 3 4 5 6  7  8
    # 1 2 3 4 5 7 9 11 13
    # do "not" use this func after 'sus', 'emb' and 'alter'

    comp_vec = np.array([None] * 9)

    # triad
    comp_vec[0] = comp[0]
    comp_vec[2] = comp[1]
    comp_vec[4] = comp[2]

    # 7
    if len(comp) >= 4:
        comp_vec[5] = comp[3]

    if len(comp) >= 5:
        comp_vec[6] = comp[4]

    if len(comp) >= 6:
        comp_vec[7] = comp[5]

    if len(comp) >= 7:
        comp_vec[8] = comp[6]

    if len(comp) >= 8:
        raise ValueError('Impossible!!')

    return comp_vec


def chord_to_string(data):
    base = to_name(data['root']) + data['quality']
    if data['quality'] in ['m', 'ø', 'o']:
        base = base.lower()
    base += ('' if data['chord_type'] is 5 else str(data['chord_type']))
    if data['inv']:
        base += (' | ' + to_name(data['bass']))
    if data['sus'] is not None:
        base += (' ' + data['sus'])
    if data['emb'] is not None:
        for e in data['emb']:
            base += ' (%s)' % e
    if data['alter'] is not None:
        for a in data['alter']:
            base += ' (%s)' % a
    return base


def to_name(input_, sys=0):
    output = VAL_TO_NAME[(input_+120) % 12][sys]
    return output


def to_chromagram(comp):
    return (np.array(comp) + 120) % 12


def to_names(input_, sys=0):
    input_ = to_chromagram(input_)
    output = [VAL_TO_NAME[i][sys] for i in input_]
    return output


def scale_extension(scale, num=5):
    scale_extended = []

    for i in range(num):
        scale_extended += [(s + 12*i) for s in scale]
    return scale_extended


def get_quality(comp):
    triad_symbol = ['', 'm', 'o']
    seventh_symbol = ['maj', 'm', '', 'ø']

    interval = [comp[idx+1] - comp[idx] for idx in range(len(comp)-1)]

    if interval == [4, 3]:
        quality = triad_symbol[0]
    elif interval == [3, 4]:
        quality = triad_symbol[1]
    elif interval == [3, 3]:
        quality = triad_symbol[2]

    elif interval == [4, 3, 4]:
        quality = seventh_symbol[0]
    elif interval == [3, 4, 3]:
        quality = seventh_symbol[1]
    elif interval == [4, 3, 3]:
        quality = seventh_symbol[2]
    elif interval == [3, 3, 4]:
        quality = seventh_symbol[3]
    else:
        raise ValueError("Unknow compostions", comp)

    return quality


def get_scale(key):
    scale = KEY_TO_SCALE[key]
    scale_extended = scale_extension(scale)
    return scale_extended


def get_num_inversion(fb):
    if fb == '6' or fb == '65':
        inv = 1
    elif fb == '64' or fb == '43':
        inv = 2
    elif fb == '42':
        inv = 3
    else:
        inv = 0
    return inv


def set_sus(comp_vec, scale, sd, input_):
    if input_ is None:
        return comp_vec
    elif input_ == 'sus2':
        comp_vec[1] = scale[sd+1]
    elif input_ == 'sus4':
        comp_vec[3] = scale[sd+3]
    elif input_ == 'sus42':
        comp_vec[1] = scale[sd+1]
        comp_vec[3] = scale[sd+3]
    else:
        raise ValueError('Unknown sus: %s' % input_)
    comp_vec[2] = None  # must omit 3
    return comp_vec


def add_comp_vec(comp_vec, add_note, sd, scale):
    if add_note == '5':
        return comp_vec
    if add_note == '9':
        comp_vec[6] = scale[sd+8]
    elif add_note == '11':
        comp_vec[7] = scale[sd+10]
    elif add_note == '13':
        comp_vec[8] = scale[sd+12]
    else:
        raise ValueError('Unknown note to add ', add_note)
    return comp_vec


def set_emb(comp_vec, scale, sd, input_):
    if input_ is None:
        return comp_vec, [], []
    if not isinstance(input_, list):
        input_ = [input_]

    alter_info = []
    emb_info = []
    for emb_event in input_:
        if emb_event in ['#5', 'b5', 'b9', '#9', '#11', 'B13']:
            alter_info.append(emb_event)
            add_note = emb_event[1:]
        elif emb_event in ['add9', 'add11', 'add13']:
            add_note = emb_event[3:]
            emb_info.append(emb_event)
        else:
            raise ValueError('Unknown emb: %s' % input_)
        add_comp_vec(comp_vec, add_note, sd, scale)
    return comp_vec, alter_info, emb_info


def set_alter(comp_vec, input_):
    if input_ is None:
        return comp_vec, None
    if not isinstance(input_, list):
        input_ = [input_]
    alter_map = np.array([.0] * 9)

    for alt_event in input_:
        # proc acc
        acc = alt_event[0]
        note = alt_event[1:]
        if acc == '#':
            op = 1
        elif acc == 'b':
            op = -1
        else:
            raise ValueError('Unknown acc in alter: %s' % input_)
        print(alt_event)

        # set idx of vec
        if note == '5':
            vidx = 4
        elif note == '9':
            vidx = 6
        elif note == '11':
            vidx = 7
        elif note == '13':
            vidx = 8
        else:
            raise ValueError('Unknown note in alter: %s' % input_)

        # set vec and map
        if comp_vec[vidx]:  # avoid None + op
            comp_vec[vidx] += op
            alter_map[vidx] = op
    return comp_vec, alter_map


def set_inversion(comp_vec, inv):
    if inv is 0:
        return comp_vec

    comp_idx = 0
    while inv:
        if comp_vec[comp_idx] is not None:
            comp_vec[comp_idx] += 12
            inv -= 1
        comp_idx += 1
    return comp_vec


def set_compositions(scale, fb, sd):
    if fb in [None, '6', '64']:
        chord_type = 5
    elif fb in ['7', '65', '43', '42']:
        chord_type = 7
    else:
        chord_type = int(fb)

    comp = np.array(scale[sd:sd+chord_type:2])
    return comp, chord_type


def is_int(input_):
    if input_ is None:
        return input_
    try:
        return int(input_)
    except ValueError:
        if input_ == 'b':
            return -3
        else:
            raise ValueError('Unknown borrowed chord')


def chord_parser(chord, mode, key_offset):
    if chord['isRest']:
        return None

    # extract basic info
    sd = int(chord['sd']) - 1     # root
    fb = chord['fb']              # tension & inversion
    sec = chord['sec']            # secondary chord
    borrowed = chord['borrowed']  # borrowed mode

    # determine the mode
    borrowed = is_int(borrowed)
    chord_key = MODE_TO_KEY[int(mode)]if borrowed is None else borrowed
    chord_key = 6 if chord_key > 6 else chord_key
    chord_key = -6 if chord_key < -6 else chord_key

    # secondary chord
    sec_offset = 0
    if sec:
        # switch to 'sec' degree note within the current mode
        scale = KEY_TO_SCALE[MODE_TO_KEY[int(mode)]]
        new_key_note = scale[int(sec) - 1]

        # set that note to new key
        new_key = VAL_TO_NAME[new_key_note][0]

        # get the key shift offset
        sec_offset = NOTE_TO_OFFSET[new_key]
        chord_key = 0

    # determine the scale according to the key(mode) of the chord
    scale = get_scale(chord_key)

    # set compositional notes
    comp, chord_type = set_compositions(scale, fb, sd)

    # determine the quality by triads or seventh
    # (9, 11, 13-th chords are seen as seventh)
    comp_t = comp[0:3] if chord_type is 5 else comp[0:4]
    quality = get_quality(comp_t)

    # add shift from secondary chords
    comp = (comp + sec_offset)

    # set compvec (for sus/add/omit)
    comp_vec = comp_to_compvec(comp)

    # sus (omit 3)
    sus = chord['sus']
    comp_vec = set_sus(comp_vec, scale, sd, sus)

    # emb (add/omit)
    if 'emb' in chord:
        emb = chord['emb']
        comp_vec, alter_info, emb_info = set_emb(comp_vec, scale, sd, emb)
    else:
        emb_info = []
        alter_info = []

    # alter (won't change the quality)
    alter_info = alter_info if len(alter_info) else chord['alternate']
    comp_vec, alter_map = set_alter(comp_vec, alter_info)

    # set inversion (won't change the root, but bass)
    inv = get_num_inversion(fb)
    comp_vec = set_inversion(comp_vec, inv)

    # set result
    comp = compvec_to_comp(comp_vec)
    root = (comp[0] + 120) % 12   # for chord name
    bass = np.nanmin(comp)         # for bass (real root)

    data = OrderedDict([
        # basic compositions
        ('root', root),
        ('bass', bass),
        ('comp_vec', comp_vec),
        ('composition', comp),

        # basic info
        ('quality', quality),
        ('chord_type', chord_type),
        ('chord_mode', chord_key),

        # event info
        ('isRest', chord['isRest']),
        ('event_on', chord['event_on']),
        ('event_off', chord['event_off']),
        ('event_duration', chord['event_duration']),

        # additional info
        ('inv', inv),
        ('sus', sus),
        ('alter', alter_info),
        ('emb', emb_info),
        ('alter_map', alter_map),
        ])

    # key shifting of the symbol
    data = chord_key_shifting(data, key_offset)

    # set chord name
    data['symbol'] = chord_to_string(data)
    return data


# note
def note_parser(note, mode, key_offset=0):
    if note['isRest']:
        return None

    octave = float(note['octave'])
    scale_degree = note['scale_degree']

    sign = ACCIDENTAL_DICT[scale_degree[1]] if len(scale_degree) is 2 else 0
    scale_degree = int(scale_degree[0])

    pitch = KEY_TO_SCALE[MODE_TO_KEY[int(mode)]][scale_degree-1] + sign + 12 * octave
    pitch += key_offset

    data = OrderedDict([
        # basic info
        ('pitch', pitch),

        # event info
        ('isRest', note['isRest']),
        ('event_on', note['event_on']),
        ('event_off', note['event_off']),
        ('event_duration', note['event_duration'])
    ])

    return data


# song-level key shifting
def get_key_offset(key):
    return NOTE_TO_OFFSET[key]


def chord_key_shifting(data, key_offset):
    comp_vec = data['comp_vec']
    for idx, c in enumerate(comp_vec):
        if c is not None:
            comp_vec[idx] = c + key_offset

    reset_chord_basic(data, comp_vec)
    return data


def reset_chord_basic(data, comp_vec):
    '''
    Everytime after changing the comp_vec (ex: key shifting),
    use this function to update the chord object to avoid
    unnecessary problems

    '''
    comp = compvec_to_comp(comp_vec)
    data['bass'] = np.nanmin(comp)
    data['comp_vec'] = comp_vec
    data['composition'] = compvec_to_comp(comp_vec)
    data['root'] = (comp[0] + 120) % 12
    return data


def proc_event_to_symbol(melody_track, chord_track, mode, key='C'):
    # get key offset
    key_offset = get_key_offset(key)

    # melody
    melody_events = []
    for nidx, note in enumerate(melody_track):
        # print(nidx, note)
        melody_events.append(note_parser(note, mode, key_offset))

    # chord
    chord_events = []
    for cidx, chord in enumerate(chord_track):
        # print(cidx, chord)
        chord_events.append(chord_parser(chord, mode, key_offset))

    return melody_events, chord_events


def proc_roman_to_symbol(raw, is_key=True, save_path=None, name='tab', save_type='pickle'):
    # metadata
    metadata = raw['metadata']
    mode = metadata['mode'] if metadata['mode'] is not None else 1
    key = metadata['key'] if is_key else 'C'

    # tracks
    melody_track = raw['tracks']['melody']
    chord_track = raw['tracks']['chord']

    # to event symbol
    melody_events, chord_events = proc_event_to_symbol(melody_track, chord_track, mode, key)

    # overwrite roman
    raw_new = copy.deepcopy(raw)
    raw_new['tracks']['melody'] = melody_events
    raw_new['tracks']['chord'] = chord_events

    # saving
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if save_type is 'pickle':
            file_name = os.path.join(save_path, name+'.pickle')
            with open(file_name, 'wb') as handle:
                pickle.dump(raw_new, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif save_type is 'json':
            file_name = os.path.join(save_path, name+'.json')
            with open(file_name, 'w') as handle:
                json.dump(raw_new, handle, cls=MyEncoder)
        else:
            raise ValueError('Unkown type for saving')

    return raw_new







def hchord_parser(chord, mode, key_offset, reset_to_base=True):
    if chord['sd'] == 'rest': return None

    # extract basic info
    sd = int(chord['sd']) - 1     # root
    fb = chord['fb']              # tension & inversion
    sec = chord['sec']            # secondary chord
    borrowed = chord['borrowed']  # borrowed mode

    # determine the mode
    borrowed = is_int(borrowed)
    chord_key = MODE_TO_KEY[int(mode)]if borrowed is None else borrowed
    chord_key = 6 if chord_key > 6 else chord_key
    chord_key = -6 if chord_key < -6 else chord_key

    # secondary chord
    sec_offset = 0
    if sec:
        # switch to 'sec' degree note within the current mode
        scale = KEY_TO_SCALE[MODE_TO_KEY[int(mode)]]
        new_key_note = scale[int(sec) - 1]

        # set that note to new key
        new_key = VAL_TO_NAME[new_key_note][0]

        # get the key shift offset
        sec_offset = NOTE_TO_OFFSET[new_key]
        chord_key = 0

    # determine the scale according to the key(mode) of the chord
    scale = get_scale(chord_key)

    # set compositional notes
    comp, chord_type = set_compositions(scale, fb, sd)

    # determine the quality by triads or seventh
    # (9, 11, 13-th chords are seen as seventh)
    comp_t = comp[0:3] if chord_type is 5 else comp[0:4]
    quality = get_quality(comp_t)

    # add shift from secondary chords
    comp = (comp + sec_offset)

    # set compvec (for sus/add/omit)
    comp_vec = comp_to_compvec(comp)

    # sus (omit 3)
    sus = chord['sus']
    comp_vec = set_sus(comp_vec, scale, sd, sus)

    # emb (add/omit)
    if 'emb' in chord:
        emb = chord['emb']
        comp_vec, alter_info, emb_info = set_emb(comp_vec, scale, sd, emb)
    else:
        emb_info = []
        alter_info = []

    # alter (won't change the quality)
    alter_info = alter_info if len(alter_info) else chord['alternate']
    comp_vec, alter_map = set_alter(comp_vec, alter_info)

    # set inversion (won't change the root, but bass)
    inv = get_num_inversion(fb)
    comp_vec = set_inversion(comp_vec, inv)

    # set result
    comp = compvec_to_comp(comp_vec)
    root = (comp[0] + 120) % 12   # for chord name
    bass = np.nanmin(comp)         # for bass (real root)

    data = OrderedDict([
        # basic compositions
        ('root', root),
        ('bass', bass),
        ('comp_vec', comp_vec),
        ('composition', comp),

        # basic info
        ('quality', quality),
        ('chord_type', chord_type),
        ('chord_mode', chord_key),
        
        # additional info
        ('inv', inv),
        ('sus', sus),
        ('alter', alter_info),
        ('emb', emb_info),
        ('alter_map', alter_map),
        ])

    # key shifting of the symbol
    data = chord_key_shifting(data, key_offset)
    
    # After offset, let's reset the chord to be the lowest possible offset on new scale
    if reset_to_base:
        reset_base = data['root'] - data['bass']
        data = chord_key_shifting(data, reset_base)

    # set chord name
    data['symbol'] = chord_to_string(data)
    return data


# note
def hnote_parser(note, mode, key_offset=0):
    if note['scale_degree'] == 'rest':
        return None

    octave = float(note['octave'])
    scale_degree = note['scale_degree']

    sign = ACCIDENTAL_DICT[scale_degree[1]] if len(scale_degree) is 2 else 0
    scale_degree = int(scale_degree[0])

    pitch = KEY_TO_SCALE[MODE_TO_KEY[int(mode)]][scale_degree-1] + sign + 12 * octave
    pitch += key_offset

    data = OrderedDict([
        # basic info
        ('pitch', pitch),
    ])

    return data
