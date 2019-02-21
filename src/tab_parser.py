import xmltodict
import os
import json
import pickle
from lxml import etree

encoding = "utf-8"


def load_data(file_path):
    with open(file_path, "r", encoding=encoding) as f:
        content = f.read()
    content = content.replace('%20', '')
    return content


def xml_parser(content):
    parser = etree.XMLParser(recover=True, encoding=encoding)
    root = etree.fromstring(content, parser=parser)
    return root


def get_metadata(root):
    meta_list = ['title', 'beats_in_measure', 'BPM', 'key', 'YouTubeID', 'mode']

    metadata = dict()
    for e in meta_list:
        tag = root.find('.//' + e)
        metadata[e] = tag.text if tag is not None else None

    # duration
    tag = root.find('.//duration')
    tag = root.find('.//section_duration') if tag is None else tag
    metadata['duration'] = tag.text if tag is not None else None

    version = root.find('version')
    version = version.text if version is not None else None
    return metadata, version


def get_lead_sheet(root, version):
    segments_tag = root.findall('.//segment')

    # set chord tag according to version
    chord_tag = 'chords' if root.tag == 'super' else 'harmony'

    segment_list = []
    num_measures = 0

    for segment in segments_tag:
        num_measure = float(segment.find('numMeasures').text)

        # melody
        note_tags = segment.findall('.//notes/note')
        note_list = [xmltodict.parse(etree.tostring(n))['note'] for n in note_tags] if note_tags else []

        # chord
        chord_tags = segment.findall('.//' + chord_tag + '/chord')
        chord_list = [xmltodict.parse(etree.tostring(c))['chord'] for c in chord_tags] if chord_tags else []

        segment_list.append({'notes': note_list, 'chords': chord_list, 'num_measure': num_measure})
        num_measures += num_measure

    return segment_list, num_measures


def event_localization(measure_offset, start_beat_abs, duration):
    event_on = measure_offset + start_beat_abs
    event_off = measure_offset + start_beat_abs + duration
    return event_on, event_off


def proc_object(object_, measure_offset, type_=None):
    if type_ is 'note':
        duration = float(object_['note_length'])
        object_.pop('note_length')
    elif type_ is 'chord':
        duration = float(object_['chord_duration'])
        object_.pop('chord_duration')
    else:
        raise ValueError('Unknown object')

    event_on, event_off = event_localization(
        measure_offset,
        float(object_['start_beat_abs']),
        duration)

    # delete item
    object_.pop('start_measure')
    object_.pop('start_beat')
    object_.pop('start_beat_abs')

    # add item
    object_['event_on'] = event_on
    object_['event_off'] = event_off
    object_['event_duration'] = duration

    # change item
    object_['isRest'] = bool(int(object_['isRest']))

    return object_


def segments_parser(segments, mode, beats_in_measure):
    measure_counter = 0

    chord_track = []
    melody_track = []

    for sidx, segment in enumerate(segments):
        measure_offset = measure_counter * float(beats_in_measure)

        for chord in segment['chords']:
            chord_track.append(proc_object(chord, measure_offset, type_='chord'))
        for note in segment['notes']:
            melody_track.append(proc_object(note, measure_offset, type_='note'))

        measure_counter += int(segment['num_measure'])

    return melody_track, chord_track


def proc_xml(file_path, save_path=None, name='tab', save_type='pickle'):
    content = load_data(file_path)
    root = xml_parser(content)
    metadata, version = get_metadata(root)
    segments, num_measures = get_lead_sheet(root, version)

    mode = int(metadata['mode']) if metadata['mode'] is not None else 1
    beats_in_measure = int(metadata['beats_in_measure'])

    melody, chord = segments_parser(segments, mode, beats_in_measure)

    data = {
        'version': version,
        'metadata': metadata,
        'tracks': {
            'melody': melody,
            'chord': chord,
        },
        'num_measures': num_measures,
    }

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if save_type is 'pickle':
            file_name = os.path.join(save_path, name+'.pickle')
            with open(file_name, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif save_type is 'json':
            file_name = os.path.join(save_path, name+'.json')
            with open(file_name, 'w') as handle:
                json.dump(data, handle)
        else:
            raise ValueError('Unkown type for saving')

    return data


def traverse_dir(root_dir, extension='.xml'):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))

    return file_list


def get_postfix_dirpath(filename, idx=-4):
    path = os.path.normpath(filename)
    dir_list = path.split(os.sep)
    new_path = ''

    for d in dir_list[idx:]:
        new_path = os.path.join(new_path, d)

    return new_path


def proc_dir(file_list, root):
    num_file = len(file_list)
    for fidx in range(num_file):
        print('(%d/%d)' % (fidx, num_file))
        file_path = file_list[fidx]
        save_path = root
        name = os.path.basename(file_path)
        proc_xml(file_path, save_path=save_path, name=name, save_type='json')
