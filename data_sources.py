import csv
import pandas as pd
from fastai.data_block import get_files
from midi_data import save_json, load_json, keyc_offset
import concurrent
from concurrent.futures import ProcessPoolExecutor
from fastprogress.fastprogress import master_bar, progress_bar
import json
import music21

def get_stream_attr(s):
    "Pull stream metadata from midi file"
    instruments = [i.instrumentName for i in list(s.getInstruments(recurse=True)) if i.instrumentName]
    metronome = list(filter(lambda x: isinstance(x, music21.tempo.MetronomeMark), s.flat))[0]
    bpm = metronome.getQuarterBPM()
    s_flat = s.flat
    key = s_flat.analyze('key')
    time_sig = s_flat.timeSignature.ratioString if hasattr(s_flat.timeSignature, 'ratioString') else None
    offset = keyc_offset(key.tonic.name, key.mode)
    return {
        'instruments': instruments,
        'bpm': bpm,
        'inferred_key': f'{key.tonic.name} {key.mode}',
        'seconds': s_flat.seconds,
        'time_signature': time_sig,
        'inferred_offset': offset
    }

def process_parallel(func, arr, total=None, max_workers=None):
    "Process array in parallel"
    if total is None: total = len(arr)
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(func,o) for i,o in enumerate(arr)]
        for f in progress_bar(concurrent.futures.as_completed(futures), total=total):
            k,v = f.result(timeout=60)
            results[k] = v
    return results

def parse_songs(data):
    "Extract stream attributes"
    fp = data.get('file_path')
    metadata = data.get('metadata', {})
    attr = {}
    try: 
        stream = music21.converter.parse(fp)
        attr = get_stream_attr(stream)
    except Exception as e: print('Midi Exeption:', fp, e)
    return str(fp), {**metadata, **attr}

def parse_midi_dir(files, out_path, meta_func, limit=None, recurse=True, key_func=str):
    "Iterate through midi_source dir and map file to metadata"
    file2metadata = load_json(out_path)
    if file2metadata is None: file2metadata = {}
        
    if limit: files = files[:limit]
    files = [meta_func(fp) for fp in files if key_func(fp) not in file2metadata]
    
    parsed = process_parallel(parse_songs, files)
    file2metadata.update(parsed)
    
    json.dump(file2metadata, open(out_path, 'w'))
    
    return file2metadata


def arr2csv(arr, out_file):
    "Convert metadata array to csv"
    all_keys = {k for d in arr for k in d.keys()}
    arr = [format_values(x) for x in arr]
    with open(out_file, 'w') as f:
        dict_writer = csv.DictWriter(f, list(all_keys))
        dict_writer.writeheader()
        dict_writer.writerows(arr)
        
def format_values(d):
    "Format array values for csv encoding"
    def format_value(v):
        if isinstance(v, list): return ','.join(v)
        return v
    return {k:format_value(v) for k,v in d.items()}