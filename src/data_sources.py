"Parallel processing for midi files"
import csv
import pandas as pd
from fastai.data_block import get_files
from src.midi_data import save_json, load_json, keyc_offset
import concurrent
from concurrent.futures import ProcessPoolExecutor
from fastprogress.fastprogress import master_bar, progress_bar
import json
import music21
from pathlib import Path
from functools import partial

def get_stream_attr(s):
    "Pull stream metadata from midi file"
    instruments = [i.instrumentName for i in list(s.getInstruments(recurse=True)) if i.instrumentName]
    
    bpm = None
    metronome = list(filter(lambda x: isinstance(x, music21.tempo.MetronomeMark), s.flat))
    if len(metronome):
        bpm = metronome[0].getQuarterBPM()
    s_flat = s.flat
    time_sig = s_flat.timeSignature.ratioString if hasattr(s_flat.timeSignature, 'ratioString') else None
    key = s_flat.analyze('key')
    offset = keyc_offset(key.tonic.name, key.mode)
    inferred_key = f'{key.tonic.name} {key.mode}'

    seconds = None
    try: seconds = s_flat.seconds
    except Exception as e: pass
        
    return {
        'instruments': instruments,
        'bpm': bpm,
        'inferred_key': inferred_key,
        'seconds': seconds,
        'quarter_length': f'{s_flat.duration.quarterLength}',
        'time_signature': time_sig,
        'inferred_offset': offset
    }

def process_parallel(func, arr, total=None, max_workers=None, timeout=None):
    "Process array in parallel"
    if total is None: total = len(arr)
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(func,o) for i,o in enumerate(arr)]
        for f in progress_bar(concurrent.futures.as_completed(futures, timeout=timeout), total=total):
            k,v = f.result()
            results[k] = v
    return results

def transform_csv_row(idxrow, transform_func, base_path, source_dir, out_dir, out_extension, source_col=None):
    idx,row = idxrow
    if source_col is not None: file = row[source_col]
    else: file = row[source_dir]
    
    if not isinstance(file, str): return idx,None
    file = Path(base_path)/file
    if not file.exists(): return idx, None
    
    out_file = Path(str(file).replace(f'/{source_dir}/', f'/{out_dir}/'))
    if out_extension is not None: out_file = out_file.with_suffix(out_extension)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists(): return idx,str(out_file.relative_to(base_path))
    try: transform_func(file, out_file, row)
    except Exception as e:
        print('Error converting midi to sequence', e, file)
        return idx,None
    return idx,str(out_file.relative_to(base_path))

def parse_songs(metadata, base_path):
    "Extract stream attributes"
    midi_path = metadata.get('midi', metadata.get('mxl', None))
    fp = Path(base_path)/midi_path
    attr = {}
    try: 
        stream = music21.converter.parse(fp)
#         is_small_file = fp.stat().st_size/1000 < 200 # (AS) this may not matter. For some reason ecomp files take a long time to parse
        attr = get_stream_attr(stream)
    
        if fp.suffix == '.mxl': 
            new_fp = Path(str(fp.with_suffix('.mid')).replace('midi_sources', 'midi_sources/from_mxl'))
            new_fp.parent.mkdir(parents=True, exist_ok=True)
            if not new_fp.exists(): stream.write('midi', new_fp)
            attr['midi'] = str(new_fp.relative_to(base_path))
        
    except Exception as e: print('Midi Exeption:', fp, e)
    return midi_path, {**metadata, **attr}

def parse_midi_dir(files, out_path, base_path, meta_func, chunk_size=None, recurse=True):
    "Iterate through midi_source dir and map file to metadata"
    file2metadata = load_json(out_path)
    if file2metadata is None: file2metadata = {}
        
    files = [meta_func(fp) for fp in files if str(fp.relative_to(base_path)) not in file2metadata]
    
    def process_files(arr):
        parse_func = partial(parse_songs, base_path=base_path)
        parsed = process_parallel(parse_func, arr)
        file2metadata.update(parsed)
        json.dump(file2metadata, open(out_path, 'w'))
        
    if chunk_size is not None:
        for chunk in chunks(files, chunk_size):
            process_files(chunk)
    else: process_files(files)

    
    return file2metadata

def chunks(l, n):
    "Yield successive `n`-sized chunks from `l`."
    for i in range(0, len(l), n): yield l[i:i+n]

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