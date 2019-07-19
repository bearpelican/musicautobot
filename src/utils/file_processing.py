"Parallel processing for midi files"
import csv
from fastprogress.fastprogress import master_bar, progress_bar
from pathlib import Path
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import numpy as np

# https://stackoverflow.com/questions/20991968/asynchronous-multiprocessing-with-a-worker-pool-in-python-how-to-keep-going-aft
def process_all(func, arr, timeout_func=None, total=None, max_workers=None, timeout=None):
    with ProcessPool() as pool:
        future = pool.map(func, arr, timeout=timeout)

        iterator = future.result()
        results = []
        for i in progress_bar(range(len(arr)), total=len(arr)):
            try:
                result = next(iterator)
                if result: results.append(result)
            except StopIteration:
                break  
            except TimeoutError as error:
                if timeout_func: timeout_func(arr[i], error.args[1])
    return results

def process_file(file_path, tfm_func=None, src_path=None, dest_path=None):
    "Utility function that transforms midi file to numpy array."
    output_file = Path(str(file_path).replace(str(src_path), str(dest_path))).with_suffix('.npy')
    if output_file.exists(): return output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Call tfm_func and save file
    npenc = tfm_func(file_path)
    if npenc is not None: 
        np.save(output_file, npenc)
        return output_file

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