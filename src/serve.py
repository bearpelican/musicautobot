import music21
import torch

from fastai.distributed import *
from fastai.text.models.transformer import *

import numpy as np

from .fastai_data import *
from .music_transformer import *
from .encode_data import *

import uuid

def v15_config(vocab):
    config = tfmerXL_lm_config.copy()
    
    config['pad_idx'] = vocab.pad_idx
    config['bos_idx'] = vocab.bos_idx
    config['sep_idx'] = vocab.sep_idx
    config['transpose_range'] = (0,12)
    config['rand_bptt'] = False
    config['note_range'] = vocab.note_range
    config['act'] = Activation.GeLU
    # config['act'] = Activation.ReLU

    config['mem_len'] = 512

    config['bs'] = 16
    config['bptt'] = 256
    
    config['d_model'] = 512
    config['vocab_size'] = len(vocab.itos)
    config['d_inner'] = 2048
    config['n_layers'] = 16
    
    config['n_heads'] = 8
    config['d_head'] = 64


    return config

def mlm_config(vocab):
    config = v15_config(vocab)
    config['bias'] = True
    config['enc_layers'] = 8
    config['dec_layers'] = 8
    del config['n_layers']
    return config

def mlm4_config(vocab):
    config = v15_config(vocab)
    config['bias'] = True
    config['enc_layers'] = 4
    config['dec_layers'] = 6
    del config['n_layers']
    return config
    
def v15m_config(vocab):
    config = v15_config(vocab)
    config['embed_p'] = 0.2
    return config

def load_music_data(path, cache_name, vocab, **kwargs):
    data = MusicDataBunch.load(path=path, cache_name=cache_name, **kwargs, 
                              train_tfms=[to_single_stream], valid_tfms=[to_single_stream])
    data.vocab = vocab
    return data

def load_music_learner(data, config, load_path=None):
    learn = music_model_learner(data, config)
    if load_path:
        state = torch.load(load_path, map_location='cpu')
        get_model(learn.model).load_state_dict(state['model'], strict=False)
    return learn

# New way 

import hashlib
import shutil

def df2records(path):
    df = pd.read_csv(path/'midi_encode.csv')
    df = df.loc[df[source_dir].notna()] # make sure it exists
    df = df.loc[df.source == 'hooktheory'] # hooktheory only
    df = df.rename(index=str, columns={source_dir: 'numpy'}) # shortdur -> numpy
    df = df.reindex(index=df.index[::-1]) # A's first
    df = df.where((pd.notnull(df)), None) # nan values break json
    return df.to_dict('records')

def format_meta(s):
    title = s['title'].title().replace('-', ' ')
    artist = s['artist'].title().replace('-', ' ')
    display = ' - '.join([title, artist])
    if s.get('section'): display += ' - ' + s['section'].title()
    sid = hashlib.md5(display.encode('utf-8')).hexdigest()
    
    source_file = file_path/data_dir/s['midi']
    to_file = encoded_path/f'{sid[::-1]}.mid'
    if not to_file.exists():
        shutil.copy(str(source_file), str(to_file))
    
    return {
        'title': title,
        'artist': artist,
        'bpm': s['ht_bpm'],
        'display': display,
        'genres': s['genres'],
        'sid': sid
    }

def build_db(path):
    recordlist = df2records(path)
    htlist = [format_meta(s) for s in recordlist]
    json_path = file_path/'data/assets/json/htlist.json'
    with open(json_path, 'w') as fp:
        json.dump(htlist, fp, separators=(',', ':'))
    return htlist
