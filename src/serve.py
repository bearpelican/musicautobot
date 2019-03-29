import music21
import torch

from fastai.distributed import *
from fastai.text.models.transformer import *

import numpy as np

from .fastai_data import *
from .lmnp_transformer import *
from .encode_data import *

import uuid

# source_dir = 'midi_encode/np/shortdur'
# path = Path('../../data/midi/v9/')/source_dir
# out_path = Path('../../data/generated/')

def get_config(vocab_path):
    bs=16
    bptt=256
    
    VOCAB_SZ = create_vocab_sizes(vocab_path)
    N_COMPS = len(VOCAB_SZ)
    N_EMBS = 128
    EMB_IDXS = range(N_COMPS)
    EMB_DIM = [N_EMBS]*len(EMB_IDXS)
    EMB_MAP = list(zip(EMB_IDXS,VOCAB_SZ,EMB_DIM))
    EMB_MAP

    idx2embidx = { i:EMB_MAP[i] for i in range(N_COMPS) }
    total_embs = sum([v[-1] for k,v in idx2embidx.items()])

    config = tfmerXL_lm_config.copy()
    config['emb_map'] = list(zip(EMB_IDXS,VOCAB_SZ,EMB_DIM))
    config['idx_map'] = idx2embidx
    config['loss_weights'] = [1,1] # note,duration
    config['pad_idx'] = PADDING_IDX+ENC_OFFSET
    config['bos_idx'] = VALTBOS+ENC_OFFSET
    config['enc_offset'] = ENC_OFFSET
    config['transpose_range'] = (0,12)
    config['mask_type'] = MaskType.RandomWindow
    config['act'] = Activation.GeLU
    # config['act'] = Activation.ReLU

    config['d_model'] = total_embs
    config['mem_len'] = 512

    config['resid_p'] = 0.1
    config['attn_p'] = 0.1 # attention dropout
    config['ff_p'] = 0.1
    config['embed_p'] = 0.1 # embedding dropout
    config['output_p'] = 0.1 # decoder dropout (before final linear layer)

    config['bs'] = 16
    config['bptt'] = 256
    # config['path'] = path
    # config['cache_name'] = cache_name
    return config

def load_data(path, cache_name, enc_offset, transpose_range, **kwargs):
    transpose_tfm = partial(rand_transpose, enc_offset=enc_offset, rand_range=transpose_range)
    data = LMNPDataBunch.load(path=path, cache_name=cache_name, **kwargs, train_tfms=[transpose_tfm])
    return data

def load_learner(data, config, load_path=None):
    learn = language_model_learner(data, config, clip=0.25)
    if load_path:
        state = torch.load(load_path, map_location='cpu')
        get_model(learn.model).load_state_dict(state['model'], strict=False)
    return learn


# Serving functions

import pandas as pd
def get_htlist(path, source_dir, use_cache=True):
    json_path = path/'htlist.json'
    if use_cache and json_path.exists():
        with open(json_path, 'r') as fp:
            htlist = json.load(fp)
    else:
        df = pd.read_csv(path/'midi_encode.csv')
        df = df.loc[df[source_dir].notna()] # make sure it exists
        df = df.loc[df.source == 'hooktheory'] # hooktheory only
        df = df.rename(index=str, columns={source_dir: 'numpy'}) # shortdur -> numpy
        df = df.reindex(index=df.index[::-1]) # A's first
        df = df.where((pd.notnull(df)), None) # nan values break json

        htlist = df.to_dict('records') # row format
        htlist = [format_htsong(s) for s in htlist] # normalize artist, title, create song ID
        htlist = { s['sid']:s for s in htlist}
        with open(json_path, 'w') as fp:
            json.dump(htlist, fp)
    return htlist

def format_htsong(s):
    s = s.copy()
    s['title'] = s['title'].title().replace('-', ' ')
    s['artist'] = s['artist'].title().replace('-', ' ')
    s['sid'] = str(hash(s['midi']))
    return s

def search_htlist(htlist, keywords='country road', max_results=10):
    keywords = keywords.split(' ')
    def contains_keywords(f): return all([k in str(f) for k in keywords])
    res = []
    for k,s in htlist.items():
        if contains_keywords(s['numpy']): res.append(s)
        if len(res) >= max_results: break
    return res

def get_filelist(path):
    files = get_files(path/'hooktheory', extensions=['.npy'], recurse=True)
    return files

def stream2midifile(stream, np_path):
    return stream.write("midi", np_path.with_suffix('.mid'))
    
def stream2scoreimg(stream, np_path):
    return stream.write('musicxml.png', np_path.with_suffix('.xml'))

def predict_from_file(learn, midi_file=None, np_file=None, seed_len=60, n_words=340, 
                         temperatures=(1.5,0.9), min_ps=(1/128,0.0), **kwargs):
    file = np_file
    song_np = np.load(file)
    seed_np = np.load(file)[:seed_len]
    xb = torch.tensor(seed_np)[None]
    pred, seed = learn.predict(xb, n_words=n_words, temperatures=temperatures, min_ps=min_ps)
    full = np.concatenate((seed,pred), axis=0)
    
    return pred, seed, full

# NOTE: looks like npenc does not include the separator. 
# This means we don't have to remove the last (separator) step from the seed in order to keep predictions
def predict_from_midi(learn, midi_path=None, n_words=340, 
                      temperatures=(1.5,0.9), min_ps=(1/128,0.0), **kwargs):
    seed_np = midi2npenc(midi_path)
    xb = torch.tensor(seed_np)[None]
    pred, seed = learn.predict(xb, n_words=n_words, temperatures=temperatures, min_ps=min_ps)
    full = np.concatenate((seed,pred), axis=0)
    
    return pred, seed, full

def save_comps(out_path, pid, nptype='pred', bpm=120): # p = pred, f = full, s = seed
    np_path = out_path/pid/f'{nptype}.npy'
    npenc = np.load(np_path)
    
    stream = npenc2stream(npenc)
    
    midi = stream2midifile(stream, np_path)
    score = stream2scoreimg(stream, np_path)
    
    return Path(midi), Path(score)

def save_preds(pred, seed, full, out_path):
    pid = str(uuid.uuid4())
    path = out_path/pid
    path.mkdir(parents=True, exist_ok=True)
    np.save(path/f"pred.npy", pred)
    np.save(path/f"seed.npy", seed)
    np.save(path/f"full.npy", full)
    return pid