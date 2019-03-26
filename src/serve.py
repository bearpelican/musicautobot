import music21
import torch

from fastai.distributed import *
from fastai.text.models.transformer import *

import numpy as np

from .fastai_data import *
from .lmnp_transformer import *
from .encode_data import *


import uuid

source_dir = 'midi_encode/np/shortdur'
path = Path('../../data/midi/v9/')/source_dir
# out_path = Path('../../data/generated/')

def get_config(path, cache='tmp/hook'):
    bs=16
    bptt=256
    
    VOCAB_SZ = create_vocab_sizes(path/'tmp/all')
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

    config['d_model'] = total_embs
    config['mem_len'] = 512

    config['resid_p'] = 0.1
    config['attn_p'] = 0.1 # attention dropout
    config['ff_p'] = 0.1
    config['embed_p'] = 0.1 # embedding dropout
    config['output_p'] = 0.1 # decoder dropout (before final linear layer)

    config['bs'] = 16
    config['bptt'] = 256
    config['cache_name'] = 'tmp/hook'
    config['path'] = path
    return config

def load_data(config):
    transpose_tfm = partial(rand_transpose, enc_offset=config['enc_offset'], rand_range=config['transpose_range'])
    data = LMNPDataBunch.load(**config, train_tfms=[transpose_tfm])
    return data

def load_learner(data, config):
    learn = language_model_learner(data, config, clip=0.25)
    if 'load_path' in config:
        state = torch.load(config['load_path'], map_location='cpu')
        get_model(learn.model).load_state_dict(state['model'], strict=False)
    return learn


# Serving functions

# import pandas as pd
# def song_csv():
#     if not (path/'midi_encode.pkl').exists():
#         df = pd.read_csv(path/'midi_encode.csv')
#         df.to_pickle(path/'midi_encode.pkl')
#     else:
#         df = pd.read_pickle(path/'midi_encode.pkl')

#     df = df.loc[df[source_dir].notna()] # make sure it exists
#     df = df.loc[df.source == 'hooktheory'] # hooktheory only
#     # df.loc[df.artist.str.contains('garrix')]
#     files = song_csv()[source_dir].values.tolist()
#     return df, files

def search_files(files, keywords='country road'):
    keywords = keywords.split(' ')
    def contains_keywords(f): return all([k in str(f) for k in keywords])
    search = [f for f in files if contains_keywords(f)]
    return search

def get_hooktheory_files(config):
    files = get_files(config['path']/'hooktheory', extensions=['.npy'], recurse=True)
    return files


def stream2midifile(stream, np_path):
    return stream.write("midi", np_path.with_suffix('.mid'))
    
def stream2scoreimg(stream, np_path):
    return stream.write('musicxml.png', np_path.with_suffix('.xml'))

def generate_predictions(learn, midi_file=None, np_file=None, seed_len=60, n_words=340, 
                         temperatures=(1.5,0.9), min_ps=(1/128,0.0), **kwargs):
    file = np_file
    song_np = np.load(file)
    seed_np = np.load(file)[:seed_len]
    xb = torch.tensor(seed_np)[None]
    pred, seed = learn.predict(xb, n_words=n_words, temperatures=temperatures, min_ps=min_ps)
    full = np.concatenate((seed,pred), axis=0)
    
    return pred, seed, full

def save_comps(out_path, pid, nptype='p'): # p = pred, f = full, s = seed
    np_path = out_path/pid/f'pred.npy'
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