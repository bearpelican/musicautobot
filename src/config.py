from fastai.text.models.transformer import tfmerXL_lm_config, Activation
from .music_transformer.dataloader import MusicVocab

def default_config(vocab):
    config = tfmerXL_lm_config.copy()
    
    config['pad_idx'] = vocab.pad_idx
    config['bos_idx'] = vocab.bos_idx
    config['sep_idx'] = vocab.sep_idx
    config['transpose_range'] = (0,12)
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
    config = default_config(vocab)
    config['bias'] = True
    config['enc_layers'] = 8
    config['dec_layers'] = 8
    del config['n_layers']
    return config