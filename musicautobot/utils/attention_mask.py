import numpy as np
import torch

def window_mask(x_len, device, m_len=0, size=(1,1)):
    win_size,k = size
    mem_mask = torch.zeros((x_len,m_len), device=device)
    tri_mask = torch.triu(torch.ones((x_len//win_size+1,x_len//win_size+1), device=device),diagonal=k)
    window_mask = tri_mask.repeat_interleave(win_size,dim=0).repeat_interleave(win_size,dim=1)[:x_len,:x_len]
    if x_len: window_mask[...,0] = 0 # Always allowing first index to see. Otherwise you'll get NaN loss
    mask = torch.cat((mem_mask, window_mask), dim=1)[None,None]
    return mask.bool() if hasattr(mask, 'bool') else mask.byte()
    
def rand_window_mask(x_len,m_len,device,max_size:int=None,p:float=0.2,is_eval:bool=False):
    if is_eval or np.random.rand() >= p or max_size is None: 
        win_size,k = (1,1)
    else: win_size,k = (np.random.randint(0,max_size)+1,0)
    return window_mask(x_len, device, m_len, size=(win_size,k))

def lm_mask(x_len, device):
    mask = torch.triu(torch.ones((x_len, x_len), device=device), diagonal=1)[None,None]
    return mask.bool() if hasattr(mask, 'bool') else mask.byte()
