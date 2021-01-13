from fastai.basics import *
from .transform import *
from ..music_transformer.dataloader import MusicDataBunch, MusicItemList
# Sequence 2 Sequence Translate

class S2SFileProcessor(PreProcessor):
    "`PreProcessor` that opens the filenames and read the texts."
    def process_one(self,item):
        out = np.load(item, allow_pickle=True)
        if out.shape != (2,): return None
        if not 16 < len(out[0]) < 2048: return None
        if not 16 < len(out[1]) < 2048: return None
        return out
    
    def process(self, ds:Collection):
        ds.items = [self.process_one(item) for item in ds.items]
        ds.items = [i for i in ds.items if i is not None] # filter out None

class S2SPartsProcessor(PreProcessor):
    "Encodes midi file into 2 separate parts - melody and chords."
    
    def process_one(self, item):
        m, c = item
        mtrack = MultitrackItem.from_npenc_parts(m, c, vocab=self.vocab)
        return mtrack.to_idx()
    
    def process(self, ds):
        self.vocab = ds.vocab
        ds.items = [self.process_one(item) for item in ds.items]
        
class Midi2MultitrackProcessor(PreProcessor):
    "Converts midi files to multitrack items"
    def process_one(self, midi_file):
        try:
            item = MultitrackItem.from_file(midi_file, vocab=self.vocab)
        except Exception as e:
            print(e)
            return None
        return item.to_idx()
        
    def process(self, ds):
        self.vocab = ds.vocab
        ds.items = [self.process_one(item) for item in ds.items]
        ds.items = [i for i in ds.items if i is not None]
    
class S2SPreloader(Callback):
    def __init__(self, dataset:LabelList, bptt:int=512, 
                 transpose_range=None, **kwargs):
        self.dataset,self.bptt = dataset,bptt
        self.vocab = self.dataset.vocab
        self.transpose_range = transpose_range
        self.rand_transpose = partial(rand_transpose_value, rand_range=transpose_range) if transpose_range is not None else None
        
    def __getitem__(self, k:int):
        item,empty_label = self.dataset[k]
        
        if self.rand_transpose is not None:
            val = self.rand_transpose()
            item = item.transpose(val)
        item = item.pad_to(self.bptt+1)
        ((m_x, m_pos), (c_x, c_pos)) = item.to_idx()
        return m_x, m_pos, c_x, c_pos
    
    def __len__(self):
        return len(self.dataset)

def rand_transpose_value(rand_range=(0,24), p=0.5):
    if np.random.rand() < p: return np.random.randint(*rand_range)-rand_range[1]//2
    return 0

class S2SItemList(MusicItemList):
    _bunch = MusicDataBunch
    def get(self, i):
        return MultitrackItem.from_idx(self.items[i], self.vocab)

# DATALOADING AND TRANSFORMATIONS
# These transforms happen on batch

def mask_tfm(b, mask_range, mask_idx, pad_idx, p=0.3):
    # mask range (min, max)
    # replacement vals - [x_replace, y_replace]. Usually [mask_idx, pad_idx]
    # p = replacement probability
    x,y = b
    x,y = x.clone(),y.clone()
    rand = torch.rand(x.shape, device=x.device)
    rand[x < mask_range[0]] = 1.0
    rand[x >= mask_range[1]] = 1.0
    
    # p(15%) of words are replaced. Of those p(15%) - 80% are masked. 10% wrong word. 10% unchanged
    y[rand > p] = pad_idx # pad unchanged 80%. Remove these from loss/acc metrics
    x[rand <= (p*.8)] = mask_idx # 80% = mask
    wrong_word = (rand > (p*.8)) & (rand <= (p*.9)) # 10% = wrong word
    x[wrong_word] = torch.randint(*mask_range, [wrong_word.sum().item()], device=x.device)
    return x, y

def mask_lm_tfm_default(b, vocab, mask_p=0.3):
    return mask_lm_tfm(b, mask_range=vocab.npenc_range, mask_idx=vocab.mask_idx, pad_idx=vocab.pad_idx, mask_p=mask_p)

def mask_lm_tfm_pitchdur(b, vocab, mask_p=0.9):
    mask_range = vocab.dur_range if np.random.rand() < 0.5 else vocab.note_range
    return mask_lm_tfm(b, mask_range=mask_range, mask_idx=vocab.mask_idx, pad_idx=vocab.pad_idx, mask_p=mask_p)

def mask_lm_tfm(b, mask_range, mask_idx, pad_idx, mask_p):
    x,y = b
    x_lm,x_pos = x[...,0], x[...,1]
    y_lm,y_pos = y[...,0], y[...,1]
    
    # Note: masking y_lm instead of x_lm. Just in case we ever do sequential s2s training
    x_msk, y_msk = mask_tfm((y_lm, y_lm), mask_range=mask_range, mask_idx=mask_idx, pad_idx=pad_idx, p=mask_p)
    msk_pos = y_pos
    
    x_dict = { 
        'msk': { 'x': x_msk, 'pos': msk_pos },
        'lm': { 'x': x_lm, 'pos': msk_pos }
    }
    y_dict = { 'msk': y_msk, 'lm': y_lm }
    return x_dict, y_dict

def melody_chord_tfm(b):
    m,m_pos,c,c_pos = b
    
    # offset x and y for next word prediction
    y_m = m[:,1:]
    x_m, m_pos = m[:,:-1], m_pos[:,:-1]
    
    y_c = c[:,1:]
    x_c, c_pos = c[:,:-1], c_pos[:,:-1]
    
    x_dict = { 
        'c2m': {
            'enc': x_c,
            'enc_pos': c_pos,
            'dec': x_m,
            'dec_pos': m_pos
        },
        'm2c': {
            'enc': x_m,
            'enc_pos': m_pos,
            'dec': x_c,
            'dec_pos': c_pos
        }
    }
    y_dict = {
        'c2m': y_m, 'm2c': y_c
    }
    return x_dict, y_dict
