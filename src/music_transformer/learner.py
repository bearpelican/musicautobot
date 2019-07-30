from fastai.basics import *
from fastai.text.learner import LanguageLearner, get_language_model, _model_meta
from .model import *
from .transform import MusicItem
from ..numpy_encode import SAMPLE_FREQ
from ..utils.top_k_top_p import top_k_top_p
from ..utils.midifile import is_empty_midi

def music_model_learner(data:DataBunch, config:dict=None, drop_mult:float=1.,
                        pretrained_path:PathOrStr=None, **learn_kwargs) -> 'LanguageLearner':
    "Create a `Learner` with a language model from `data` and `arch`."
    _model_meta[MusicTransformerXL] = _model_meta[TransformerXL] # copy over fastai's model metadata
    meta = _model_meta[MusicTransformerXL]
    model = get_language_model(MusicTransformerXL, len(data.vocab.itos), config=config, drop_mult=drop_mult)
    learn = MusicLearner(data, model, split_func=meta['split_lm'], **learn_kwargs)

    if pretrained_path:
        state = torch.load(pretrained_path, map_location='cpu')
        get_model(model).load_state_dict(state['model'], strict=False)
        
    return learn

# Predictions
from fastai import basic_train # for predictions
class MusicLearner(LanguageLearner):
    def beam_search(self, xb:Tensor, n_words:int, top_k:int=10, beam_sz:int=10, temperature:float=1.,
                    ):
        "Return the `n_words` that come after `text` using beam search."
        self.model.reset()
        self.model.eval()
        xb_length = xb.shape[-1]
        if xb.shape[0] > 1: xb = xb[0][None]
        yb = torch.ones_like(xb)

        nodes = None
        xb = xb.repeat(top_k, 1)
        nodes = xb.clone()
        scores = xb.new_zeros(1).float()
        with torch.no_grad():
            for k in progress_bar(range(n_words), leave=False):
                out = F.log_softmax(self.model(xb)[0][:,-1], dim=-1)
    #             if no_unk: out[:,self.data.vocab.stoi[UNK]] = -float('Inf')
                values, indices = out.topk(top_k, dim=-1)
                scores = (-values + scores[:,None]).view(-1)
                indices_idx = torch.arange(0,nodes.size(0))[:,None].expand(nodes.size(0), top_k).contiguous().view(-1)
                sort_idx = scores.argsort()[:beam_sz]
                scores = scores[sort_idx]
                nodes = torch.cat([nodes[:,None].expand(nodes.size(0),top_k,nodes.size(1)),
                                indices[:,:,None].expand(nodes.size(0),top_k,1),], dim=2)
                nodes = nodes.view(-1, nodes.size(2))[sort_idx]
                self.model[0].select_hidden(indices_idx[sort_idx])
                xb = nodes[:,-1][:,None]
        if temperature != 1.: scores.div_(temperature)
        node_idx = torch.multinomial(torch.exp(-scores), 1).item()
        return [i.item() for i in nodes[node_idx][xb_length:] ]

    def predict(self, item:MusicItem, n_words:int=128,
                     temperatures:float=(1.0,1.0), min_bars=4,
                     top_k=40, top_p=0.9):
        "Return the `n_words` that come after `text`."
        self.model.reset()
        y = torch.tensor([0])
        new_idx = []

        sep_count = 0

        bar_len = SAMPLE_FREQ * 4 # assuming 4/4 time
        vocab = self.data.vocab
        x = item.to_tensor()
        with torch.no_grad():
            for i in progress_bar(range(n_words), leave=True):

                res = self.pred_batch(batch=(x[None],y))[0][-1]

                # bar = 16 beats
                if (sep_count // 16) <= min_bars: res[vocab.bos_idx] = 0.

                # Use first temperatures value if last prediction was duration
                temperature = temperatures[0] if (len(new_idx)==0 or self.data.vocab.is_duration(new_idx[-1])) else temperatures[1]
                if temperature != 1.: res.pow_(1 / temperature)

                prev_idx = new_idx[-1] if len(new_idx) else x[-1].item()
                res = filter_invalid_indexes(res, prev_idx, vocab)
                res = top_k_top_p(res, top_k=top_k, top_p=top_p, filter_value=0)
                idx = torch.multinomial(res, 1).item()

                if new_idx and new_idx[-1]==vocab.sep_idx: 
                    duration = idx - vocab.dur_range[0]
                    sep_count += duration
                    # print('Bars', duration, sep_count // 16)

                if idx==vocab.bos_idx: 
                    print('Predicted BOS token. Returning prediction...')
                    break


                new_idx.append(idx)
                x = x.new_tensor([idx])
        return vocab.to_music_item(np.array(new_idx))
    
def nw_predict_from_midi(learn, midi=None, n_words=400, 
                      temperatures=(1.0,1.0), top_k=30, top_p=0.6, seed_len=None, **kwargs):
    vocab = learn.data.vocab
    seed = MusicItem.from_file(midi, vocab) if not is_empty_midi(midi) else MusicItem.empty(vocab)
    if seed_len is not None: seed = seed.trim_to_beat(seed_len)
        
    pred = learn.predict_nw(seed, n_words=n_words, temperatures=temperatures, top_k=top_k, top_p=top_p, **kwargs)
    return seed.append(pred)

def filter_invalid_indexes(res, prev_idx, vocab):
    if vocab.is_duration(prev_idx) or prev_idx == vocab.pad_idx:
        res[list(range(*vocab.dur_range))] = 0
    else:
        res[list(range(*vocab.note_range))] = 0
    return res
