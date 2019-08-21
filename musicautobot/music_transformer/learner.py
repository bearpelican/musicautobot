from fastai.basics import *
from fastai.text.learner import LanguageLearner, get_language_model, _model_meta
from .model import *
from .transform import MusicItem
from ..numpy_encode import SAMPLE_FREQ
from ..utils.top_k_top_p import top_k_top_p
from ..utils.midifile import is_empty_midi

_model_meta[MusicTransformerXL] = _model_meta[TransformerXL] # copy over fastai's model metadata

def music_model_learner(data:DataBunch, arch=MusicTransformerXL, config:dict=None, drop_mult:float=1.,
                        pretrained_path:PathOrStr=None, **learn_kwargs) -> 'LanguageLearner':
    "Create a `Learner` with a language model from `data` and `arch`."
    meta = _model_meta[arch]
    model = get_language_model(arch, len(data.vocab.itos), config=config, drop_mult=drop_mult)
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
                # Predict
                with torch.no_grad():
                    logits = self.model(x[None])[0][-1]
                # res = self.pred_batch(batch=(x[None],y))[0][-1] # returns softmax values which we don't wnat

                # Temperature
                # Use first temperatures value if last prediction was duration
                prev_idx = new_idx[-1] if len(new_idx) else x[-1].item()
                temperature = temperatures[0] if vocab.is_duration_or_pad(prev_idx) else temperatures[1]
                if temperature != 1.: logits = logits / temperature

                # Filter
                filter_value = -float('Inf')
                # bar = 16 beats
                if (sep_count // 16) <= min_bars: logits[vocab.bos_idx] = filter_value
                logits = filter_invalid_indexes(logits, prev_idx, vocab, filter_value=filter_value)
                logits = top_k_top_p(logits, top_k=top_k, top_p=top_p, filter_value=filter_value)

                # Sample
                probs = F.softmax(logits, dim=-1)
                idx = torch.multinomial(probs, 1).item()

                if new_idx and new_idx[-1]==vocab.sep_idx: 
                    duration = idx - vocab.dur_range[0]
                    sep_count += duration
                    # print('Bars', duration, sep_count // 16)

                if idx==vocab.bos_idx: 
                    print('Predicted BOS token. Returning prediction...')
                    break


                new_idx.append(idx)
                x = x.new_tensor([idx])
        pred = vocab.to_music_item(np.array(new_idx))
        full = item.append(pred)
        return pred, full
    
# High level prediction functions from midi file
def predict_from_midi(learn, midi=None, n_words=400, 
                      temperatures=(1.0,1.0), top_k=30, top_p=0.6, seed_len=None, **kwargs):
    vocab = learn.data.vocab
    seed = MusicItem.from_file(midi, vocab) if not is_empty_midi(midi) else MusicItem.empty(vocab)
    if seed_len is not None: seed = seed.trim_to_beat(seed_len)

    pred, full = learn.predict(seed, n_words=n_words, temperatures=temperatures, top_k=top_k, top_p=top_p, **kwargs)
    return full

def filter_invalid_indexes(res, prev_idx, vocab, filter_value=-float('Inf')):
    if vocab.is_duration_or_pad(prev_idx):
        res[list(range(*vocab.dur_range))] = filter_value
    else:
        res[list(range(*vocab.note_range))] = filter_value
    return res
