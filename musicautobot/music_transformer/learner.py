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

    if pretrained_path: 
        state = torch.load(pretrained_path, map_location='cpu')
        if config is None: config = state['config']
        
    model = get_language_model(arch, len(data.vocab.itos), config=config, drop_mult=drop_mult)
    learn = MusicLearner(data, model, split_func=meta['split_lm'], **learn_kwargs)

    if pretrained_path: 
        get_model(model).load_state_dict(state['model'], strict=False)
        if not hasattr(learn, 'opt'): learn.create_opt(defaults.lr, learn.wd)
        try:    learn.opt.load_state_dict(state['opt'])
        except: pass
        del state
        gc.collect()

    return learn

# Predictions
from fastai import basic_train # for predictions
class MusicLearner(LanguageLearner):
    def save(self, file:PathLikeOrBinaryStream=None, with_opt:bool=True, config=None):
        "Save model and optimizer state (if `with_opt`) with `file` to `self.model_dir`. `file` can be file-like (file or buffer)"
        out_path = super().save(file, return_path=True, with_opt=with_opt)
        if config and out_path:
            state = torch.load(out_path)
            state['config'] = config
            torch.save(state, out_path)
            del state
            gc.collect()
        return out_path

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
                     top_k=30, top_p=0.6):
        "Return the `n_words` that come after `text`."
        self.model.reset()
        new_idx = []
        vocab = self.data.vocab
        x, pos = item.to_tensor(), item.get_pos_tensor()
        last_pos = pos[-1] if len(pos) else 0
        y = torch.tensor([0])

        start_pos = last_pos

        sep_count = 0
        bar_len = SAMPLE_FREQ * 4 # assuming 4/4 time
        vocab = self.data.vocab

        repeat_count = 0
        if hasattr(self.model[0], 'encode_position'):
            encode_position = self.model[0].encode_position
        else: encode_position = False

        for i in progress_bar(range(n_words), leave=True):
            with torch.no_grad():
                if encode_position:
                    batch = { 'x': x[None], 'pos': pos[None] }
                    logits = self.model(batch)[0][-1][-1]
                else:
                    logits = self.model(x[None])[0][-1][-1]

            prev_idx = new_idx[-1] if len(new_idx) else vocab.pad_idx

            # Temperature
            # Use first temperatures value if last prediction was duration
            temperature = temperatures[0] if vocab.is_duration_or_pad(prev_idx) else temperatures[1]
            repeat_penalty = max(0, np.log((repeat_count+1)/4)/5) * temperature
            temperature += repeat_penalty
            if temperature != 1.: logits = logits / temperature
                

            # Filter
            # bar = 16 beats
            filter_value = -float('Inf')
            if ((last_pos - start_pos) // 16) <= min_bars: logits[vocab.bos_idx] = filter_value

            logits = filter_invalid_indexes(logits, prev_idx, vocab, filter_value=filter_value)
            logits = top_k_top_p(logits, top_k=top_k, top_p=top_p, filter_value=filter_value)
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()

            # Update repeat count
            num_choices = len(probs.nonzero().view(-1))
            if num_choices <= 2: repeat_count += 1
            else: repeat_count = repeat_count // 2

            if prev_idx==vocab.sep_idx: 
                duration = idx - vocab.dur_range[0]
                last_pos = last_pos + duration

                bars_pred = (last_pos - start_pos) // 16
                abs_bar = last_pos // 16
                # if (bars % 8 == 0) and (bars_pred > min_bars): break
                if (i / n_words > 0.80) and (abs_bar % 4 == 0): break


            if idx==vocab.bos_idx: 
                print('Predicted BOS token. Returning prediction...')
                break

            new_idx.append(idx)
            x = x.new_tensor([idx])
            pos = pos.new_tensor([last_pos])

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
