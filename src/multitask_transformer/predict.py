# from .fastai_data import *
# from .encode_data import *
# from .music_transformer import *
# from fastai.basics import *
# from fastai.text.models.transformer import _line_shift, init_transformer
# from fastai.text.models.awd_lstm import *
# from fastai.text.models.transformer import *
# from fastai.callbacks.rnn import RNNTrainer
from fastai.basics import *
from ..vocab import *
from ..utils.top_k_top_p import top_k_top_p
from ..music_transformer.transform import *
from .model import get_multitask_model
from .dataloader import *

def multitask_model_learner(data:DataBunch, config:dict=None, drop_mult:float=1., pretrained:bool=False,
                        pretrained_fnames:OptStrTuple=None, **learn_kwargs) -> 'LanguageLearner':
    "Create a `Learner` with a language model from `data` and `arch`."
    vocab = data.vocab
    vocab_size = len(vocab)
    model = get_multitask_model(vocab_size, config=config, drop_mult=drop_mult, pad_idx=vocab.pad_idx)
    metrics = [AverageMultiMetric(partial(m, pad_idx=vocab.pad_idx)) for m in [mask_acc, lm_acc, c2m_acc, m2c_acc]]
    loss_func = MultiLoss(ignore_index=data.vocab.pad_idx)
    learn = MultitaskLearner(data, model, loss_func=loss_func, metrics=metrics, **learn_kwargs)
    return learn

class MultitaskLearner(Learner):
    def predict_nw(self, xb:Tensor, n_words:int=128,
                     temperatures:float=(1.0,1.0), min_bars=4,
                     top_k=30, top_p=0.6):
        "Return the `n_words` that come after `text`."
        self.model.reset()
        new_idx = []
        vocab = self.data.vocab
        xb = xb.squeeze()
        pos = torch.tensor(neg_position_enc(xb.cpu().numpy(), vocab), device=xb.device)
        last_pos = pos[-1]
        yb = torch.tensor([0])

        sep_count = 0
        bar_len = SAMPLE_FREQ * 4 # assuming 4/4 time
        vocab = self.data.vocab

        with torch.no_grad():
            for i in progress_bar(range(n_words), leave=True):
                batch = { 'lm': { 'x': xb[None], 'pos': pos[None] } }, yb
                res = self.pred_batch(batch=batch)['lm'][-1][-1]
                res = F.softmax(res, dim=-1)

                # bar = 16 beats
                if (sep_count // 16) <= min_bars: res[vocab.bos_idx] = 0.

                # Use first temperatures value if last prediction was duration
                temperature = temperatures[0] if (len(new_idx)==0 or vocab.is_duration(new_idx[-1])) else temperatures[1]
                if temperature != 1.: res.pow_(1 / temperature)

                res = top_k_top_p(res, top_k=top_k, top_p=top_p, filter_value=0)
                idx = torch.multinomial(res, 1).item()

                if new_idx and new_idx[-1]==vocab.sep_idx: 
                    duration = idx - vocab.dur_range[0]
                    sep_count += duration
                    last_pos = last_pos - duration # position is negative

                if idx==vocab.bos_idx: 
                    print('Predicted BOS token. Returning prediction...')
                    break

                new_idx.append(idx)
                xb = xb.new_tensor([idx])
                pos = pos.new_tensor([last_pos])
        return np.array(new_idx)

    def predict_mask(self, x:Tensor, pos=None,
                    temperatures:float=(1.0,1.0),
                    top_k=20, top_p=0.8):
        x = x.clone().squeeze()
        y = torch.tensor([0])
        vocab = self.data.vocab
        if pos is None:
            pos = torch.tensor(neg_position_enc(x.cpu().numpy(), vocab), device=x.device)
        self.model.reset()
        mask_idxs = (x == vocab.mask_idx).nonzero().view(-1)

        with torch.no_grad():
            for midx in progress_bar(mask_idxs, leave=True):

                # Using original positions, otherwise model gets too off track
        #         pos = torch.tensor(-position_enc(xb[0].cpu().numpy()), device=xb.device)[None]

                # Next Word
                res = self.pred_batch(batch=({ 'msk': { 'x': x[None], 'pos': pos[None] } }, y) )['msk'][0]
                res = F.softmax(res[midx], dim=-1) # task1, task2 - (bs x ts x vocab)

                # Don't allow any special tokens (as we are only removing notes and durations)
                res[vocab.bos_idx] = 0.
                res[vocab.sep_idx] = 0.
                res[vocab.stoi[EOS]] = 0

                # Use first temperatures value if last prediction was duration
                prev_idx = x[midx-1]
                temperature = temperatures[0] if vocab.is_duration(prev_idx) else temperatures[1]
                if temperature != 1.: res.pow_(1 / temperature)

                res = top_k_top_p(res, top_k=top_k, top_p=top_p, filter_value=0)
                idx = torch.multinomial(res, 1).item()
                #         idx = res.argmax()

                x[midx] = idx

        return x.cpu().numpy()


    def predict_s2s(self, xb_msk:Tensor, xb_lm:Tensor, n_words:int=128,
                    temperatures:float=(1.0,1.0),
                    top_k=30, top_p=0.8):
        self.model.reset()
        vocab = self.data.vocab
        x_lm = xb_lm.tolist()
        lm_pos = (neg_position_enc(xb_lm.cpu().numpy(), vocab)).tolist()
        last_pos = lm_pos[-1]

        msk_pos = torch.tensor(neg_position_enc(xb_msk.cpu().numpy(), vocab), device=xb_msk.device)
        x_enc = self.model.encoder(xb_msk.view(1, -1), msk_pos.view(1, -1))

        max_pos = msk_pos[-1] - SAMPLE_FREQ * 4

        with torch.no_grad():
            for i in progress_bar(range(n_words), leave=True):

                # Next Word
                x, pos = torch.tensor(x_lm, device=xb_lm.device)[None], torch.tensor(lm_pos, device=xb_lm.device)[None]
                dec = self.model.decoder(x, pos, x_enc) # all tasks include mask decoding
                res = F.softmax(self.model.head(dec), dim=-1)[-1, -1]

                # Use first temperatures value if last prediction was duration
                temperature = temperatures[0] if (len(x_lm)==0 or vocab.is_duration(x_lm[-1])) else temperatures[1]
                if temperature != 1.: res.pow_(1 / temperature)

                res = top_k_top_p(res, top_k=top_k, top_p=top_p, filter_value=0)
                idx = torch.multinomial(res, 1).item()
                #         idx = res.argmax()

                if idx == vocab.bos_idx | idx == vocab.stoi[EOS]: 
                    print('Predicting BOS/EOS')
                    break

                if x_lm and x_lm[-1]==vocab.sep_idx: 
                    duration = idx - vocab.dur_range[0]
                    last_pos = last_pos - duration # position is negative
                    if last_pos < max_pos:
                        print('Predicted past counter-part length. Returning early')
                        break

                lm_pos.append(last_pos)
                x_lm.append(idx)

        return np.array(x_lm)
    

# High level prediction functions from midi file

def s2s_predict_from_midi(learn, midi=None, n_words=200, 
                      temperatures=(1.0,1.0), top_k=24, top_p=0.7, seed_len=None, pred_melody=True, **kwargs):
    vocab = learn.data.vocab
    mpart, cpart = midi_extract_melody_chords(midi, vocab=vocab)
    
    x_np, y_np = (cpart, mpart) if pred_melody else (mpart, cpart)
    
    # if seed_len is passed, cutoff sequence so we can predict the rest
    y_cut = y_np if seed_len is None else trim_tfm(y_np, vocab=vocab, to_beat=seed_len)
    
    x, y = torch.tensor(x_np), torch.tensor(y_cut)
    if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
    pred = learn.predict_s2s(x, y, n_words=n_words, temperatures=temperatures, top_k=top_k, top_p=top_p)

    part_order = [pred, cpart] if pred_melody else [mpart, cpart]
    chordarr_comb = s2s_combine2chordarr(*part_order, vocab)

    return chordarr_comb

def nw_predict_from_midi(learn, midi=None, n_words=600, 
                      temperatures=(1.0,1.0), top_k=30, top_p=0.6, seed_len=None, **kwargs):
    vocab = learn.data.vocab
    try:
        seed_np = midi2idxenc(midi, vocab=vocab) # music21 can handle bytes directly
        if seed_len is not None:
            seed_np = trim_tfm(seed_np, vocab=vocab, to_beat=seed_len)
    except IndexError:
        # midi file has empty notes/tracks. Create empty stream
        seed_np = npenc2idxenc(np.zeros((0, 2), dtype=int))
    x = torch.tensor(seed_np)
    if torch.cuda.is_available(): x = x.cuda()
    pred = learn.predict_nw(x, n_words=n_words, temperatures=temperatures, top_k=top_k, top_p=top_p)
    return np.concatenate((seed_np,pred), axis=0)

def mask_predict_from_midi(learn, midi=None,
                           temperatures=(1.0,1.0), top_k=30, top_p=0.7, 
                           predict_notes=True, **kwargs):
    seed_np = midi2idxenc(midi, vocab=learn.data.vocab) # music21 can handle bytes directly
    x = torch.tensor(seed_np)
    vocab = learn.data.vocab
    pos = torch.tensor(neg_position_enc(x.cpu().numpy(), vocab), device=x.device)
    mask_range = vocab.note_range if predict_notes else vocab.dur_range
    x_msk = mask_input(x, mask_range=mask_range, replacement_idx=vocab.mask_idx)
    if torch.cuda.is_available(): 
        x_msk = x_msk.cuda()
        pos = pos.cuda()
    pred = learn.predict_mask(x_msk, pos, temperatures=temperatures, top_k=top_k, top_p=top_p)
    return pred

# Utility for predictions
def mask_input(xb, mask_range, replacement_idx, clone=True):
    if clone: xb = xb.clone()
    xb[(xb >= mask_range[0]) & (xb < mask_range[1])] = replacement_idx
    return xb


# LOSS AND METRICS

class MultiLoss():
    def __init__(self, ignore_index=None):
        "Loss mult - Mask, NextWord, Seq2Seq"
        self.loss = CrossEntropyFlat(ignore_index=ignore_index)
        
    def __call__(self, inputs:Dict[str,Tensor], targets:Dict[str,Tensor])->Rank0Tensor:
        losses = [self.loss(inputs[key], target) for key,target in targets.items()]
        return sum(losses)
    
def acc_ignore_pad(input:Tensor, targ:Tensor, pad_idx)->Rank0Tensor:
    if input is None or targ is None: return None
    n = targ.shape[0]
    input = input.argmax(dim=-1).view(n,-1)
    targ = targ.view(n,-1)
    mask = targ != pad_idx
    return (input[mask]==targ[mask]).float().mean()

def acc_index(inputs, targets, key, pad_idx):
    return acc_ignore_pad(inputs.get(key), targets.get(key), pad_idx)
    
def mask_acc(inputs, targets, pad_idx): return acc_index(inputs, targets, 'lm', pad_idx)
def lm_acc(inputs, targets, pad_idx): return acc_index(inputs, targets, 'msk', pad_idx)
def c2m_acc(inputs, targets, pad_idx): return acc_index(inputs, targets, 'c2m', pad_idx)
def m2c_acc(inputs, targets, pad_idx): return acc_index(inputs, targets, 'm2c', pad_idx)


class AverageMultiMetric(AverageMetric):
    "Updated fastai.AverageMetric to support multi task metrics."
    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        if not is_listy(last_target): last_target=[last_target]
        val = self.func(last_output, *last_target)
        if val is None: return
        self.count += first_el(last_target).size(0)
        if self.world:
            val = val.clone()
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            val /= self.world
        self.val += first_el(last_target).size(0) * val.detach().cpu()

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        if self.count == 0: return add_metrics(last_metrics, 0)
        return add_metrics(last_metrics, self.val/self.count)
    

# MODEL LOADING
class MLMTrainer(LearnerCallback):
    "`Callback` that regroups lr adjustment to seq_len, AR and TAR."
    def __init__(self, learn:Learner, dataloaders=None, starting_mask_window=1):
        super().__init__(learn)
        self.count = 1
        self.mw_start = starting_mask_window
        self.dataloaders = dataloaders

    def on_epoch_begin(self, **kwargs):
        "Reset the hidden state of the model."
        model = get_model(self.learn.model)
        model.reset()
        model.encoder.mask_size = max(self.count+self.mw_start, 100)
        
    def on_epoch_end(self, last_metrics, **kwargs):
        "Finish the computation and sends the result to the Recorder."
        if self.dataloaders is not None: 
            self.learn.data = self.dataloaders[self.count % len(self.dataloaders)]
        self.count += 1

