from fastai.basics import *
from ..vocab import *
from ..utils.top_k_top_p import top_k_top_p
from ..utils.midifile import is_empty_midi
from ..music_transformer.transform import *
from ..music_transformer.learner import filter_invalid_indexes
from .model import get_multitask_model
from .dataloader import *

def multitask_model_learner(data:DataBunch, config:dict=None, drop_mult:float=1., 
                            pretrained_path:PathOrStr=None, **learn_kwargs) -> 'LanguageLearner':
    "Create a `Learner` with a language model from `data` and `arch`."
    vocab = data.vocab
    vocab_size = len(vocab)
    model = get_multitask_model(vocab_size, config=config, drop_mult=drop_mult, pad_idx=vocab.pad_idx)
    metrics = [AverageMultiMetric(partial(m, pad_idx=vocab.pad_idx)) for m in [mask_acc, lm_acc, c2m_acc, m2c_acc]]
    loss_func = MultiLoss(ignore_index=data.vocab.pad_idx)
    learn = MultitaskLearner(data, model, loss_func=loss_func, metrics=metrics, **learn_kwargs)
    
    if pretrained_path:
        state = torch.load(pretrained_path, map_location='cpu')
        get_model(model).load_state_dict(state['model'], strict=False)
        
    return learn

class MultitaskLearner(Learner):
    def predict_nw(self, item:MusicItem, n_words:int=128,
                     temperatures:float=(1.0,1.0), min_bars=4,
                     top_k=30, top_p=0.6):
        "Return the `n_words` that come after `text`."
        self.model.reset()
        new_idx = []
        vocab = self.data.vocab
        x, pos = item.to_tensor(), item.get_pos_tensor()
        last_pos = pos[-1] if len(pos) else 0
        y = torch.tensor([0])

        sep_count = 0
        bar_len = SAMPLE_FREQ * 4 # assuming 4/4 time
        vocab = self.data.vocab

        with torch.no_grad():
            for i in progress_bar(range(n_words), leave=True):
                batch = { 'lm': { 'x': x[None], 'pos': pos[None] } }, y
                res = self.pred_batch(batch=batch)['lm'][-1][-1]
                res = F.softmax(res, dim=-1)

                # bar = 16 beats
                if (sep_count // 16) <= min_bars: res[vocab.bos_idx] = 0.

                # Use first temperatures value if last prediction was duration
                temperature = temperatures[0] if (len(new_idx)==0 or vocab.is_duration(new_idx[-1])) else temperatures[1]
                if temperature != 1.: res.pow_(1 / temperature)

                prev_idx = new_idx[-1] if len(new_idx) else -1
                res = filter_invalid_indexes(res, prev_idx, vocab)
                res = top_k_top_p(res, top_k=top_k, top_p=top_p, filter_value=0)
                idx = torch.multinomial(res, 1).item()

                if prev_idx==vocab.sep_idx: 
                    duration = idx - vocab.dur_range[0]
                    sep_count += duration
                    last_pos = last_pos + duration

                if idx==vocab.bos_idx: 
                    print('Predicted BOS token. Returning prediction...')
                    break

                new_idx.append(idx)
                x = x.new_tensor([idx])
                pos = pos.new_tensor([last_pos])

        pred = vocab.to_music_item(np.array(new_idx))
        full = item.append(pred)
        return pred, full

    def predict_mask(self, masked_item:MusicItem, pos=None,
                    temperatures:float=(1.0,1.0),
                    top_k=20, top_p=0.8):
        x = masked_item.to_tensor()
        pos = masked_item.get_pos_tensor()
        y = torch.tensor([0])
        vocab = self.data.vocab
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

                prev_idx = x[midx-1]
                res = filter_invalid_indexes(res, prev_idx, vocab)

                # Use first temperatures value if last prediction was duration
                temperature = temperatures[0] if vocab.is_duration(prev_idx) else temperatures[1]
                if temperature != 1.: res.pow_(1 / temperature)

                res = top_k_top_p(res, top_k=top_k, top_p=top_p, filter_value=0)
                idx = torch.multinomial(res, 1).item()
                #         idx = res.argmax()

                x[midx] = idx

        return vocab.to_music_item(x.cpu().numpy())

    def predict_s2s(self, input_item:MusicItem, target_item:MusicItem, n_words:int=256,
                        temperatures:float=(1.0,1.0), top_k=30, top_p=0.8,
                        use_memory=True):
        vocab = self.data.vocab
        
        # Input doesn't change. We can reuse the encoder output on each prediction
        inp, inp_pos = input_item.to_tensor(), input_item.get_pos_tensor()
        x_enc = self.model.encoder(inp[None], inp_pos[None])
        
        # target
        targ = target_item.data.tolist()
        targ_pos = target_item.position.tolist()
        last_pos = targ_pos[-1]
        self.model.reset()

        max_pos = input_item.position[-1] + SAMPLE_FREQ * 4 # Only predict until both tracks/parts have the same length
        x, pos = inp.new_tensor(targ), inp_pos.new_tensor(targ_pos)
        
        with torch.no_grad():
            for i in progress_bar(range(n_words), leave=True):
                dec = self.model.decoder(x[None], pos[None], x_enc)
                res = F.softmax(self.model.head(dec), dim=-1)[-1, -1]

                # Use first temperatures value if last prediction was duration
                temperature = temperatures[0] if (len(targ)==0 or vocab.is_duration(targ[-1])) else temperatures[1]
                if temperature != 1.: res.pow_(1 / temperature)
                    
                prev_idx = targ[-1] if len(targ) else -1
                res = filter_invalid_indexes(res, prev_idx, vocab)
                res = top_k_top_p(res, top_k=top_k, top_p=top_p, filter_value=0)
                idx = torch.multinomial(res, 1).item()
                #         idx = res.argmax()

                if idx == vocab.bos_idx | idx == vocab.stoi[EOS]: 
                    print('Predicting BOS/EOS')
                    break

                if prev_idx == vocab.sep_idx: 
                    duration = idx - vocab.dur_range[0]
                    last_pos = last_pos + duration
                    if last_pos > max_pos:
                        print('Predicted past counter-part length. Returning early')
                        break

                targ_pos.append(last_pos)
                targ.append(idx)
                
                if use_memory:
                    # Relying on memory for kv. Only need last prediction index
                    x, pos = inp.new_tensor([targ[-1]]), inp_pos.new_tensor([targ_pos[-1]])
                else:
                    # Reset memory after each prediction, since we feeding the whole sequence every time
                    self.model.reset()
                    x, pos = inp.new_tensor(targ), inp_pos.new_tensor(targ_pos)

        return vocab.to_music_item(np.array(targ))
    
# High level prediction functions from midi file
def nw_predict_from_midi(learn, midi=None, n_words=400, 
                      temperatures=(1.0,1.0), top_k=30, top_p=0.6, seed_len=None, **kwargs):
    vocab = learn.data.vocab
    seed = MusicItem.from_file(midi, vocab) if not is_empty_midi(midi) else MusicItem.empty(vocab)
    if seed_len is not None: seed = seed.trim_to_beat(seed_len)
        
    pred, full = learn.predict_nw(seed, n_words=n_words, temperatures=temperatures, top_k=top_k, top_p=top_p, **kwargs)
    return full

def s2s_predict_from_midi(learn, midi=None, n_words=200, 
                      temperatures=(1.0,1.0), top_k=24, top_p=0.7, seed_len=None, pred_melody=True, **kwargs):
    multitrack_item = MultitrackItem.from_file(midi, learn.data.vocab)
    melody, chords = multitrack_item.melody, multitrack_item.chords
    
    inp, targ = (chords, melody) if pred_melody else (melody, chords)
    
    # if seed_len is passed, cutoff sequence so we can predict the rest
    if seed_len is not None: targ = targ.trim_to_beat(seed_len)
        
    pred = learn.predict_s2s(inp, targ, n_words=n_words, temperatures=temperatures, top_k=top_k, top_p=top_p, **kwargs)
    
    part_order = (pred, inp) if pred_melody else (inp, pred)
    return MultitrackItem(*part_order)

def mask_predict_from_midi(learn, midi=None, predict_notes=True,
                           temperatures=(1.0,1.0), top_k=30, top_p=0.7, **kwargs):
    item = MusicItem.from_file(midi, learn.data.vocab)
    masked_item = item.mask_notes() if predict_notes else item.mask_duration()
    pred = learn.predict_mask(masked_item, temperatures=temperatures, top_k=top_k, top_p=top_p, **kwargs)
    return pred

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
    
def mask_acc(inputs, targets, pad_idx): return acc_index(inputs, targets, 'msk', pad_idx)
def lm_acc(inputs, targets, pad_idx): return acc_index(inputs, targets, 'lm', pad_idx)
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
class MTTrainer(LearnerCallback):
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

