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

    if pretrained_path: 
        state = torch.load(pretrained_path, map_location='cpu')
        if config is None: config = state['config']

    model = get_multitask_model(vocab_size, config=config, drop_mult=drop_mult, pad_idx=vocab.pad_idx)
    metrics = [AverageMultiMetric(partial(m, pad_idx=vocab.pad_idx)) for m in [mask_acc, lm_acc, c2m_acc, m2c_acc]]
    loss_func = MultiLoss(ignore_index=data.vocab.pad_idx)
    learn = MultitaskLearner(data, model, loss_func=loss_func, metrics=metrics, **learn_kwargs)
    
    if pretrained_path: 
        get_model(model).load_state_dict(state['model'], strict=False)
        if not hasattr(learn, 'opt'): learn.create_opt(defaults.lr, learn.wd)
        try:    learn.opt.load_state_dict(state['opt'])
        except: pass
        del state
        gc.collect()
        
    return learn

class MultitaskLearner(Learner):
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

        start_pos = last_pos

        sep_count = 0
        bar_len = SAMPLE_FREQ * 4 # assuming 4/4 time
        vocab = self.data.vocab

        repeat_count = 0

        for i in progress_bar(range(n_words), leave=True):
            batch = { 'lm': { 'x': x[None], 'pos': pos[None] } }, y
            logits = self.pred_batch(batch=batch)['lm'][-1][-1]

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

    def predict_mask(self, masked_item:MusicItem,
                    temperatures:float=(1.0,1.0),
                    top_k=20, top_p=0.8):
        x = masked_item.to_tensor()
        pos = masked_item.get_pos_tensor()
        y = torch.tensor([0])
        vocab = self.data.vocab
        self.model.reset()
        mask_idxs = (x == vocab.mask_idx).nonzero().view(-1)

        repeat_count = 0

        for midx in progress_bar(mask_idxs, leave=True):
            prev_idx = x[midx-1]

            # Using original positions, otherwise model gets too off track
            # pos = torch.tensor(-position_enc(xb[0].cpu().numpy()), device=xb.device)[None]
    
            # Next Word
            logits = self.pred_batch(batch=({ 'msk': { 'x': x[None], 'pos': pos[None] } }, y) )['msk'][0][midx]

            # Temperature
            # Use first temperatures value if last prediction was duration
            temperature = temperatures[0] if vocab.is_duration_or_pad(prev_idx) else temperatures[1]
            repeat_penalty = max(0, np.log((repeat_count+1)/4)/5) * temperature
            temperature += repeat_penalty
            if temperature != 1.: logits = logits / temperature

            # Filter
            filter_value = -float('Inf')
            special_idxs = [vocab.bos_idx, vocab.sep_idx, vocab.stoi[EOS]]
            logits[special_idxs] = filter_value # Don't allow any special tokens (as we are only removing notes and durations)
            logits = filter_invalid_indexes(logits, prev_idx, vocab, filter_value=filter_value)
            logits = top_k_top_p(logits, top_k=top_k, top_p=top_p, filter_value=filter_value)

            # Sampling
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()

            # Update repeat count
            num_choices = len(probs.nonzero().view(-1))
            if num_choices <= 2: repeat_count += 1
            else: repeat_count = repeat_count // 2

            x[midx] = idx

        return vocab.to_music_item(x.cpu().numpy())

    def predict_s2s(self, input_item:MusicItem, target_item:MusicItem, n_words:int=256,
                        temperatures:float=(1.0,1.0), top_k=30, top_p=0.8,
                        use_memory=True):
        vocab = self.data.vocab
        
        # Input doesn't change. We can reuse the encoder output on each prediction
        with torch.no_grad():
            inp, inp_pos = input_item.to_tensor(), input_item.get_pos_tensor()
            x_enc = self.model.encoder(inp[None], inp_pos[None])
        
        # target
        targ = target_item.data.tolist()
        targ_pos = target_item.position.tolist()
        last_pos = targ_pos[-1]
        self.model.reset()

        repeat_count = 0

        max_pos = input_item.position[-1] + SAMPLE_FREQ * 4 # Only predict until both tracks/parts have the same length
        x, pos = inp.new_tensor(targ), inp_pos.new_tensor(targ_pos)
        
        for i in progress_bar(range(n_words), leave=True):
            # Predict
            with torch.no_grad():
                dec = self.model.decoder(x[None], pos[None], x_enc)
                logits = self.model.head(dec)[-1, -1]

            # Temperature
            # Use first temperatures value if last prediction was duration
            prev_idx = targ[-1] if len(targ) else vocab.pad_idx
            temperature = temperatures[0] if vocab.is_duration_or_pad(prev_idx) else temperatures[1]
            repeat_penalty = max(0, np.log((repeat_count+1)/4)/5) * temperature
            temperature += repeat_penalty
            if temperature != 1.: logits = logits / temperature
                
            # Filter
            filter_value = -float('Inf')
            logits = filter_invalid_indexes(logits, prev_idx, vocab, filter_value=filter_value)
            logits = top_k_top_p(logits, top_k=top_k, top_p=top_p, filter_value=filter_value)

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()

            # Update repeat count
            num_choices = len(probs.nonzero().view(-1))
            if num_choices <= 2: repeat_count += 1
            else: repeat_count = repeat_count // 2

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
    targ = targ.remove_eos()
        
    pred = learn.predict_s2s(inp, targ, n_words=n_words, temperatures=temperatures, top_k=top_k, top_p=top_p, **kwargs)
    
    part_order = (pred, inp) if pred_melody else (inp, pred)
    return MultitrackItem(*part_order)

def mask_predict_from_midi(learn, midi=None, predict_notes=True,
                           temperatures=(1.0,1.0), top_k=30, top_p=0.7, section=None, **kwargs):
    item = MusicItem.from_file(midi, learn.data.vocab)
    masked_item = item.mask_pitch(section) if predict_notes else item.mask_duration(section)
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
        model.encoder.mask_steps = max(self.count+self.mw_start, 100)
        
    def on_epoch_end(self, last_metrics, **kwargs):
        "Finish the computation and sends the result to the Recorder."
        if self.dataloaders is not None: 
            self.learn.data = self.dataloaders[self.count % len(self.dataloaders)]
        self.count += 1

