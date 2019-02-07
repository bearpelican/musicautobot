from fastai.text import *
from fastai.callbacks.tracker import *
from fastai.callbacks.rnn import RNNTrainer
from fastai.distributed import *
from fastai_data import *
from transformer_xl import default_txl

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='data/midi/midi_transcribe_v3_shortdur/')
parser.add_argument('--cache', type=str, default='tmp_clc')
parser.add_argument('--save', type=str, default='first_run')
parser.add_argument('--load', type=str, default=None)
parser.add_argument("--local_rank", type=int)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--bptt", type=int, default=500)
parser.add_argument('--half', action='store_true', help='Use half precision')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay for adam')
parser.add_argument('--epochs', type=int, default=5, help='num epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
args = parser.parse_args()

if args.local_rank != 0:
    f = open('/dev/null', 'w')
    sys.stdout = f
    
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

bs=args.batch_size
bptt=args.bptt
path = Path(args.path)
data = TextLMDataBunch.load(path, cache_name=args.cache, bs=bs, bptt=bptt)
data.valid_ds.x.processor[0] = TokenizeProcessor(tokenizer=MusicTokenizer())

vocab = data.train_ds.vocab
vocab_size = len(vocab.itos)

model = default_txl.get_default_model(vocab_size).cuda()
model.reset = lambda: None

class TXLTrainer(LearnerCallback):
    def __init__(self, learn:Learner):
        super().__init__(learn)
        self.mems = None

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        # model expects bptt, batch_size
        return (last_input, self.mems), last_target
        
    def on_loss_begin(self, last_output:Tensor, train, **kwargs:Any) -> Tensor:
        out, self.mems = last_output
        return out

full_clip = None if args.half else 0.3
learn = LanguageLearner(data, model, bptt, clip=full_clip)
if args.load:
    load_path = Path(args.path)/args.load
    state = torch.load(load_path, map_location='cpu')
    get_model(learn.model).load_state_dict(state['model'], strict=True)
    learn.model.cuda()
if args.save:
    save_path = Path(args.path)/learn.model_dir/args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
learn.callbacks = [c for c in learn.callbacks if not isinstance(c, RNNTrainer)]
learn.callbacks.append(TXLTrainer(learn))
if args.half: learn = learn.to_fp16(clip=.4)
learn = learn.distributed(args.local_rank, drop_last=True)
if args.local_rank == 0: learn.callbacks.append(SaveModelCallback(learn, name=f'{args.save}_best'))

learn.fit_one_cycle(args.epochs, args.lr, div_factor=30, moms=(0.7,0.5))

if args.local_rank == 0: learn.save(f'{args.save}')
