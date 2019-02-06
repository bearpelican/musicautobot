import argparse
from .mem_transformer import MemTransformerLM
from .utils import *
from functools import partial
import torch

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--n_layer', type=int, default=12, help='number of total layers')
parser.add_argument('--n_head', type=int, default=10, help='number of heads')
parser.add_argument('--d_head', type=int, default=50, help='head dimension')
parser.add_argument('--d_embed', type=int, default=400, help='embedding dimension')
parser.add_argument('--d_model', type=int, default=400, help='model dimension')
parser.add_argument('--d_inner', type=int, default=1000, help='inner dimension in FF')
parser.add_argument('--dropout', type=float, default=0.1, help='global dropout rate')
parser.add_argument('--dropatt', type=float, default=0, help='attention probability dropout rate')
parser.add_argument('--init', default='normal', type=str, help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str, help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.1, help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range', type=float, default=0.01, help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std', type=float, default=0.02, help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std', type=float, default=0.01, help='parameters initialized by N(0, init_std)')
parser.add_argument('--batch_chunk', type=int, default=1, help='split batch into chunks to save memory')
parser.add_argument('--tgt_len', type=int, default=150, help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=150, help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len', type=int, default=0, help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=150, help='length of the retained previous heads')
parser.add_argument('--not_tied', action='store_true', help='do not tie the word embedding and softmax weights')
parser.add_argument('--adaptive', action='store_true', help='use adaptive softmax')
parser.add_argument('--div_val', type=int, default=1, help='divident value for adapative input and softmax')
parser.add_argument('--pre_lnorm', action='store_true', help='apply LayerNorm to the input instead of the output')
parser.add_argument('--varlen', action='store_true', help='use variable length')
parser.add_argument('--same_length', action='store_true', help='use the same attn length for all tokens')
parser.add_argument('--attn_type', type=int, default=0, 
                    help='attention type. 0 for ours, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.')
parser.add_argument('--clamp_len', type=int, default=-1, help='use the same pos embeddings after clamp_len')
parser.add_argument('--max_eval_steps', type=int, default=-1, help='max eval steps')
parser.add_argument('--sample_softmax', type=int, default=-1, help='number of samples in sampled softmax')
parser.add_argument('--patience', type=int, default=0, help='patience')
parser.add_argument('--fp16', action='store_true', help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')

def get_default_model(vocab_size, additional_args=None):
    if additional_args is None: additional_args = []
    args = parser.parse_args(additional_args)
    args.tied = not args.not_tied
    cutoffs, tie_projs = [], [False]

    mem_model = MemTransformerLM(vocab_size, args.n_layer, args.n_head, args.d_model,
        args.d_head, args.d_inner, args.dropout, args.dropatt,
        tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
        tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
        ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
        same_length=args.same_length, attn_type=args.attn_type,
        clamp_len=args.clamp_len, sample_softmax=args.sample_softmax)

    mem_model.apply(partial(weights_init, args=args))
    mem_model.word_emb.apply(partial(weights_init, args=args)) # ensure embedding init is not overridden by out_layer in case of weight sharing

    # Applying linear head instead of hiding it in crit like original code
    class TransformerLM(nn.Module):
        def __init__(self, mem_model):
            super().__init__()
            self.mem_model = mem_model
            if args.d_embed != args.d_model: 
                self.linear = torch.nn.Linear(in_features=args.d_model, out_features=args.d_embed)
            else: self.linear = None
            self.final = torch.nn.Linear(in_features=args.d_embed, out_features=vocab_size)
            self.final.weight = mem_model.word_emb.emb_layers[0].weight
        def forward(self, input, mems):
            out, mems = self.mem_model(input.transpose(0, 1).contiguous(), mems)
            if self.linear: out = self.linear(out)
            return self.final(out).transpose(0,1).contiguous(), mems

    model = TransformerLM(mem_model)
    return model