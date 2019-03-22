import torch
from fastprogress.fastprogress import master_bar, progress_bar
from fastai.text.data import UNK, TK_MAJ
import numpy as np
cuda_enabled = torch.cuda.is_available()

def get_x_input(partial, bptt):
    _, _, _, seq = partial
    max_len = bptt-1
    input = seq[-max_len:]
    input_var = torch.LongTensor([input])
    if cuda_enabled:
        input_var = input_var.cuda()
    return input_var
#     return input_var.unsqueeze(dim=0)

# song = string
# seq_length = generated song length
# beam_size = what to choose from
def beam_search(seed, seq_length, beam_size, learn, temperature=2):    
    xb, yb = learn.data.one_item(seed)
    seed_list = xb.squeeze().tolist()
    learn.model.reset()
    partial_sequences = [(0, 0, [], seed_list)]
    
    for i in progress_bar(range(seq_length), leave=True):
        partial_sequences = find_partials(partial_sequences, beam_size, learn, temperature=temperature)
    final_sequence = partial_sequences[0][3] # 0 = highest prob, 3 = seq
    return learn.data.vocab.textify(final_sequence), final_sequence
    
def find_partials(partial_sequences, beam_size, learn, 
                  no_maj:bool=False, no_unk:bool=True, temperature:float=2, 
                  min_p:float=None, random=True):
    partial_next = []
    bptt = learn.data.bptt
    for partial in partial_sequences:
        it, tot_p, p_list, seq = partial
        x_input = get_x_input(partial, bptt=bptt)
        y = torch.zeros_like(x_input)
        
        predict_probs = learn.pred_batch(batch=(x_input,y)).squeeze()[-1]
        if no_unk: predict_probs[learn.data.vocab.stoi[UNK]] = 0.
        if no_maj: predict_probs[learn.data.vocab.stoi[TK_MAJ]] = 0.
        if min_p is not None: predict_probs[res < min_p] = 0.
        if temperature != 1.: predict_probs.pow_(1 / temperature)
            
            
        if random:
            idxs = torch.multinomial(predict_probs, beam_size)
            probs = predict_probs[idxs]
        else:
            # last_it_probs = torch.exp(predict_probs[-(it+1):]) # this is to predict the last few iterations
            last_it_probs = torch.exp(predict_probs)
            probs, idxs = torch.topk(last_it_probs, beam_size, 0)
        
#         reps = [(data.vocab.itos[i],p) for i,p in zip(idxs,probs) if i in [8, 9]]
#         if reps: print(reps)
            
        def calc_prob(probs):
#             return np.prod(-np.log(probs[-beam_size:]))
            return -np.prod(np.abs(np.log(probs[-(beam_size*10):])))
            
        for prob,idx in zip(probs,idxs):
            new_p_list = p_list+[prob.item()]
            partial_next.append((it+1, calc_prob(new_p_list), new_p_list, seq+[idx.item()]))
    partial_sequences = sorted(partial_next, key=lambda x: x[1], reverse=True)[:beam_size]
#     print([np.average(x[2]) for x in partial_sequences])
    return partial_sequences