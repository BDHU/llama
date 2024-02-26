import os
import fire
from llama.generation import Llama
from llama.model import ModelArgs
from typing import List
import torch.nn as nn
import tqdm as tqdm
# from data import get_loaders 
import time
import pickle

import numpy as np
import random
import torch
from datasets import load_dataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)

"""
python -m torch.distributed.launch --nproc_per_node=1 --master_port=25691 perplexity_eval.py 
"""

def calc_skip_layer(min_skip_layer, max_skip_layer, max_seq_len, num_decoder_layers, token_id, drop_type):
    if drop_type == "linear":
        interval = max_seq_len / (max_skip_layer - min_skip_layer + 1)
        to_reduce = int(token_id / interval)
        layer_to_exit = max_skip_layer - to_reduce
        
        if layer_to_exit > max_skip_layer:
            layer_to_exit = max_skip_layer
        if layer_to_exit < min_skip_layer:
            layer_to_exit = min_skip_layer
        return layer_to_exit, max_skip_layer
    elif drop_type == "linear_flat":
        interval = (max_seq_len/2) / (max_skip_layer - min_skip_layer + 1)
        if token_id < max_seq_len:
            to_reduce = int(token_id / interval)
            layer_to_exit = max_skip_layer - to_reduce
        else:
            to_add = int(token_id / interval)
            layer_to_exit = min_skip_layer + to_add
        
        if layer_to_exit > max_skip_layer:
            layer_to_exit = max_skip_layer
        if layer_to_exit < min_skip_layer:
            layer_to_exit = min_skip_layer
        return layer_to_exit, max_skip_layer
    elif drop_type == "linear_U":
        interval = (max_seq_len/2) / (max_skip_layer - min_skip_layer + 1)
        if token_id < max_seq_len/2:
            to_reduce = int(token_id / interval)
            layer_to_exit = max_skip_layer - to_reduce
        else:
            to_add = int((token_id - max_seq_len/2) / interval)
            layer_to_exit = min_skip_layer + to_add
        
        if layer_to_exit > max_skip_layer:
            layer_to_exit = max_skip_layer
        if layer_to_exit < min_skip_layer:
            layer_to_exit = min_skip_layer
        return layer_to_exit, max_skip_layer
    elif drop_type == "linear_full":
        interval = max_seq_len / (max_skip_layer - min_skip_layer + 1)
        to_reduce = int(token_id / interval)
        layer_to_exit = max_skip_layer - to_reduce
        
        if layer_to_exit > max_skip_layer:
            layer_to_exit = max_skip_layer
        if layer_to_exit < min_skip_layer:
            layer_to_exit = min_skip_layer
        
        if token_id > max_seq_len - 250:
            return 9999, 9999
        return layer_to_exit, max_skip_layer
    elif drop_type == "periodic_512_100":
        if token_id < 50:
            return 9999, 9999
        token_id -= 128
        token_id = token_id % (512 + 100)

        interval = 512 / (max_skip_layer - min_skip_layer + 1)
        to_reduce = int(token_id / interval)
        layer_to_exit = max_skip_layer - to_reduce
        if token_id > 512:
            return 9999, 9999
        
        if layer_to_exit > max_skip_layer:
            layer_to_exit = max_skip_layer
        if layer_to_exit < min_skip_layer:
            layer_to_exit = min_skip_layer
        return layer_to_exit, max_skip_layer
   

def calc_kv_drop(token_id):
    if (token_id-128) % (512+100) == 0 and (token_id-128) / 612 < 1.1:
        pos = token_id - (512+100)
        length = 512
        return (pos, length)
    elif (token_id-128) % (512+100) == 0 and (token_id-128) / 612 > 1.1:
        pos = token_id - (512+100+100)
        length = 512+100
        return (pos, length)
    return None

def get_window_size(model):
    cur_window_size = model.layers[0].attention.cache_k.size()[1]
    assert(cur_window_size == model.layers[0].attention.cache_v.size()[1])
    return cur_window_size

def shrink_window(model, window_size, sink):
    if get_window_size(model) >= window_size:
        model.fix_windows_size(sink, window_size)

    

def evaluate_perplexity(model, tokenizer, model_args):
    dataset = "wikitext2"
    bs = 1
    # Print status
    print(f"evaluating on {dataset}")
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', ignore_verifications=True)
    testenc = tokenizer.encode("\n\n".join(testdata['text']), False, False)
    testenc = torch.Tensor(testenc).unsqueeze(0).type(torch.int)
    seqlen = model_args.max_seq_len
    nsamples = testenc.numel() // seqlen
    nlls = []
    print(f"nsamples {nsamples}")
    # nsamples = 10 #Sanity check
    nsamples = 1
    
    
    data_store = [] 
    import time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # start.record()
    # Loop through each batch
    for i in range(0, nsamples, bs):
        data_store_per_batch = {}
        # print(f"sample {i}/{nsamples} || skipped layers {model.skiped_layers}/{model.total_layers}")
        # model.sk = 0
        # Calculate end index
        j = min(i+bs, nsamples)
        # Prepare inputs and move to device
        inputs = testenc[:,(i * seqlen):(j * seqlen)].cuda()
        inputs = inputs.reshape(j-i, seqlen)
        batch_logits = torch.zeros((bs, seqlen, 32000), dtype=torch.float32).cuda()
        prev_pos = 0
        prompt_size = 128
        prompt = inputs[:, :prompt_size]
        # clear up the kv cache
        model.clear_kv_cache()
        logits = model.forward(prompt, prev_pos)
        batch_logits[:, prev_pos:prompt_size, :] = logits
        
        dropped_layers = []
        exit_layer_states = []
        prev_words = []
        incr_counter = 0
        step_size = 40
        error_count = 0
        check_period = 15

        start.record()

        for curr_pos in range(prompt_size, seqlen):

            # copy_states(model, dropped_layers, bsz=1)

            if get_window_size(model) >= 512+prompt_size:
                shrink_window(model, 512+prompt_size, prompt_size)



            if (curr_pos-prompt_size) % step_size == 0:
                start_skip = 22 - incr_counter
                end_skip = 22
                incr_counter += 1
                if 8 >= start_skip:
                    start_skip = 8

            # start_skip = 9999
            # end_skip = 9999


            # start_skip, end_skip = calc_skip_layer(8, 22, seqlen, 32, curr_pos-prompt_size+1, "linear_U")
            dropped_layers.append((start_skip, end_skip))
            drop_param = calc_kv_drop(curr_pos)
            logits, exit_layer_state = model.forward_with_skip_drop(inputs[:, curr_pos: curr_pos + 1], curr_pos, [(start_skip, end_skip)],
                                                                    data_store_per_batch, tokenizer, drop_param, start_skip-1, True)
            # logits, exit_layer_state = model.forward_with_skip_drop(inputs[:, curr_pos: curr_pos + 1], curr_pos, [(9999, 9999)],
            #                                                         data_store_per_batch, tokenizer, drop_param, start_skip-1, True)
            exit_layer_states.append(exit_layer_state)
            batch_logits[:, curr_pos, :] = logits

            next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            word = tokenizer.decode(next_token.tolist())
            prev_words.append(word)
            # print(word, end='')


            sample_pos = curr_pos - 5
            if curr_pos % check_period == 0:
                # recompute, choose the first one for now
                # sample_pos = curr_pos - 5
                state = exit_layer_states[sample_pos-prompt_size]
                restart_layer = dropped_layers[sample_pos-prompt_size][0]
                prev_word = prev_words[sample_pos-prompt_size]
                recomputed_out = model.recompute(state, sample_pos, restart_layer, data_store_per_batch, True)
                next_token = torch.argmax(recomputed_out[:, -1], dim=-1)
                next_token = next_token.reshape(-1)
                word = tokenizer.decode(next_token.tolist())
                if word != prev_word:
                    error_count += 1
                    check_period = int(check_period / 2)
                    if check_period <= 0:
                        check_period = 1

                if error_count >= 20:
                    incr_counter = 0
                    error_count = 0
                    check_period = 15
                    



        data_store.append(data_store_per_batch)


        shift_logits = batch_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss_fct_2 = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1).type(torch.long))
        loss_2 = loss_fct_2(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1).type(torch.long))
        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * seqlen * (j-i)
        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)
        # if i % 5 == 0:
        #     end.record()
        #     torch.cuda.synchronize()
        #     print(f"Total Elapsed Time Till Now: {start.elapsed_time(end)/1000:.2f} sec.")
    
    end.record()
    torch.cuda.synchronize()
    print(f"Total Elapsed Time Till Now: {start.elapsed_time(end)/1000:.2f} sec.")

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(f"PPL calculation completed. {ppl}")

    with open("results/skip_8_22_linear_U_recompute_2_error_20_check_3.txt", 'w') as f:
        for nll in loss_2:
            f.write(f"{nll}\n")
    
    with open('results/skip_8_22_linear_U_recompute_2_error_20_check_3.pkl', 'wb') as f:
        pickle.dump(data_store, f)
    
    return ppl

def copy_states(model, dropped_layers, bsz=1):
    if len(dropped_layers) <= 0:
        return
    
    prev_it = dropped_layers[-1]
    prev_start = prev_it[0]
    prev_end = prev_it[1]

    # copy the last layer
    from_layer = model.layers[-1]
    for i in range(prev_start, prev_end+1):
        layer = model.layers[i]
        # print("before copy")
        # print(layer.attention.cache_k[:bsz, pos:pos+1])
        copy_kv_cache(from_layer, layer, bsz)
        # print("after copy")
        # print(layer.attention.cache_k[:bsz, pos:pos+1])

@torch.inference_mode()
def copy_kv_cache(from_layer, to_layer, bsz):
    # get kv cache
    # to_layer.attention.cache_k[:bsz, start_pos:start_pos+num_elem] = torch.clone(from_layer.attention.cache_k[:bsz, start_pos:start_pos+num_elem])
    # to_layer.attention.cache_v[:bsz, start_pos:start_pos+num_elem] = torch.clone(from_layer.attention.cache_v[:bsz, start_pos:start_pos+num_elem])
    to_layer.attention.cache_k= torch.cat((to_layer.attention.cache_k[:bsz, :-1], torch.clone(from_layer.attention.cache_k[:bsz, -1:])), dim=1)
    to_layer.attention.cache_v= torch.cat((to_layer.attention.cache_v[:bsz, :-1], torch.clone(from_layer.attention.cache_v[:bsz, -1:])), dim=1)
    torch.cuda.empty_cache()
    


def eval_ppl(model, tokenizer, model_args):
    # Set dataset
    dataset = "wikitext2"
    # Print status
    print(f"evaluating on {dataset}")
    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model_args.max_seq_len, tokenizer=tokenizer 
    )
    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl = eval_ppl_wikitext(model, testloader, model_args.max_seq_len, 1 )
    return ppl 
# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, seqlen, bs=1):
    # Get input IDs
    testenc = testenc.input_ids
    # Calculate number of samples
    nsamples = testenc.numel() // seqlen
    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")
    nsamples = 10 #Sanity check
    # Loop through each batch
    for i in range(0, nsamples, bs):
        print(f"sample {i}/{nsamples}")
        # Calculate end index
        j = min(i+bs, nsamples)
        # Prepare inputs and move to device
        inputs = testenc[:,(i * seqlen):(j * seqlen)].cuda()
        inputs = inputs.reshape(j-i, seqlen)
        # Forward pass through the model
        lm_logits = model(inputs).logits
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:].type(torch.float32)
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * seqlen * (j-i)
        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    # Empty CUDA cache to save memory
    # torch.cuda.empty_cache()
    return ppl.item()
def main(
    ckpt_dir = "./llama-2-7b-chat",
    tokenizer_path = "./tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
    max_gen_len: int = 128,
):

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    print("Model Loading completed")
    model = generator.model
    tokenizer = generator.tokenizer
    # model_args = generator.model_args

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    ppl = evaluate_perplexity(model, tokenizer, model_args)
    fopen = open("logs/llama7b_chat_11_22.txt", "a")
    print(f"PPL on wikitext: {ppl}\n", flush=True, file=fopen)
    # skip = True
    # print(f"{model_args}\nPrompt Len = 128\nSkip = {skip}\nTotal Layer skipped : {model.skiped_layers}/{model.total_layers} = {(float(model.skiped_layers)/model.total_layers) * 100:.3f}%\nPPL on wikitext: {ppl}\n", flush=True, file=fopen)
    # fopen.close()
if __name__ == "__main__":
    fire.Fire(main)
