import torch
import torch.nn as nn
import fire
import os
import random
import numpy as np
import pickle

from llama.generation import Llama
from llama.model import ModelArgs
from skip.streaming_llm import *
from skip.utils import *
from skip.recompute import *

def generate(model, tokenizer, model_args, params, prompt):
    enc = tokenizer.encode(prompt, True, False)
    enc = torch.Tensor(enc).unsqueeze(0).type(torch.int).cuda()
    seqlen = model_args.max_seq_len
    nlls = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # parameters need for each batch
    dropped_layers = [] # record which layers (start, end_layer) are dropped for each token
    exit_layer_states = []  # for recomputation, keep the layer states of the before start_layer
    prev_words = [] # record words/tokens generated with layer skipping
    incr_counter = 0    # specify which layer to stop skipping: (max_skip_layer + 1) - incr_counter
    error_count = 0 # keep track of # of divergences occuring after recomputation
    data_store_per_batch = {}   # used for debugging
    gen_counter = 0 # keep track of number of token with layer skipping
    prompt_size = enc.size()[1]
    no_skip = True if params.policy == "no_skip" else False
    
    output_tokens = []
    
    model.clear_kv_cache()
    logits = model.forward(enc, 0)
    if params.temperature > 0:
        probs = torch.softmax(logits[:, -1] / params.temperature, dim=-1)
        next_token = sample_top_p(probs, params.top_p)
    else:
        next_token = torch.argmax(logits[:, -1], dim=-1)
    next_token = next_token.reshape(-1)
    output_tokens.append(next_token.item())
    word = tokenizer.decode(next_token.tolist())

    total_num_skipped = 0
    total_layers = 0

    newline_count = 0

    for curr_pos in range(prompt_size, model_args.max_seq_len):
        if curr_pos < params.start_size:
            next_token = next_token.reshape(1, 1)
            logits = model.forward(next_token, curr_pos)
            if params.temperature > 0:
                probs = torch.softmax(logits[:, -1] / params.temperature, dim=-1)
                next_token = sample_top_p(probs, params.top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            output_tokens.append(next_token.item())
            word = tokenizer.decode(next_token.tolist())
            continue

        gen_counter += 1
        
        # 1. resize the kv cache to keep a fixed size
        kv_cache_resize(model, params.start_size, params.recent_size)

        # 2. calculate which layers should be skipped
        start_skip, end_skip, incr_counter = get_skip_param(gen_counter,
                                                params.step_size,
                                                params.min_skip_layer,
                                                params.max_skip_layer,
                                                incr_counter,
                                                no_skip)
        
        # 3. perform layer skipping
        next_token = next_token.reshape(1, 1)
        logits, exit_layer_state, num_skipped = model.forward_with_skipping(next_token,
                                                               curr_pos,
                                                               (start_skip, end_skip),
                                                                params.policy,
                                                               data_store_per_batch,
                                                               params.debug)
        total_num_skipped += num_skipped
        total_layers += params.num_decoder
        if params.temperature > 0:
            probs = torch.softmax(logits[:, -1] / params.temperature, dim=-1)
            next_token = sample_top_p(probs, params.top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        # 4. keep track of skipped layers and intermediate states for recomputation
        dropped_layers.append((start_skip, end_skip))
        exit_layer_states.append(exit_layer_state)
        
        next_token = torch.argmax(logits[:, -1], dim=-1)
        next_token = next_token.reshape(-1)
        output_tokens.append(next_token.item())

        word = tokenizer.decode(next_token.tolist())
        prev_words.append(word)

         # 5. perform recomputation
        new_word, incr_counter, error_count, gen_counter = recompute(
            curr_pos,
            gen_counter,
            model,
            tokenizer,
            exit_layer_states,
            dropped_layers,
            prev_words,
            error_count,
            incr_counter,
            data_store_per_batch,
            params,
            )

        if next_token == tokenizer.eos_id:
            break
        if sum(output_tokens[-3:]) == 13 * 3:
            break

    data_store = []
    data_store.append(data_store_per_batch)
    print(output_tokens)
    print(prompt)
    print(tokenizer.decode(output_tokens))
    print(total_num_skipped / total_layers)
    with open('results/test.pkl', 'wb') as f:
        pickle.dump(data_store, f)   

def main(
    ckpt_dir = "./llama-2-7b-chat",
    tokenizer_path = "./tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
    max_gen_len: int = 128,
    prompt = "",
    step_size: int = 40,
    start_size: int = 128,
    recent_size: int = 256,
    min_skip_layer: int = 8,
    max_skip_layer: int = 24,
    num_decoder: int = 32,
    look_back: int = 1,
    error_threshold = 6,
    check_period: int = 10,
    policy = "attention",
    debug = False,
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

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    params: SkipParams = SkipParams(
        temperature = temperature,
        top_p = top_p,
        step_size = step_size,
        start_size = start_size,
        recent_size = recent_size,
        min_skip_layer = min_skip_layer,
        max_skip_layer = max_skip_layer,
        num_decoder = num_decoder,
        look_back = look_back,
        error_threshold = error_threshold,
        check_period = check_period,
        policy = policy,
        debug = debug,
    )

    ppl = generate(model, tokenizer, model_args, params, prompt)
    # fopen = open("logs/llama7b_chat_11_22.txt", "a")
    # print(f"PPL on wikitext: {ppl}\n", flush=True, file=fopen)
    # print(f"{model_args}\nPrompt Len = 128\nSkip = {skip}\nTotal Layer skipped : {model.skiped_layers}/{model.total_layers} = {(float(model.skiped_layers)/model.total_layers) * 100:.3f}%\nPPL on wikitext: {ppl}\n", flush=True, file=fopen)
    # fopen.close()

if __name__ == "__main__":
    fire.Fire(main)
