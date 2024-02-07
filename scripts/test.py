import os
import fire
import torch
from llama.generation import Llama
from typing import List
import torch.nn as nn
import tqdm as tqdm
from data import get_loaders 
import time
from datasets import load_dataset
"""
python -m torch.distributed.launch --nproc_per_node=1 --master_port=25691 perplexity_eval.py 
"""
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
    nsamples = 10 #Sanity check
    
    
    import time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    # Loop through each batch
    for i in range(0, nsamples, bs):
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
        logits = model.forward(prompt, prev_pos)
        batch_logits[:, prev_pos:prompt_size, :] = logits
        for curr_pos in range(prompt_size, seqlen):
            logits = model.forward(inputs[:, curr_pos: curr_pos + 1], curr_pos)
            batch_logits[:, curr_pos, :] = logits
        
        shift_logits = batch_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1).type(torch.long))
        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * seqlen * (j-i)
        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)
        if i % 5 == 0:
            end.record()
            torch.cuda.synchronize()
            print(f"Total Elapsed Time Till Now: {start.elapsed_time(end)/1000:.2f} sec.")
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(f"PPL calculation completed. {ppl}")
    
    end.record()
    torch.cuda.synchronize()
    print(f"Total Elapsed Time Till Now: {start.elapsed_time(end)/1000:.2f} sec.")
    
    return ppl
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
    ckpt_dir = "./llama-2-13b",
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
    model_args = generator.model_args
    ppl = evaluate_perplexity(model, tokenizer, model_args)
    fopen = open("logs/llama13b_results.txt", "a")
    print(f"PPL on wikitext: {ppl}\n", flush=True, file=fopen)
    # skip = True
    # print(f"{model_args}\nPrompt Len = 128\nSkip = {skip}\nTotal Layer skipped : {model.skiped_layers}/{model.total_layers} = {(float(model.skiped_layers)/model.total_layers) * 100:.3f}%\nPPL on wikitext: {ppl}\n", flush=True, file=fopen)
    # fopen.close()
if __name__ == "__main__":
    fire.Fire(main)