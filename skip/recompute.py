import torch
from skip.utils import *

def recompute(curr_pos,
              gen_counter,
              model,
              tokenizer,
              look_back,
              start_size,
              recent_size,
              exit_layer_states,
              dropped_layers,
              prev_words,
              error_count,
              error_threshold,
              check_period,
              incr_counter,
              data_store_per_batch,
              debug
              ):
    sample_pos = curr_pos - start_size - look_back
    if sample_pos < 0:
        sample_pos = 0
    prev_state = exit_layer_states[sample_pos]
    prev_start_layer, prev_end_layer = dropped_layers[sample_pos]
    prev_word = prev_words[sample_pos]
    new_word = prev_word
    if not skipped(prev_start_layer, prev_end_layer):
        return new_word, incr_counter, error_count, gen_counter

    if curr_pos % check_period == 0:
        new_out = model.recompute(prev_state,
                                curr_pos - look_back,
                                look_back,
                                prev_start_layer,
                                data_store_per_batch,
                                debug)
        new_token = torch.argmax(new_out[:, -1], dim=-1)
        new_token = new_token.reshape(-1)
        new_word = tokenizer.decode(new_token.tolist())
        if new_word != prev_word:
            error_count += 1
            # TODO change check period?
        if error_count >= error_threshold:
            error_count = 0
            incr_counter = 0
            gen_counter = 0
    
    return new_word, incr_counter, error_count, gen_counter