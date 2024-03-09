import torch

def get_attention_size(model):
    cur_window_size = model.layers[0].attention.cache_k.size()[1]
    assert(cur_window_size == model.layers[0].attention.cache_v.size()[1])
    return cur_window_size

def kv_cache_resize(model, start_size, recent_size):
    if get_attention_size(model) >= start_size + recent_size:
        model.kv_cache_resize(start_size, recent_size)