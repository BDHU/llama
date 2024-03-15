import torch
from dataclasses import dataclass

@dataclass
class SkipParams:
    temperature: float = -1
    top_p: float = 0.9
    step_size: int = 40
    start_size: int = 128
    recent_size: int = 256
    min_skip_layer: int = 8
    max_skip_layer: int = 24
    num_decoder: int = 32
    look_back: int = 1
    error_threshold: int = 5
    check_period: int = 10
    policy: str = "no_skip"
    debug: bool = False
   
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def get_skip_param(gen_counter,
                    step_size,
                    min_skip_layer,
                    max_skip_layer,
                    incr_counter,
                    no_skip):
    start_skip = (max_skip_layer + 1) - incr_counter
    end_skip = max_skip_layer
    if incr_counter < 0:
        raise ValueError("incr_counter can not be < 0")

    if no_skip is True:
        start_skip, end_skip = (-1, -1)
        return -1, -1, incr_counter

    if gen_counter % step_size == 0:
        incr_counter += 1
    if start_skip <= min_skip_layer:
        start_skip = min_skip_layer
    return start_skip, end_skip, incr_counter

def skipped(start_skip, end_skip):
    if start_skip <= -1 and end_skip <= -1:
        return False
    if start_skip > end_skip:
        return False
    return True