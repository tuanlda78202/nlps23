# LR decay scheduler (cosine with warmup)
import math
import torch


def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    # 2) If it > lr_decay_iters, return min lr
    if it > lr_decay_iters:
        return min_lr

    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1

    # Coef ranges [0,1]
    coef = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coef * (learning_rate - min_lr)


def new_gelu(x):
    """GELU activation function"""

    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )
