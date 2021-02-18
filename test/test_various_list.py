import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_list_func(sample):
    return sample.tolist()


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


l1, l2, l3 = 3,4,5

a = [[torch.ones(l1) for _ in range(l2)] for _ in range(l3)]
b = [torch.ones(l1) for _ in range(l2)]

print(apply_to_sample(to_list_func, a))
print(apply_to_sample(to_list_func, b))

print(apply_to_sample(to_list_func, [1,2,3,4]))
print(apply_to_sample(to_list_func, torch.tensor([1,2,3,4])))
