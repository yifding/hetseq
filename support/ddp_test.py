import inspect

import torch
import torch.nn as nn


def test():
    ddp_class = nn.parallel.DistributedDataParallel
    """
    init_kwargs = dict(
        module=model,
        device_ids=[args.device_id],
        output_device=args.device_id,
        broadcast_buffers=False,
        bucket_cap_mb=args.bucket_cap_mb,
        check_reduction=True,
    )
    """
    print(inspect.getfullargspec(ddp))

if __name__ == "__main__":
    test()