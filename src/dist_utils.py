import builtins
import torch.distributed as dist

def setup_rank_print(rank: int):
    """
    Override the global `print` function to automatically prepend [Rank {rank}]
    to every printed message. Call this ONCE after you know the rank.
    """
    builtin_print = builtins.print

    def rank_print(*args, **kwargs):
        prefix = f"[Rank {rank}]"
        # Prepend prefix to the first argument if it's a string, else just add it
        if args:
            new_args = (f"{prefix} {args[0]}",) + args[1:]
        else:
            new_args = (prefix,)
        builtin_print(*new_args, **kwargs)

    builtins.print = rank_print


def print0(*args, **kwargs):
    """
    Print only if rank == 0. 
    This is useful for logging in multi-process training.
    """
    if not dist.is_available() or not dist.is_initialized():
        # Fallback: just print if no distributed context
        builtins.print(*args, **kwargs)
    else:
        if dist.get_rank() == 0:
            builtins.print(*args, **kwargs)


