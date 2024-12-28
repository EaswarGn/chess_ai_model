import math

def get_lr(step, d_model, warmup_steps, schedule="vaswani", decay=0.06):
    """
    The LR schedule.

    Args:

        step (int): Training step number.

        d_model (int): Size of vectors throughout the transformer model.

        warmup_steps (int): Number of warmup steps where learning rate
        is increased linearly; twice the value in the paper, as in the
        official T2T repo.

    Returns:

        float: Updated learning rate.

    Args:

        step (int): Training step number.

        d_model (int): Size of vectors throughout the transformer model.

        warmup_steps (int): Number of warmup steps where learning rate
        is increased linearly.

        schedule (str, optional): The learning rate schedule. Defaults
        to "vaswani", in which case the schedule in "Attention Is All
        You Need", by Vasvani et. al. is followed. This version below is
        twice the definition in the paper, as used in the official T2T
        repository. If the schedule is "exp_decay", the learning rate is
        exponentially decayed after the warmup stage.

        decay (float, optional): The decay rate per 10000 training steps
        for the "exp_decay" schedule. Defaults to 0.06, i.e. 6%.

    Raises:

        NotImplementedError: If the schedule is not one of "vaswani" or
        "exp_decay".

    Returns:

        float: Updated learning rate.
    """
    if schedule == "vaswani":
        lr = (
            2.0
            * math.pow(d_model, -0.5)
            * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))
        )
    elif schedule == "exp_decay":
        if step <= warmup_steps:
            lr = 1e-3 * step / warmup_steps
        else:
            lr = 1e-3 * ((1 - decay) ** ((step - warmup_steps) / 10000))
    else:
        raise NotImplementedError

    return lr