"""Loss helpers."""
import torch


def multitask_loss(losses: dict[str, torch.Tensor]) -> torch.Tensor:
    iterator = iter(losses.values())
    try:
        total = next(iterator).clone()
    except StopIteration:
        raise ValueError("losses is empty")
    for value in iterator:
        total = total + value
    return total / len(losses)
