"""Generation helpers."""

import torch


def greedy_decode(model: torch.nn.Module, input_ids: torch.Tensor, max_length: int) -> torch.Tensor:
    """Run greedy decoding with ``model.generate`` and return generated token ids."""

    return model.generate(
        input_ids,
        max_length=max_length,
        do_sample=False,
        num_beams=1,
    )
