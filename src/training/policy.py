"""Native encoder-decoder rollout/scoring adapters; no data or run admission.

Sampling uses the full categorical policy at an explicit temperature. Dropout and
ambient autocast are suppressed for collection and scoring at the same declared
model-weight dtype, with fp32 log-softmax. Caller modes/autocast are restored.
These utilities do not load checkpoints, collect rewards or start an optimizer.
"""

from __future__ import annotations

import math
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ..models.heads import LMHead
from .rl import GroupRelativeConfig, RolloutContract, group_relative_loss, offline_preference_loss


@contextmanager
def deterministic_policy(model):
    modes = [(module, module.training) for module in model.modules()]
    try:
        with ExitStack() as contexts:
            for device_type in {parameter.device.type for parameter in model.parameters()}:
                contexts.enter_context(torch.autocast(device_type, enabled=False))
            model.eval()
            yield
    finally:
        for module, training in modes:
            module.training = training


def policy_precision(model) -> tuple[str, str]:
    """Explicit native scoring contract; mixed parameter dtypes are unsupported."""
    dtypes = {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()}
    if len(dtypes) != 1:
        raise ValueError("Native policy precision requires one floating model-weight dtype")
    return "no_autocast_fp32_log_softmax", str(next(iter(dtypes)))


def _temperature(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("Policy temperature must be positive and finite")


def _token_ids(model, start, end, pad):
    if any(type(value) is not int or value < 0 for value in (start, end, pad)):
        raise ValueError("Special token IDs must be nonnegative integers")
    vocab = getattr(model.decoder, "vocab_size", None)
    if vocab is not None and max(start, end, pad) >= vocab:
        raise ValueError("Special token IDs must belong to the decoder vocabulary")


def _prompts(ids, mask):
    if (
        ids.ndim != 2
        or not all(ids.shape)
        or ids.dtype not in (torch.int32, torch.int64)
        or mask.dtype != torch.bool
        or mask.shape != ids.shape
        or mask.device != ids.device
        or bool((ids < 0).any())
        or not bool(mask.any(dim=1).all())
    ):
        raise ValueError(
            "Prompts require nonnegative B x S IDs and nonempty boolean attention masks"
        )


def _head(model, task):
    head = model.heads.get(task)
    if not isinstance(getattr(head, "_orig_mod", head), LMHead) or model.decoder is None:
        raise ValueError(
            "Policy objectives require a registered encoder-decoder language-model head"
        )
    return head


def _response_mask(tokens, mask, end_token_id, pad_token_id):
    if (
        tokens.ndim != 2
        or not all(tokens.shape)
        or tokens.dtype not in (torch.int32, torch.int64)
        or mask.dtype != torch.bool
        or mask.shape != tokens.shape
        or mask.device != tokens.device
        or bool((mask[:, 1:] & ~mask[:, :-1]).any())
        or not bool(mask.any(dim=1).all())
        or bool((tokens < 0).any())
        or bool((tokens[~mask] != pad_token_id).any())
    ):
        raise ValueError(
            "Responses require nonempty contiguous token prefixes followed by declared padding"
        )
    eos = (tokens == end_token_id) & mask
    if bool((eos.cumsum(dim=1)[:, :-1].bool() & mask[:, 1:]).any()):
        raise ValueError("Response mask must include first EOS and exclude every token after it")


def _encode(model, ids, mask):
    return model.encoder(ids, mask=mask.unsqueeze(1) & mask.unsqueeze(2))


def _score(model, task, memory, source_mask, tokens, mask, *, start_token_id, temperature):
    full_length = tokens.shape[1]
    observed_length = int(mask.sum(dim=1).max())
    tokens, mask = tokens[:, :observed_length], mask[:, :observed_length]
    target_ids = torch.full_like(tokens, start_token_id)
    target_ids[:, 1:] = tokens[:, :-1]
    # PAD can be a real sampled action under full-support sampling. The decoder
    # must not reinterpret its value as padding; causal order protects active
    # positions from the right-padding that follows a completed response.
    logits = model.decoder(target_ids, memory, memory_mask=source_mask, skip_padding_mask=True)
    if not model.decoder_outputs_logits:
        logits = _head(model, task)(logits)
    if (
        logits.ndim != 3
        or logits.shape[:2] != tokens.shape
        or bool((tokens >= logits.shape[-1]).any())
    ):
        raise ValueError("Decoder logits do not cover the response token IDs")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Policy logits must be finite before masked normalization")
    # Padding positions do not contribute to response log probabilities.
    logits = torch.where(mask.unsqueeze(-1), logits.float(), 0.0) / temperature
    scores = F.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return F.pad(torch.where(mask, scores, 0.0), (0, full_length - observed_length))


def score_responses(
    model,
    source_ids,
    source_mask,
    response_ids,
    response_mask,
    *,
    start_token_id,
    end_token_id,
    pad_token_id,
    temperature,
    task="summarization",
):
    """Score response tokens only, with a correctly shifted decoder input."""
    _temperature(temperature)
    _prompts(source_ids, source_mask)
    _response_mask(response_ids, response_mask, end_token_id, pad_token_id)
    _head(model, task)
    _token_ids(model, start_token_id, end_token_id, pad_token_id)
    policy_precision(model)
    if source_ids.shape[0] != response_ids.shape[0] or source_ids.device != response_ids.device:
        raise ValueError("Prompt and response batches must align on one device")
    with deterministic_policy(model):
        return _score(
            model,
            task,
            _encode(model, source_ids, source_mask),
            source_mask,
            response_ids,
            response_mask,
            start_token_id=start_token_id,
            temperature=temperature,
        )


@dataclass(frozen=True)
class SampledResponses:
    source_ids: torch.Tensor
    source_mask: torch.Tensor
    response_ids: torch.Tensor
    response_mask: torch.Tensor
    behavior_log_probs: torch.Tensor
    group_ids: torch.Tensor
    terminated: torch.Tensor
    behavior_policy_revision: str
    temperature: float
    tokenizer_revision: str
    token_contract: tuple[int, int, int]
    precision_contract: tuple[str, str]

    def training_batch(self, rewards, provenance):
        """Attach separately verified rewards; this does not admit a dataset."""
        return {
            "src_ids": self.source_ids,
            "src_mask": self.source_mask,
            "response_ids": self.response_ids,
            "response_mask": self.response_mask,
            "behavior_log_probs": self.behavior_log_probs,
            "group_ids": self.group_ids,
            "rewards": rewards,
            "tokenizer_revision": self.tokenizer_revision,
            "token_contract": self.token_contract,
            "precision_contract": self.precision_contract,
            "rollout": RolloutContract(
                self.behavior_policy_revision, self.temperature, tuple(provenance)
            ),
        }


@torch.no_grad()
def sample_responses(
    model,
    source_ids,
    source_mask,
    *,
    group_size,
    max_response_tokens,
    start_token_id,
    end_token_id,
    pad_token_id,
    temperature,
    behavior_policy_revision,
    tokenizer_revision,
    generator=None,
    task="summarization",
):
    """Collect bounded native rollouts; reward assignment is a separate operation."""
    _prompts(source_ids, source_mask)
    _temperature(temperature)
    _head(model, task)
    if not model.decoder_outputs_logits:
        raise ValueError("Native step sampling requires decoder vocabulary logits")
    if any(type(n) is not int or n < 1 for n in (group_size, max_response_tokens)):
        raise ValueError("Group size and response cap must be positive integers")
    _token_ids(model, start_token_id, end_token_id, pad_token_id)
    if not isinstance(behavior_policy_revision, str) or not behavior_policy_revision.strip():
        raise ValueError("Sampling requires its behavior-policy revision")
    if not isinstance(tokenizer_revision, str) or not tokenizer_revision.strip():
        raise ValueError("Sampling requires its tokenizer revision")
    precision = policy_precision(model)
    with deterministic_policy(model):
        # One encoder pass per distinct prompt; groups share its detached memory.
        memory = _encode(model, source_ids, source_mask).repeat_interleave(group_size, dim=0)
        sources = source_ids.repeat_interleave(group_size, dim=0)
        attention = source_mask.repeat_interleave(group_size, dim=0)
        batch = sources.shape[0]
        tokens = torch.full(
            (batch, max_response_tokens), pad_token_id, device=sources.device, dtype=torch.long
        )
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        log_probs = torch.zeros_like(tokens, dtype=torch.float32)
        finished = torch.zeros(batch, dtype=torch.bool, device=sources.device)
        last = torch.full((batch, 1), start_token_id, device=sources.device, dtype=torch.long)
        cache: dict[str, Any] = {"past_length": 0, "memory_mask": attention}
        for step in range(max_response_tokens):
            logits, cache = model.decoder.step(last, memory, cache)
            active = ~finished
            if (
                logits.ndim != 2
                or logits.shape[0] != batch
                or max(start_token_id, end_token_id, pad_token_id) >= logits.shape[-1]
            ):
                raise ValueError("Decoder sampling logits/special token IDs are incompatible")
            if not bool(torch.isfinite(logits[active]).all()):
                raise ValueError("Active sampling logits must be finite")
            distribution = F.log_softmax(
                torch.where(active[:, None], logits.float(), 0.0) / temperature, dim=-1
            )
            selected = torch.multinomial(distribution.exp(), 1, generator=generator).squeeze(-1)
            tokens[:, step] = torch.where(active, selected, pad_token_id)
            mask[:, step] = active
            log_probs[:, step] = torch.where(
                active, distribution.gather(1, selected[:, None]).squeeze(1), 0.0
            )
            finished |= active & (selected == end_token_id)
            last = torch.where(finished, end_token_id, selected)[:, None]
            if bool(finished.all()):
                break
    return SampledResponses(
        sources,
        attention,
        tokens,
        mask,
        log_probs,
        torch.arange(source_ids.shape[0], device=sources.device).repeat_interleave(group_size),
        finished,
        behavior_policy_revision,
        float(temperature),
        tokenizer_revision,
        (start_token_id, end_token_id, pad_token_id),
        precision,
    )


class PolicyBatchMetrics(dict[str, float]):
    def __init__(self, *, responses, has_signal, loss_weight=None, **values):
        super().__init__(values)
        self.responses = responses
        self.has_signal = has_signal
        self.loss_weight = responses if loss_weight is None else loss_weight


class PolicyTask:
    """Optional objective callback for the existing Trainer; no second training loop.

    Frozen behavior/reference scores must come with their exact revisions. Outcome
    evaluation is separate: a clipped surrogate is not a checkpoint quality score.
    """

    def __init__(
        self,
        model,
        *,
        mode,
        start_token_id,
        end_token_id,
        pad_token_id,
        tokenizer_revision,
        temperature=1.0,
        config: GroupRelativeConfig | None = None,
        expected_behavior_revision=None,
        expected_reference_revision=None,
        beta=0.1,
        task="summarization",
    ):
        if mode not in {"group_relative", "dpo"}:
            raise ValueError("Policy mode must be group_relative or dpo")
        _temperature(temperature)
        _head(model, task)
        if not isinstance(tokenizer_revision, str) or not tokenizer_revision.strip():
            raise ValueError("Policy scoring requires its tokenizer revision")
        _token_ids(model, start_token_id, end_token_id, pad_token_id)
        for revision in (expected_behavior_revision, expected_reference_revision):
            if revision is not None and (not isinstance(revision, str) or not revision.strip()):
                raise ValueError("Expected policy revisions must be nonblank strings")
        if mode == "group_relative" and (config is None or not expected_behavior_revision):
            raise ValueError("Group-relative tasks require an objective and behavior revision")
        if (
            mode == "dpo" or (config is not None and config.kl_coefficient)
        ) and not expected_reference_revision:
            raise ValueError("Reference-scored objectives require an explicit reference revision")
        self.model, self.mode, self.task = model, mode, task
        self.start, self.end, self.pad = start_token_id, end_token_id, pad_token_id
        self.temperature, self.config, self.beta = temperature, config, beta
        self.behavior_revision, self.reference_revision = (
            expected_behavior_revision,
            expected_reference_revision,
        )
        self.tokenizer_revision = tokenizer_revision
        self.precision_contract = policy_precision(model)

    def __call__(self, batch) -> tuple[torch.Tensor, PolicyBatchMetrics]:
        ids, source_mask = batch["src_ids"], batch["src_mask"]
        _prompts(ids, source_mask)
        if batch.get("tokenizer_revision") != self.tokenizer_revision or batch.get(
            "token_contract"
        ) != (self.start, self.end, self.pad):
            raise ValueError("Scored batch tokenizer or special-token contract differs")
        if (
            policy_precision(self.model) != self.precision_contract
            or batch.get("precision_contract") != self.precision_contract
        ):
            raise ValueError(
                "Policy scoring precision or model-weight dtype differs from the collected batch"
            )
        if (
            self.reference_revision is not None
            and batch.get("reference_revision") != self.reference_revision
        ):
            raise ValueError("Reference policy revision differs from the scoring snapshot")
        if self.mode == "dpo" or "reference_log_probs" in batch:
            _temperature(batch.get("reference_temperature"))
            if (
                self.reference_revision is None
                or batch.get("reference_temperature") != self.temperature
            ):
                raise ValueError(
                    "Reference policy revision and scoring temperature must be explicit"
                )
            if batch.get("reference_precision_contract") != self.precision_contract:
                raise ValueError(
                    "Reference scoring precision must be explicit and match policy scoring"
                )
        if self.mode == "dpo":
            chosen, rejected = batch["chosen_ids"], batch["rejected_ids"]
            for prefix in ("chosen", "rejected"):
                tokens, mask = batch[prefix + "_ids"], batch[prefix + "_mask"]
                _response_mask(tokens, mask, self.end, self.pad)
                if tokens.shape[0] != ids.shape[0] or tokens.device != ids.device:
                    raise ValueError("Preference responses must align with prompts")
            width = max(chosen.shape[1], rejected.shape[1])
            same_tokens = (
                F.pad(chosen, (0, width - chosen.shape[1]), value=self.pad)
                == F.pad(rejected, (0, width - rejected.shape[1]), value=self.pad)
            ).all(dim=1)
            same_lengths = batch["chosen_mask"].sum(dim=1) == batch["rejected_mask"].sum(dim=1)
            if bool((same_tokens & same_lengths).any()):
                raise ValueError(
                    "A strict preference cannot rank identical response token sequences"
                )
            with deterministic_policy(self.model):
                memory = _encode(self.model, ids, source_mask)
                scores = []
                for prefix in ("chosen", "rejected"):
                    tokens, mask = batch[prefix + "_ids"], batch[prefix + "_mask"]
                    scores.append(
                        _score(
                            self.model,
                            self.task,
                            memory,
                            source_mask,
                            tokens,
                            mask,
                            start_token_id=self.start,
                            temperature=self.temperature,
                        )
                    )
            loss = offline_preference_loss(
                scores[0],
                scores[1],
                batch["chosen_mask"],
                batch["rejected_mask"],
                batch["reference_chosen"],
                batch["reference_rejected"],
                beta=self.beta,
                provenance=batch["provenance"],
            )
            return loss, PolicyBatchMetrics(responses=ids.shape[0], has_signal=True)
        tokens, mask = batch["response_ids"], batch["response_mask"]
        _response_mask(tokens, mask, self.end, self.pad)
        if tokens.shape[0] != ids.shape[0] or tokens.device != ids.device:
            raise ValueError("Responses must align with prompts")
        # Each group must actually represent one prompt, not just claim a group ID.
        groups = batch["group_ids"]
        if (
            groups.shape != (ids.shape[0],)
            or groups.dtype not in (torch.int32, torch.int64)
            or groups.device != ids.device
        ):
            raise ValueError("Group IDs must align with the prompt batch")
        _, inverse = torch.unique(groups, return_inverse=True)
        indices = torch.arange(len(ids), device=ids.device)
        first = torch.full_like(indices, len(ids)).scatter_reduce(
            0, inverse, indices, reduce="amin"
        )
        representatives = first[: int(inverse.max()) + 1]
        if not torch.equal(ids, ids[representatives][inverse]) or not torch.equal(
            source_mask, source_mask[representatives][inverse]
        ):
            raise ValueError("Every response group must share identical prompt IDs and masks")
        with deterministic_policy(self.model):
            memory = _encode(self.model, ids[representatives], source_mask[representatives])[
                inverse
            ]
            scores = _score(
                self.model,
                self.task,
                memory,
                source_mask,
                tokens,
                mask,
                start_token_id=self.start,
                temperature=self.temperature,
            )
        if not isinstance(batch.get("rollout"), RolloutContract):
            raise ValueError("Group-relative batches require a frozen rollout contract")
        assert self.config is not None
        result = group_relative_loss(
            scores,
            batch["behavior_log_probs"],
            mask,
            batch["rewards"],
            groups,
            config=self.config,
            rollout=batch["rollout"],
            expected_behavior_revision=self.behavior_revision,
            policy_temperature=self.temperature,
            reference_log_probs=batch.get("reference_log_probs"),
            reference_temperature=self.temperature if "reference_log_probs" in batch else None,
        )
        values = (
            torch.stack((result.policy_loss, result.sampled_reference_penalty))
            .detach()
            .cpu()
            .tolist()
        )
        return result.loss, PolicyBatchMetrics(
            responses=ids.shape[0],
            has_signal=result.has_signal,
            loss_weight=int(mask.sum()) if self.config.method == "dapo" else ids.shape[0],
            policy_loss=values[0],
            sampled_reference_penalty=values[1],
        )


@dataclass(frozen=True)
class ContinuationTarget:
    """Observed continuation with token-byte pieces from a pinned tokenizer adapter.

    Concatenation must reproduce the source bytes exactly. The caller must verify
    that these pieces really are that tokenizer's boundaries; a digest alone is
    not an implementation of an arbitrary tokenizer's byte-decoding rules.
    """

    observed_bytes: bytes
    token_bytes: tuple[bytes, ...]
    tokenizer_sha256: str
    source_evidence_sha256: str
    work_group: str
    split: str

    def __post_init__(self):
        from .rl import RewardProvenance

        if not isinstance(self.observed_bytes, bytes) or not self.observed_bytes:
            raise ValueError("Continuation target requires nonempty observed source bytes")
        if (
            not isinstance(self.token_bytes, tuple)
            or not self.token_bytes
            or any(not isinstance(token, bytes) or not token for token in self.token_bytes)
            or b"".join(self.token_bytes) != self.observed_bytes
        ):
            raise ValueError(
                "Verified token bytes must exactly reconstruct the observed continuation"
            )
        RewardProvenance("tokenizer", "pinned", self.tokenizer_sha256)
        RewardProvenance("source-continuation", "pinned", self.source_evidence_sha256)
        if (
            not isinstance(self.work_group, str)
            or not self.work_group.strip()
            or self.split not in {"train", "dev", "test"}
        ):
            raise ValueError("Continuation evidence requires a work group and explicit split")


def rpt_prefix_reward(prediction: bytes, target: ContinuationTarget) -> float:
    """RPT Eq3 byte-prefix rule; no trimming, normalization or empty-answer reward.

    This verifies an already extracted final prediction, not a whole reasoning
    trace. Source licensing, tokenizer boundary verification, answer extraction
    and train/held-out work separation remain explicit preparation requirements.
    """
    if not isinstance(prediction, bytes):
        raise ValueError("Prediction must be exact bytes from the declared answer extractor")
    if not prediction or not target.observed_bytes.startswith(prediction):
        return 0.0
    boundary = 0
    for token in target.token_bytes:
        boundary += len(token)
        if boundary == len(prediction):
            return 1.0
        if boundary > len(prediction):
            break
    return 0.0


def complete_field_reward(predictions, labels, known):
    """Balanced accuracy on complete reviewed field rows, not weak BGC omissions.

    This pure verifier cannot certify annotation provenance. A later admitted
    reward dataset must supply the review/evidence receipt for every row.
    """
    from ..models.losses import observed_label_mask

    if predictions.ndim != 2 or not all(predictions.shape) or predictions.dtype != torch.bool:
        raise ValueError("Field predictions must be boolean")
    if not isinstance(known, torch.Tensor) or known.dtype != torch.bool:
        raise ValueError("Complete field reward requires an explicit boolean review mask")
    mask = observed_label_mask(predictions.float(), labels, known)
    if not bool(mask.all()):
        raise ValueError("Field reward requires complete reviewed labels; unknown is not negative")
    positive = labels == 1
    positives, negatives = positive.sum(dim=1), (~positive).sum(dim=1)
    if not bool(((positives > 0) & (negatives > 0)).all()):
        raise ValueError("Field reward requires both positive and negative evidence per row")
    sensitivity = (predictions & positive).sum(dim=1) / positives
    specificity = (~predictions & ~positive).sum(dim=1) / negatives
    return ((sensitivity + specificity) / 2).detach()
