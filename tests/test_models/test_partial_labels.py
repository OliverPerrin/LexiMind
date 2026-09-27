"""Synthetic checks of loss semantics; no pretrained checkpoint or research data."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.models.factory import ModelConfig, build_multitask_model
from src.models.heads import ClassificationHead
from src.models.losses import masked_binary_cross_entropy


def tiny_model(problem_type="single_label"):
    return build_multitask_model(
        SimpleNamespace(vocab_size=13, pad_token_id=0, config=SimpleNamespace(max_length=4)),
        num_emotions=2,
        num_topics=2,
        topic_problem_type=problem_type,
        config=ModelConfig(
            d_model=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_attention_heads=1,
            ffn_dim=8,
            dropout=0.0,
        ),
        load_pretrained=False,
    )


def test_masked_bce_matches_dense_and_selects_known_cells_before_computation():
    logits = torch.tensor([[1.0, -2.0], [0.5, 3.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    dense = F.binary_cross_entropy_with_logits(logits, labels)
    torch.testing.assert_close(masked_binary_cross_entropy(logits, labels), dense)
    torch.testing.assert_close(
        masked_binary_cross_entropy(logits, labels, torch.ones_like(labels, dtype=torch.bool)),
        dense,
    )
    mask = torch.tensor([[True, False], [True, False]])
    labels[~mask] = float("nan")
    loss = masked_binary_cross_entropy(logits, labels, mask)
    expected = F.binary_cross_entropy_with_logits(logits[mask], labels[mask])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(
        logits.grad[mask], (logits.detach()[mask].sigmoid() - labels[mask]) / 2
    )
    assert torch.equal(logits.grad[~mask], torch.zeros(2))


def test_all_unknown_is_finite_graph_zero_even_for_nonfinite_unknown_placeholders():
    logits = torch.tensor([[float("nan"), float("inf")]], requires_grad=True)
    labels = torch.tensor([[float("nan"), -999.0]])
    loss = masked_binary_cross_entropy(logits, labels, torch.zeros((1, 2), dtype=torch.bool))
    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_unknown_rows_do_not_dilute_loss_or_gradients_within_a_batch():
    logits = torch.tensor([[0.5, -0.5]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0]])
    loss = masked_binary_cross_entropy(logits, labels)
    loss.backward()
    extended = torch.cat([logits.detach(), torch.tensor([[4.0, -7.0]])]).requires_grad_()
    extended_loss = masked_binary_cross_entropy(
        extended,
        torch.cat([labels, torch.full((1, 2), float("nan"))]),
        torch.tensor([[True, True], [False, False]]),
    )
    extended_loss.backward()
    torch.testing.assert_close(extended_loss, loss)
    torch.testing.assert_close(extended.grad[:1], logits.grad)
    assert not extended.grad[1].any()


@pytest.mark.parametrize(
    "labels,mask,error",
    [
        (torch.tensor([1.0, 0.0]), None, "same B x C"),
        (torch.tensor([[1, 0]]), None, "floating point"),
        (torch.tensor([[1.0, 0.0]]), torch.tensor([[1, 0]]), "must be bool"),
        (torch.tensor([[1.0, 0.0]]), torch.tensor([True, False]), "same B x C"),
        (torch.tensor([[0.5, 0.0]]), None, "finite binary"),
        (torch.tensor([[float("nan"), 0.0]]), None, "finite binary"),
        (torch.tensor([[float("inf"), 0.0]]), None, "finite binary"),
        (torch.tensor([[-1.0, 0.0]]), None, "finite binary"),
    ],
)
def test_malformed_partial_targets_are_rejected(labels, mask, error):
    with pytest.raises(ValueError, match=error):
        masked_binary_cross_entropy(torch.zeros((1, 2)), labels, mask)


def test_mode_is_explicit_and_state_dict_keys_shapes_and_logits_are_unchanged():
    original = tiny_model()
    partial = tiny_model("multi_label")
    assert original.heads["topic"].problem_type == "single_label"
    assert original.heads["emotion"].problem_type == "multi_label"
    assert partial.heads["topic"].problem_type == "multi_label"
    partial.load_state_dict(original.state_dict(), strict=True)
    original.eval()
    partial.eval()
    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    torch.testing.assert_close(original("topic", inputs), partial("topic", inputs))
    labels = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError, match="labels with shape B"):
        original("topic", {**inputs, "labels": labels}, return_loss=True)
    with pytest.raises(ValueError, match="requires a multi_label"):
        original(
            "topic",
            {
                **inputs,
                "labels": torch.tensor([0]),
                "label_mask": torch.ones_like(labels, dtype=torch.bool),
            },
            return_loss=True,
        )
    with pytest.raises(ValueError, match="same B x C"):
        partial("topic", {**inputs, "labels": torch.tensor([0])}, return_loss=True)


def test_model_return_loss_uses_same_masked_bce_and_legacy_topic_ce():
    model = tiny_model("multi_label")
    inputs = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "labels": torch.tensor([[1.0, float("nan")], [0.0, 1.0]]),
        "label_mask": torch.tensor([[True, False], [True, True]]),
    }
    loss, logits = model("topic", inputs, return_loss=True)
    torch.testing.assert_close(
        loss, masked_binary_cross_entropy(logits, inputs["labels"], inputs["label_mask"])
    )
    dense_inputs = {
        "input_ids": inputs["input_ids"],
        "labels": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }
    emotion_loss, emotion_logits = model("emotion", dense_inputs, return_loss=True)
    torch.testing.assert_close(
        emotion_loss, F.binary_cross_entropy_with_logits(emotion_logits, dense_inputs["labels"])
    )
    model.heads["topic"].problem_type = "single_label"
    legacy_inputs = {"input_ids": inputs["input_ids"], "labels": torch.tensor([0, 1])}
    legacy_loss, legacy_logits = model("topic", legacy_inputs, return_loss=True)
    torch.testing.assert_close(legacy_loss, F.cross_entropy(legacy_logits, legacy_inputs["labels"]))


def test_invalid_mode_fails_before_allocating_the_factory_model():
    with pytest.raises(ValueError, match="topic_problem_type"):
        tiny_model("auto")
    with pytest.raises(ValueError, match="problem_type"):
        ClassificationHead(4, 2, problem_type="auto")
