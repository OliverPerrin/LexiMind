import torch
import pytest
import torch.nn as nn
from src.models.heads import (
    ClassificationHead,
    TokenClassificationHead,
    LMHead,
    ProjectionHead,
)


def test_classification_head_shapes_and_dropout():
    torch.manual_seed(0)
    d_model = 64
    num_labels = 5
    batch_size = 3
    seq_len = 10

    head = ClassificationHead(d_model=d_model, num_labels=num_labels, pooler="mean", dropout=0.5)
    head.train()
    x = torch.randn(batch_size, seq_len, d_model)

    out1 = head(x)
    out2 = head(x)
    # With dropout in train mode, outputs should usually differ
    assert out1.shape == (batch_size, num_labels)
    assert out2.shape == (batch_size, num_labels)
    assert not torch.allclose(out1, out2)

    head.eval()
    out3 = head(x)
    out4 = head(x)
    assert torch.allclose(out3, out4), "Eval mode should be deterministic"


def test_token_classification_head_shapes_and_grads():
    torch.manual_seed(1)
    d_model = 48
    num_labels = 7
    batch_size = 2
    seq_len = 6

    head = TokenClassificationHead(d_model=d_model, num_labels=num_labels, dropout=0.0)
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    out = head(x)
    assert out.shape == (batch_size, seq_len, num_labels)

    loss = out.sum()
    loss.backward()
    grads = [p.grad for name, p in head.named_parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_lm_head_tie_weights_and_shapes():
    torch.manual_seed(2)
    vocab_size = 50
    d_model = 32
    batch_size = 2
    seq_len = 4

    embedding = nn.Embedding(vocab_size, d_model)
    lm_tied = LMHead(d_model=d_model, vocab_size=vocab_size, tie_embedding=embedding)
    lm_untied = LMHead(d_model=d_model, vocab_size=vocab_size, tie_embedding=None)

    hidden = torch.randn(batch_size, seq_len, d_model)

    # Shapes
    logits_tied = lm_tied(hidden)
    logits_untied = lm_untied(hidden)
    assert logits_tied.shape == (batch_size, seq_len, vocab_size)
    assert logits_untied.shape == (batch_size, seq_len, vocab_size)

    # Weight tying: projection weight should be the same object as embedding.weight
    assert lm_tied.proj.weight is embedding.weight

    # Grad flows through tied weights
    loss = logits_tied.sum()
    loss.backward()
    assert embedding.weight.grad is not None


def test_projection_head_2d_and_3d_behavior_and_grad():
    torch.manual_seed(3)
    d_model = 40
    proj_dim = 16
    batch_size = 2
    seq_len = 5

    head = ProjectionHead(d_model=d_model, proj_dim=proj_dim, hidden_dim=64, dropout=0.0)
    # 2D input
    vec = torch.randn(batch_size, d_model, requires_grad=True)
    out2 = head(vec)
    assert out2.shape == (batch_size, proj_dim)

    # 3D input
    seq = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    out3 = head(seq)
    assert out3.shape == (batch_size, seq_len, proj_dim)

    # Grad flow
    loss = out3.sum()
    loss.backward()
    grads = [p.grad for p in head.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)