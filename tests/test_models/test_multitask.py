import torch

from src.models.decoder import TransformerDecoder
from src.models.encoder import TransformerEncoder
from src.models.heads import ClassificationHead, LMHead, TokenClassificationHead
from src.models.multitask import MultiTaskModel


def test_multitask_encoder_classification_forward_and_loss():
    torch.manual_seed(0)
    vocab_size = 30
    d_model = 32
    num_layers = 2
    num_heads = 4
    d_ff = 64
    batch_size = 3
    seq_len = 8
    num_labels = 5

    enc = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0,
        max_len=seq_len,
        pad_token_id=0,
    )

    mt = MultiTaskModel(encoder=enc)
    head = ClassificationHead(d_model=d_model, num_labels=num_labels, pooler="mean", dropout=0.0)
    mt.add_head("sentiment", head)

    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    labels = torch.randint(0, num_labels, (batch_size,), dtype=torch.long)

    logits = mt.forward("sentiment", {"input_ids": input_ids})
    assert logits.shape == (batch_size, num_labels)

    loss, logits2 = mt.forward(
        "sentiment", {"input_ids": input_ids, "labels": labels}, return_loss=True
    )
    assert loss.item() >= 0
    # grads
    loss.backward()
    grads = [p.grad for p in mt.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_multitask_seq2seq_lm_forward_and_loss():
    torch.manual_seed(1)
    vocab_size = 40
    d_model = 32
    num_layers = 2
    num_heads = 4
    d_ff = 64
    batch_size = 2
    src_len = 7
    tgt_len = 6

    enc = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0,
        max_len=src_len,
        pad_token_id=0,
    )
    dec = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0,
        max_len=tgt_len,
        pad_token_id=0,
    )
    mt = MultiTaskModel(encoder=enc, decoder=dec)
    lm_head = LMHead(d_model=d_model, vocab_size=vocab_size, tie_embedding=None)
    mt.add_head("summarize", lm_head)

    src_ids = torch.randint(1, vocab_size, (batch_size, src_len), dtype=torch.long)
    # for training: provide decoder inputs (typically shifted right) and labels
    tgt_ids = torch.randint(1, vocab_size, (batch_size, tgt_len), dtype=torch.long)
    labels = tgt_ids.clone()

    logits = mt.forward("summarize", {"src_ids": src_ids, "tgt_ids": tgt_ids})
    assert logits.shape == (batch_size, tgt_len, vocab_size)

    loss, logits2 = mt.forward(
        "summarize", {"src_ids": src_ids, "tgt_ids": tgt_ids, "labels": labels}, return_loss=True
    )
    assert loss.item() >= 0
    loss.backward()
    grads = [p.grad for p in mt.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_token_classification_forward_and_loss():
    torch.manual_seed(2)
    vocab_size = 20
    d_model = 24
    num_layers = 2
    num_heads = 4
    d_ff = 64
    batch_size = 2
    seq_len = 5
    num_labels = 7

    enc = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0,
        max_len=seq_len,
        pad_token_id=0,
    )
    mt = MultiTaskModel(encoder=enc)
    head = TokenClassificationHead(d_model=d_model, num_labels=num_labels, dropout=0.0)
    mt.add_head("ner", head)

    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    labels = torch.randint(0, num_labels, (batch_size, seq_len), dtype=torch.long)

    logits = mt.forward("ner", {"input_ids": input_ids})
    assert logits.shape == (batch_size, seq_len, num_labels)

    loss, logits2 = mt.forward("ner", {"input_ids": input_ids, "labels": labels}, return_loss=True)
    assert loss.item() >= 0
    loss.backward()
    grads = [p.grad for p in mt.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
