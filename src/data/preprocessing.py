"""Lightweight preprocessing utilities built around the in-repo transformer."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from ..models.decoder import TransformerDecoder
from ..models.encoder import TransformerEncoder

SPECIAL_TOKENS: Tuple[str, str, str, str] = ("<pad>", "<bos>", "<eos>", "<unk>")


def _normalize(text: str, lowercase: bool) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if lowercase:
        text = text.lower()
    return text


def _basic_tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b|[.,;:?!]", text)


class TransformerTokenizer:
    """Minimal tokenizer that keeps vocabulary aligned with the custom transformer."""

    def __init__(
        self,
        stoi: Dict[str, int],
        itos: List[str],
        specials: Sequence[str] = SPECIAL_TOKENS,
        lowercase: bool = True,
    ) -> None:
        self.stoi = stoi
        self.itos = itos
        self.specials = tuple(specials)
        self.lowercase = lowercase
        self.pad_id = self._lookup(self.specials[0])
        self.bos_id = self._lookup(self.specials[1])
        self.eos_id = self._lookup(self.specials[2])
        self.unk_id = self._lookup(self.specials[3])

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        min_freq: int = 1,
        lowercase: bool = True,
        specials: Sequence[str] = SPECIAL_TOKENS,
    ) -> "TransformerTokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            normalized = _normalize(text, lowercase)
            counter.update(_basic_tokenize(normalized))

        ordered_specials = list(dict.fromkeys(specials))
        itos: List[str] = ordered_specials.copy()
        for token, freq in counter.most_common():
            if freq < min_freq:
                continue
            if token in itos:
                continue
            itos.append(token)

        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos, specials=ordered_specials, lowercase=lowercase)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def tokenize(self, text: str) -> List[str]:
        normalized = _normalize(text, self.lowercase)
        return _basic_tokenize(normalized)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        tokens = self.tokenize(text)
        pieces = [self.stoi.get(tok, self.unk_id) for tok in tokens]
        if add_special_tokens:
            pieces = [self.bos_id] + pieces + [self.eos_id]

        if max_length is not None and len(pieces) > max_length:
            if add_special_tokens and max_length >= 2:
                inner_max = max_length - 2
                trimmed = pieces[1:-1][:inner_max]
                pieces = [self.bos_id] + trimmed + [self.eos_id]
            else:
                pieces = pieces[:max_length]
        return pieces

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        tokens: List[str] = []
        for idx in ids:
            if idx < 0 or idx >= len(self.itos):
                continue
            token = self.itos[idx]
            if skip_special_tokens and token in self.specials:
                continue
            tokens.append(token)
        return " ".join(tokens).strip()

    def pad_batch(
        self,
        sequences: Sequence[Sequence[int]],
        pad_to_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not sequences:
            raise ValueError("pad_batch requires at least one sequence")
        if pad_to_length is None:
            pad_to_length = max(len(seq) for seq in sequences)
        padded: List[List[int]] = []
        mask: List[List[int]] = []
        for seq in sequences:
            trimmed = list(seq[:pad_to_length])
            pad_len = pad_to_length - len(trimmed)
            padded.append(trimmed + [self.pad_id] * pad_len)
            mask.append([1] * len(trimmed) + [0] * pad_len)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def save(self, path: Path) -> None:
        payload = {
            "itos": self.itos,
            "specials": list(self.specials),
            "lowercase": self.lowercase,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TransformerTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        itos = list(data["itos"])
        stoi = {token: idx for idx, token in enumerate(itos)}
        specials = data.get("specials", list(SPECIAL_TOKENS))
        lowercase = bool(data.get("lowercase", True))
        return cls(stoi=stoi, itos=itos, specials=specials, lowercase=lowercase)

    def _lookup(self, token: str) -> int:
        if token not in self.stoi:
            raise ValueError(f"token '{token}' missing from vocabulary")
        return self.stoi[token]


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    lengths: List[int]


class TextPreprocessor:
    """Prepares text so it can flow directly into the custom transformer stack."""

    def __init__(
        self,
        max_length: int = 512,
        tokenizer: Optional[TransformerTokenizer] = None,
        *,
        min_freq: int = 1,
        lowercase: bool = True,
    ) -> None:
        self.max_length = max_length
        self.min_freq = min_freq
        self.lowercase = lowercase
        self.tokenizer = tokenizer

    def clean_text(self, text: str) -> str:
        return _normalize(text, self.lowercase)

    def fit_tokenizer(self, texts: Iterable[str]) -> TransformerTokenizer:
        cleaned = [self.clean_text(text) for text in texts]
        self.tokenizer = TransformerTokenizer.build(
            cleaned,
            min_freq=self.min_freq,
            lowercase=False,
        )
        return self.tokenizer

    def encode(self, text: str, *, add_special_tokens: bool = True) -> List[int]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not fitted")
        cleaned = self.clean_text(text)
        return self.tokenizer.encode(cleaned, add_special_tokens=add_special_tokens, max_length=self.max_length)

    def batch_encode(self, texts: Sequence[str]) -> Batch:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not fitted")
        sequences = [self.encode(text) for text in texts]
        lengths = [len(seq) for seq in sequences]
        input_ids, attention_mask = self.tokenizer.pad_batch(sequences, pad_to_length=self.max_length)
        return Batch(input_ids=input_ids, attention_mask=attention_mask, lengths=lengths)

    def build_encoder(self, **encoder_kwargs) -> TransformerEncoder:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not fitted")
        return TransformerEncoder(
            vocab_size=self.tokenizer.vocab_size,
            max_len=self.max_length,
            pad_token_id=self.tokenizer.pad_id,
            **encoder_kwargs,
        )

    def build_decoder(self, **decoder_kwargs) -> TransformerDecoder:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not fitted")
        return TransformerDecoder(
            vocab_size=self.tokenizer.vocab_size,
            max_len=self.max_length,
            pad_token_id=self.tokenizer.pad_id,
            **decoder_kwargs,
        )

    def save_tokenizer(self, path: Path) -> None:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not fitted")
        self.tokenizer.save(path)

    def load_tokenizer(self, path: Path) -> TransformerTokenizer:
        self.tokenizer = TransformerTokenizer.load(path)
        return self.tokenizer

    def chunk_text(self, text: str, *, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be larger than overlap")
        words = self.clean_text(text).split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += chunk_size - overlap
        return chunks

    def save_book_chunks(
        self,
        input_path: Path,
        out_dir: Path,
        *,
        chunk_size: int = 1000,
        overlap: int = 100,
    ) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_text = input_path.read_text(encoding="utf-8", errors="ignore")
        chunks = self.chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
        out_file = out_dir / f"{input_path.stem}.json"
        out_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_file