"""Dataset definitions for the LexiMind multitask training pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, TypeVar

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import Dataset


@dataclass
class SummarizationExample:
    """Container for abstractive summarization samples."""

    source: str
    summary: str


@dataclass
class EmotionExample:
    """Container for multi-label emotion classification samples."""

    text: str
    emotions: Sequence[str]


@dataclass
class TopicExample:
    """Container for topic clustering / classification samples."""

    text: str
    topic: str


class SummarizationDataset(Dataset[SummarizationExample]):
    """Dataset yielding encoder-decoder training pairs."""

    def __init__(self, examples: Iterable[SummarizationExample]) -> None:
        self._examples = list(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> SummarizationExample:
        return self._examples[index]


class EmotionDataset(Dataset[EmotionExample]):
    """Dataset that owns a scikit-learn MultiLabelBinarizer for emissions."""

    def __init__(
        self,
        examples: Iterable[EmotionExample],
        *,
        binarizer: MultiLabelBinarizer | None = None,
    ) -> None:
        self._examples = list(examples)
        all_labels = [example.emotions for example in self._examples]
        if binarizer is None:
            self._binarizer = MultiLabelBinarizer()
            self._binarizer.fit(all_labels)
        else:
            self._binarizer = binarizer
            if not hasattr(self._binarizer, "classes_"):
                raise ValueError(
                    "Provided MultiLabelBinarizer must be pre-fitted with 'classes_' attribute."
                )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> EmotionExample:
        return self._examples[index]

    @property
    def binarizer(self) -> MultiLabelBinarizer:
        return self._binarizer

    @property
    def emotion_classes(self) -> List[str]:
        return list(self._binarizer.classes_)


class TopicDataset(Dataset[TopicExample]):
    """Dataset that owns a LabelEncoder for topic ids."""

    def __init__(
        self,
        examples: Iterable[TopicExample],
        *,
        encoder: LabelEncoder | None = None,
    ) -> None:
        self._examples = list(examples)
        topics = [example.topic for example in self._examples]
        if encoder is None:
            self._encoder = LabelEncoder().fit(topics)
        else:
            self._encoder = encoder
            if not hasattr(self._encoder, "classes_"):
                raise ValueError(
                    "Provided LabelEncoder must be pre-fitted with 'classes_' attribute."
                )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> TopicExample:
        return self._examples[index]

    @property
    def encoder(self) -> LabelEncoder:
        return self._encoder

    @property
    def topic_classes(self) -> List[str]:
        return list(self._encoder.classes_)


T = TypeVar("T")


def _safe_json_load(handle, path: Path) -> object:
    try:
        return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON in '{path}': {exc}") from exc


def _safe_json_loads(data: str, path: Path, line_number: int) -> object:
    try:
        return json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON in '{path}' at line {line_number}: {exc}") from exc


def _validate_keys(
    payload: dict,
    required_keys: Sequence[str],
    position: int,
    *,
    path: Path,
    is_array: bool = False,
) -> None:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        keys = ", ".join(sorted(missing))
        location = "index" if is_array else "line"
        raise KeyError(f"Missing required keys ({keys}) at {location} {position} of '{path}'")


def _load_jsonl_generic(
    path: str,
    constructor: Callable[[dict], T],
    required_keys: Sequence[str],
) -> List[T]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file '{data_path}' does not exist")
    if not data_path.is_file():
        raise ValueError(f"Dataset path '{data_path}' is not a file")

    items: List[T] = []
    with data_path.open("r", encoding="utf-8") as handle:
        first_non_ws = ""
        while True:
            pos = handle.tell()
            char = handle.read(1)
            if not char:
                break
            if not char.isspace():
                first_non_ws = char
                handle.seek(pos)
                break
        if not first_non_ws:
            raise ValueError(f"Dataset file '{data_path}' is empty or contains only whitespace")

        if first_non_ws == "[":
            payloads = _safe_json_load(handle, data_path)
            if not isinstance(payloads, list):
                raise ValueError(
                    f"Expected a JSON array in '{data_path}' but found {type(payloads).__name__}"
                )
            for idx, payload in enumerate(payloads):
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Expected objects in array for '{data_path}', found {type(payload).__name__} at index {idx}"
                    )
                _validate_keys(payload, required_keys, idx, path=data_path, is_array=True)
                items.append(constructor(payload))
        else:
            handle.seek(0)
            line_number = 0
            for line in handle:
                line_number += 1
                if not line.strip():
                    continue
                payload = _safe_json_loads(line, data_path, line_number)
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Expected JSON object per line in '{data_path}', found {type(payload).__name__} at line {line_number}"
                    )
                _validate_keys(payload, required_keys, line_number, path=data_path)
                items.append(constructor(payload))

    return items


def load_summarization_jsonl(path: str) -> List[SummarizationExample]:
    return _load_jsonl_generic(
        path,
        lambda payload: SummarizationExample(source=payload["source"], summary=payload["summary"]),
        required_keys=("source", "summary"),
    )


def load_emotion_jsonl(path: str) -> List[EmotionExample]:
    return _load_jsonl_generic(
        path,
        lambda payload: EmotionExample(text=payload["text"], emotions=payload.get("emotions", [])),
        required_keys=("text",),
    )


def load_topic_jsonl(path: str) -> List[TopicExample]:
    return _load_jsonl_generic(
        path,
        lambda payload: TopicExample(text=payload["text"], topic=payload["topic"]),
        required_keys=("text", "topic"),
    )
