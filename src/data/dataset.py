"""Task examples, bounded JSONL reads and split preparation.

JSONL may be indexed without retaining document text in memory. Legacy JSON
arrays still load eagerly; convert them to JSONL for bounded text storage.
"""

from __future__ import annotations

import json
import operator
from array import array
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar, cast, overload

import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import Dataset

T = TypeVar("T")
TASK_NAMES = ("summarization", "emotion", "topic")


@dataclass(slots=True)
class SummarizationExample:
    source: str
    summary: str
    domain: str = "unknown"


@dataclass(slots=True)
class EmotionExample:
    text: str
    emotions: Sequence[str]


@dataclass(slots=True)
class TopicExample:
    text: str
    topic: str


def _validate_limit(limit: int | None) -> None:
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError("Sample limit must be a nonnegative integer or None")


def _snapshot(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _parse_record(raw: bytes | str, path: Path, line: int, required: Sequence[str]) -> dict:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to parse JSON in '{path}' at line {line}: {exc}") from exc
    _validate_record(payload, path, f"line {line}", required)
    return cast(dict, payload)


def _validate_record(payload: object, path: Path, location: str, required: Sequence[str]) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {location} of '{path}'")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(
            f"Missing required keys ({', '.join(sorted(missing))}) at {location} of '{path}'"
        )


class IndexedJsonl(Sequence[T], Generic[T]):
    """Read-only row index with O(rows) offsets and O(batch text) decoded memory.

    Row decoding/validation is deferred until access. No token or document cache
    is retained. Each batch opens its own handle, so fork/spawn workers never
    share seek state or serialize corpus text. A changed source must be reloaded.
    This file-stat guard detects ordinary edits, not adversarial tampering; the
    separate admission manifest is responsible for content hashes/provenance.
    """

    def __init__(
        self,
        path: Path,
        constructor: Callable[[dict], T],
        required: Sequence[str],
        *,
        limit: int | None = None,
    ):
        _validate_limit(limit)
        self.path = path.resolve()
        self.constructor = constructor
        self.required = tuple(required)
        self._snapshot = _snapshot(self.path)
        self._offsets = array("Q")
        self._lines = array("Q")
        with self.path.open("rb") as handle:
            line_number = 0
            while limit is None or len(self._offsets) < limit:
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                line_number += 1
                if raw.strip():
                    self._offsets.append(offset)
                    self._lines.append(line_number)
        self._check_source()

    def _check_source(self) -> None:
        if _snapshot(self.path) != self._snapshot:
            raise RuntimeError(f"Dataset changed after indexing: '{self.path}'; reload its index")

    def __len__(self) -> int:
        return len(self._offsets)

    def _index(self, index: int) -> int:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Dataset index out of range")
        return index

    def read_many(self, indices: Iterable[int]) -> list[T]:
        self._check_source()
        with self.path.open("rb") as handle:
            result = []
            for requested in indices:
                index = self._index(requested)
                handle.seek(self._offsets[index])
                payload = _parse_record(
                    handle.readline(), self.path, self._lines[index], self.required
                )
                result.append(self.constructor(payload))
        self._check_source()
        return result

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> list[T]: ...

    def __getitem__(self, index: int | slice) -> T | list[T]:
        if isinstance(index, slice):
            return self.read_many(range(*index.indices(len(self))))
        return self.read_many([index])[0]

    def __iter__(self) -> Iterator[T]:
        self._check_source()
        remaining = len(self)
        with self.path.open("rb") as handle:
            line = 0
            while remaining:
                raw = handle.readline()
                if not raw:
                    break
                line += 1
                if raw.strip():
                    yield self.constructor(_parse_record(raw, self.path, line, self.required))
                    remaining -= 1
        self._check_source()


class _ExampleDataset(Dataset[T], Generic[T]):
    def __init__(self, examples: Iterable[T]) -> None:
        # Keep immutable file-backed storage; preserve the old list snapshot API.
        self._examples = examples if isinstance(examples, IndexedJsonl) else list(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> T:
        return self._examples[index]

    def __getitems__(self, indices: list[int]) -> list[T]:
        # PyTorch's map-style batch fetch opens a single handle per minibatch.
        if isinstance(self._examples, IndexedJsonl):
            return self._examples.read_many(indices)
        return [self._examples[index] for index in indices]


class SummarizationDataset(_ExampleDataset[SummarizationExample]):
    """Encoder-decoder samples, with tokenization deferred to the collator."""


class EmotionDataset(_ExampleDataset[EmotionExample]):
    def __init__(
        self, examples: Iterable[EmotionExample], *, binarizer: MultiLabelBinarizer | None = None
    ) -> None:
        super().__init__(examples)
        if binarizer is None:
            self._binarizer = MultiLabelBinarizer().fit(
                example.emotions for example in self._examples
            )
        else:
            if not hasattr(binarizer, "classes_"):
                raise ValueError(
                    "Provided MultiLabelBinarizer must be pre-fitted with 'classes_' attribute."
                )
            self._binarizer = binarizer

    @property
    def binarizer(self) -> MultiLabelBinarizer:
        return self._binarizer

    @property
    def emotion_classes(self) -> list[str]:
        return list(self._binarizer.classes_)


class TopicDataset(_ExampleDataset[TopicExample]):
    def __init__(
        self, examples: Iterable[TopicExample], *, encoder: LabelEncoder | None = None
    ) -> None:
        super().__init__(examples)
        if encoder is None:
            self._encoder = LabelEncoder().fit(
                sorted({example.topic for example in self._examples})
            )
        else:
            if not hasattr(encoder, "classes_"):
                raise ValueError(
                    "Provided LabelEncoder must be pre-fitted with 'classes_' attribute."
                )
            self._encoder = encoder

    @property
    def encoder(self) -> LabelEncoder:
        return self._encoder

    @property
    def topic_classes(self) -> list[str]:
        return list(self._encoder.classes_)


# --------------- Calibration Split ---------------
#
# The GoEmotions validation set serves two distinct purposes: (1) model
# selection during training (early stopping on combined val loss), and
# (2) per-class threshold calibration for emotion evaluation. Using the
# same samples for both creates an optimistic bias in tuned-threshold
# metrics. ``split_emotion_val`` deterministically partitions val into a
# model-selection half and a calibration half so the threshold-tuning
# step in ``scripts/evaluate.py`` uses samples the model never influenced
# via early stopping. The split is driven by a fixed seed so training and
# evaluation always agree on which half is which.

EMOTION_CALIBRATION_SPLIT_SEED = 20260416


def split_emotion_val(
    examples: Sequence[EmotionExample],
    *,
    seed: int = EMOTION_CALIBRATION_SPLIT_SEED,
    calibration_fraction: float = 0.5,
) -> tuple[list[EmotionExample], list[EmotionExample]]:
    """Deterministically split val examples into (model_selection, calibration).

    Both training (early stopping) and evaluation (threshold tuning) must
    call this with the same seed/fraction so they agree on which samples
    belong to which half.

    Args:
        examples: Full emotion validation split.
        seed: Random seed controlling the shuffle.
        calibration_fraction: Fraction of val assigned to the calibration
            half (default 0.5).

    Returns:
        (model_selection_half, calibration_half)
    """
    import random as _random

    rng = _random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    n_calib = int(round(len(examples) * calibration_fraction))
    calib_idx = set(indices[:n_calib])
    model_sel: list[EmotionExample] = []
    calib: list[EmotionExample] = []
    for i, ex in enumerate(examples):
        if i in calib_idx:
            calib.append(ex)
        else:
            model_sel.append(ex)
    return model_sel, calib


def _load_jsonl_generic(
    path: str | Path,
    constructor: Callable[[dict], T],
    required_keys: Sequence[str],
    *,
    limit: int | None = None,
    lazy: bool = False,
) -> Sequence[T]:
    _validate_limit(limit)
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file '{data_path}' does not exist")
    if not data_path.is_file():
        raise ValueError(f"Dataset path '{data_path}' is not a file")
    if limit == 0:
        return []
    with data_path.open("rb") as handle:
        # Detect legacy arrays without decoding or retaining corpus text.
        first = b""
        while not first:
            char = handle.read(1)
            if not char:
                raise ValueError(f"Dataset file '{data_path}' is empty or contains only whitespace")
            if not char.isspace():
                first = char
        handle.seek(0)
        if first == b"[":
            try:
                payloads = json.load(handle)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(f"Failed to parse JSON in '{data_path}': {exc}") from exc
            items = []
            for idx, payload in enumerate(payloads[:limit] if limit is not None else payloads):
                _validate_record(payload, data_path, f"index {idx}", required_keys)
                items.append(constructor(payload))
            return items
        if not lazy:
            items = []
            for line, raw in enumerate(handle, start=1):
                if raw.strip():
                    items.append(constructor(_parse_record(raw, data_path, line, required_keys)))
                    if limit is not None and len(items) == limit:
                        break
            return items
    return IndexedJsonl(data_path, constructor, required_keys, limit=limit)


def _summary(payload: dict) -> SummarizationExample:
    return SummarizationExample(
        payload["source"], payload["summary"], payload.get("type", payload.get("domain", "unknown"))
    )


def _emotion(payload: dict) -> EmotionExample:
    return EmotionExample(payload["text"], payload.get("emotions", []))


def _topic(payload: dict) -> TopicExample:
    return TopicExample(payload["text"], payload["topic"])


def load_summarization_jsonl(
    path: str | Path, *, limit: int | None = None, lazy: bool = False
) -> Sequence[SummarizationExample]:
    return _load_jsonl_generic(path, _summary, ("source", "summary"), limit=limit, lazy=lazy)


def load_emotion_jsonl(
    path: str | Path, *, limit: int | None = None, lazy: bool = False
) -> Sequence[EmotionExample]:
    return _load_jsonl_generic(path, _emotion, ("text",), limit=limit, lazy=lazy)


def load_topic_jsonl(
    path: str | Path, *, limit: int | None = None, lazy: bool = False
) -> Sequence[TopicExample]:
    return _load_jsonl_generic(path, _topic, ("text", "topic"), limit=limit, lazy=lazy)


def resolve_split_path(directory: Path, split: str) -> Path | None:
    aliases = ("val", "validation") if split in {"val", "validation"} else (split,)
    return next(
        (
            directory / f"{alias}.jsonl"
            for alias in aliases
            if (directory / f"{alias}.jsonl").is_file()
        ),
        None,
    )


def validate_task_directories(
    processed: Mapping, tasks: Sequence[str], *, split: str = "train"
) -> dict[str, Path]:
    """Require explicit paths only for active tasks, before any model/tokenizer load."""
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in TASK_NAMES for task in tasks):
        raise ValueError(f"Choose a nonempty, unique subset of tasks: {', '.join(TASK_NAMES)}")
    directories = {}
    for task in tasks:
        value = processed.get(task)
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(
                f"Set an explicit reviewed directory for '{task}' (data.processed.{task}=/path/to/splits); no corpus is selected by default"
            )
        directory = Path(value)
        if not directory.is_dir() or resolve_split_path(directory, split) is None:
            raise ValueError(
                f"Dataset directory for '{task}' must contain {split}.jsonl (val/validation aliases supported): {directory}"
            )
        directories[task] = directory
    return directories


def load_splits(
    data_dir: Path,
    loader_fn: Callable,
    *,
    include_test: bool = False,
    include_validation: bool = True,
    limits: Mapping[str, int | None] | None = None,
    lazy: bool = False,
) -> dict[str, Sequence]:
    """Read requested splits only; test is opt-in, limits apply while reading JSONL."""
    names = ["train"] + (["val"] if include_validation else []) + (["test"] if include_test else [])
    splits = {}
    for name in names:
        path = resolve_split_path(data_dir, name)
        if path is not None:
            kwargs: dict[str, Any] = {}
            if limits is not None:
                kwargs["limit"] = limits.get(name)
            if lazy:
                kwargs["lazy"] = True
            splits[name] = loader_fn(str(path), **kwargs)
    return splits


def validate_known_labels(labels: Sequence[str], known: set[str], task: str) -> None:
    """Reject malformed/unknown labels before encoding; sklearn may silently drop them."""
    if (
        isinstance(labels, (str, bytes))
        or not isinstance(labels, Sequence)
        or any(not isinstance(label, str) or not label.strip() for label in labels)
    ):
        raise ValueError(f"{task} labels must be a sequence of nonblank strings")
    unknown = sorted(set(labels) - known)
    if unknown:
        raise ValueError(f"Unknown {task} labels outside the training vocabulary: {unknown}")


def _training_vocabulary(directory: Path, task: str, full_training: Iterable | None) -> list[str]:
    """Prefer an ordered sidecar; otherwise discover labels only from complete training.

    Sidecars are nonempty JSON arrays of unique nonblank strings. They declare
    model-column order, so checkpoint continuation must match that order exactly.
    No validation/test labels enter vocabulary discovery. Unknown labels in any
    consumed batch are separately rejected by the collators.
    """
    sidecar = directory / "labels.json"
    if sidecar.exists():
        with sidecar.open(encoding="utf-8") as handle:
            labels = json.load(handle)
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError(
                f"{sidecar} must be a nonempty JSON array of unique nonblank label strings"
            )
        return labels
    if full_training is None:
        loader = load_emotion_jsonl if task == "emotion" else load_topic_jsonl
        full_training = loader(directory / "train.jsonl", lazy=True)
    found: set[str] = set()
    for example in full_training:
        labels = (
            cast(EmotionExample, example).emotions
            if task == "emotion"
            else [cast(TopicExample, example).topic]
        )
        # Validate types without treating a newly discovered label as unknown.
        if (
            isinstance(labels, (str, bytes))
            or not isinstance(labels, Sequence)
            or any(not isinstance(label, str) or not label.strip() for label in labels)
        ):
            raise ValueError(f"{task} training labels must be nonblank strings")
        found.update(labels)
    if not found:
        raise ValueError(f"Enabled {task} task requires at least one training label")
    return sorted(found)


def load_training_datasets(
    processed: Mapping,
    tasks: Sequence[str],
    *,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    include_validation: bool = True,
) -> tuple[dict[str, Dataset], dict[str, Dataset]]:
    """Prepare active task datasets once, retaining lazy training/validation text.

    The emotion validation partition is formed from the complete split before a
    model-selection cap, keeping calibration membership independent of run size.
    Classification vocabularies use ordered labels.json sidecars when present;
    otherwise a streaming pass discovers the full training vocabulary before caps.
    """
    _validate_limit(max_train_samples)
    _validate_limit(max_val_samples)
    directories = validate_task_directories(processed, tasks)
    loaders = {
        "summarization": load_summarization_jsonl,
        "emotion": load_emotion_jsonl,
        "topic": load_topic_jsonl,
    }
    train: dict[str, Dataset] = {}
    validation: dict[str, Dataset] = {}
    for task, directory in directories.items():
        splits = load_splits(
            directory,
            loaders[task],
            include_validation=include_validation,
            limits={
                "train": max_train_samples,
                "val": None if task == "emotion" else max_val_samples,
            },
            lazy=True,
        )
        if not splits["train"]:
            raise ValueError(
                f"Training split for '{task}' is empty; choose a nonempty reviewed split and positive sample limit"
            )
        val = splits.get("val", [])
        vocabulary: list[str] = []
        if task != "summarization":
            vocabulary = _training_vocabulary(
                directory, task, splits["train"] if max_train_samples is None else None
            )
        if task == "emotion":
            val, _ = split_emotion_val(val)
            if max_val_samples is not None:
                val = val[:max_val_samples]
            binarizer = MultiLabelBinarizer(classes=vocabulary).fit([])
            emotion_train = EmotionDataset(splits["train"], binarizer=binarizer)
            train[task] = emotion_train
            validation[task] = EmotionDataset(val, binarizer=emotion_train.binarizer)
        elif task == "topic":
            encoder = LabelEncoder()
            # String labels use a mapping in LabelEncoder.transform; preserve the
            # declared model-column order rather than fit()'s automatic sorting.
            encoder.classes_ = np.asarray(vocabulary, dtype=object)
            topic_train = TopicDataset(splits["train"], encoder=encoder)
            train[task] = topic_train
            validation[task] = TopicDataset(val, encoder=topic_train.encoder)
        else:
            train[task] = SummarizationDataset(splits["train"])
            validation[task] = SummarizationDataset(val)
    return train, {task: dataset for task, dataset in validation.items() if len(dataset)}
