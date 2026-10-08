"""
Inference script for the LexiMind multitask model.

Command-line interface for running summarization, emotion detection, and topic
classification on arbitrary text inputs.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import EmotionPrediction, TopicPrediction, create_inference_pipeline


def _load_texts(positional: List[str], file_path: Path | None) -> List[str]:
    texts = [text for text in positional if text]
    if file_path is not None:
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        with file_path.open("r", encoding="utf-8") as handle:
            texts.extend([line.strip() for line in handle if line.strip()])
    if not texts:
        raise ValueError("No input texts provided. Pass text arguments or use --file.")
    return texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LexiMind multitask inference.")
    parser.add_argument("text", nargs="*", help="Input text(s) to analyse.")
    parser.add_argument("--file", type=Path, help="Path to a file containing one text per line.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best.pt"),
        help="Path to the model checkpoint produced during training.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="JSON file containing emotion/topic label vocabularies.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Optional path to a tokenizer directory exported during training.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Model architecture config used to rebuild the transformer stack.",
    )
    parser.add_argument("--device", default="cpu", help="Device to run inference on (cpu or cuda).")
    parser.add_argument(
        "--summary-max-length",
        type=int,
        default=None,
        help="Optional maximum length for generated summaries.",
    )
    parser.add_argument("--title", action="append", help="Book title; repeat once per description.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Explicit sigmoid score threshold for multi-label book fields (0 to 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = _load_texts(args.text, args.file)

    paired_labels = args.checkpoint.parent / "labels.json"
    labels_path = args.labels or (
        paired_labels if paired_labels.exists() else Path("artifacts/labels.json")
    )
    pipeline, labels = create_inference_pipeline(
        checkpoint_path=args.checkpoint,
        labels_path=labels_path,
        tokenizer_dir=args.tokenizer,
        model_config_path=args.model_config,
        device=args.device,
        summary_max_length=args.summary_max_length,
    )

    if labels.topic_problem_type == "multi_label":
        if args.title is None or len(args.title) != len(texts):
            raise ValueError("Book inference requires one explicit --title per input description")
        if args.threshold is None:
            raise ValueError("Book inference requires an explicit --threshold")
        books = [
            {"title": title, "description": text}
            for title, text in zip(args.title, texts, strict=True)
        ]
        predictions = pipeline.predict_book_fields(books, thresholds=args.threshold)
        packaged = [
            {**book, "fields": prediction.fields, "scores": prediction.scores}
            for book, prediction in zip(books, predictions, strict=True)
        ]
        print(json.dumps(packaged, indent=2, ensure_ascii=False))
        return
    if args.title is not None or args.threshold is not None:
        raise ValueError("--title and --threshold require a multi-label book checkpoint")

    results = pipeline.batch_predict(texts)
    summaries = cast(List[str], results["summaries"])
    emotion_preds = cast(List[EmotionPrediction], results["emotion"])
    topic_preds = cast(List[TopicPrediction], results["topic"])

    packaged = []
    for idx, text in enumerate(texts):
        emotion = emotion_preds[idx]
        topic = topic_preds[idx]
        packaged.append(
            {
                "text": text,
                "summary": summaries[idx],
                "emotion": {
                    "labels": emotion.labels,
                    "scores": emotion.scores,
                },
                "topic": {
                    "label": topic.label,
                    "confidence": topic.confidence,
                },
            }
        )

    print(json.dumps(packaged, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
