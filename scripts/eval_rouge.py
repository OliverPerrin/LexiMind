"""Utility script to evaluate LexiMind summaries with ROUGE."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, List, Sequence, Tuple

import sys

from rouge_score import rouge_scorer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.factory import create_inference_pipeline


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate LexiMind summaries with ROUGE metrics.")
	parser.add_argument("data", type=Path, help="Path to JSONL file with source text and gold summaries.")
	parser.add_argument("checkpoint", type=Path, help="Path to the trained checkpoint (e.g., checkpoints/best.pt).")
	parser.add_argument("labels", type=Path, help="Path to label metadata (e.g., artifacts/labels.json).")
	parser.add_argument(
		"--tokenizer-dir",
		type=Path,
		default=Path("artifacts/hf_tokenizer"),
		help="Directory containing the saved tokenizer artifacts.",
	)
	parser.add_argument(
		"--model-config",
		type=Path,
		default=None,
		help="Optional YAML config describing the model architecture.",
	)
	parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu or cuda).")
	parser.add_argument("--batch-size", type=int, default=8, help="Number of samples per inference batch.")
	parser.add_argument(
		"--max-samples",
		type=int,
		default=None,
		help="If provided, limit evaluation to the first N samples for quick smoke tests.",
	)
	parser.add_argument(
		"--max-length",
		type=int,
		default=128,
		help="Maximum length to pass into the summarization head during generation.",
	)
	parser.add_argument(
		"--metrics",
		type=str,
		nargs="+",
		default=("rouge1", "rouge2", "rougeL"),
		help="ROUGE metrics to compute.",
	)
	parser.add_argument(
		"--source-field",
		type=str,
		default="source",
		help="Field name containing the input document in the JSONL examples.",
	)
	parser.add_argument(
		"--target-field",
		type=str,
		default="summary",
		help="Field name containing the reference summary in the JSONL examples.",
	)
	parser.add_argument(
		"--no-stemmer",
		action="store_true",
		help="Disable Porter stemming inside the ROUGE scorer (defaults to enabled).",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Optional path to save a JSON report with aggregate metrics and sample counts.",
	)
	return parser.parse_args()


def load_examples(
	path: Path,
	source_field: str,
	target_field: str,
	max_samples: int | None,
) -> List[Tuple[str, str]]:
	examples: List[Tuple[str, str]] = []
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			record = json.loads(line)
			try:
				source = str(record[source_field])
				target = str(record[target_field])
			except KeyError as exc:  # pragma: no cover - invalid data surface at runtime
				raise KeyError(f"Missing field in record: {exc} (available keys: {list(record)})") from exc
			examples.append((source, target))
			if max_samples is not None and len(examples) >= max_samples:
				break
	if not examples:
		raise ValueError(f"No examples loaded from {path}")
	return examples


def batched(items: Sequence[Tuple[str, str]], batch_size: int) -> Iterable[Sequence[Tuple[str, str]]]:
	for start in range(0, len(items), batch_size):
		yield items[start : start + batch_size]


def aggregate_scores(raw_scores: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
	aggregated: Dict[str, Dict[str, float]] = {}
	for metric, components in raw_scores.items():
		aggregated[metric] = {
			component: (fmean(values) if values else 0.0) for component, values in components.items()
		}
	return aggregated


def main() -> None:
	args = parse_args()

	pipeline, _ = create_inference_pipeline(
		checkpoint_path=args.checkpoint,
		labels_path=args.labels,
		tokenizer_dir=args.tokenizer_dir,
		model_config_path=args.model_config,
		device=args.device,
		summary_max_length=args.max_length,
	)

	examples = load_examples(args.data, args.source_field, args.target_field, args.max_samples)
	scorer = rouge_scorer.RougeScorer(list(args.metrics), use_stemmer=not args.no_stemmer)

	score_store: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

	for batch in tqdm(
		list(batched(examples, args.batch_size)),
		desc="Evaluating",
		total=(len(examples) + args.batch_size - 1) // args.batch_size,
	):
		documents = [item[0] for item in batch]
		references = [item[1] for item in batch]
		predictions = pipeline.summarize(documents, max_length=args.max_length)

		for reference, prediction in zip(references, predictions):
			scores = scorer.score(reference, prediction)
			for metric_name, score in scores.items():
				score_store[metric_name]["precision"].append(score.precision)
				score_store[metric_name]["recall"].append(score.recall)
				score_store[metric_name]["fmeasure"].append(score.fmeasure)

	aggregated = aggregate_scores(score_store)
	report = {
		"num_examples": len(examples),
		"metrics": aggregated,
		"config": {
			"data": str(args.data),
			"checkpoint": str(args.checkpoint),
			"tokenizer_dir": str(args.tokenizer_dir),
			"metrics": list(args.metrics),
			"max_length": args.max_length,
			"batch_size": args.batch_size,
			"device": args.device,
		},
	}

	print(json.dumps(report, indent=2))
	if args.output:
		args.output.parent.mkdir(parents=True, exist_ok=True)
		with args.output.open("w", encoding="utf-8") as handle:
			json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
	main()