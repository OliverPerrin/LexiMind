"""Generate historical Markdown or LaTeX tables from hash-checked archived JSON.

This formats existing reports; it never evaluates a model. These are historical
observations with incomplete provenance, not a new controlled comparison.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

try:  # Support both `python scripts/build_tables.py` and module imports in tests.
    from scripts.audit_research_artifacts import (
        DEFAULT_MANIFEST,
        ROOT,
        audit_archive,
        load_json,
        safe_path,
    )
except ModuleNotFoundError:
    from audit_research_artifacts import (
        DEFAULT_MANIFEST,
        ROOT,
        audit_archive,
        load_json,
        safe_path,
    )

NOTICE = (
    "September 22, 2026 archive. Historical reports only; incomplete checkpoint/data provenance. "
    "No rerun or controlled MTL comparison. LexiMind's fixed threshold is unpinned "
    "(code default at audit: 0.5); BERT reports 0.3. Calibration protocols differ. "
    "No significance or across-seed claims are supported."
)
HEADERS = ("Model", "Task / setting", "Metric", "Value", "Source JSON path")


def metric_at(report: dict, keys: tuple[str, ...]) -> float:
    value = report
    for key in keys:
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Non-numeric metric: {'/'.join(keys)}")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Metric out of [0, 1] range: {'/'.join(keys)}")
    return float(value)


def historical_rows(root: Path, manifest: dict) -> list[tuple[str, ...]]:
    errors = audit_archive(root, manifest)
    if errors:
        raise ValueError("Archive verification failed: " + "; ".join(errors))
    paths = {item["id"]: item["path"] for item in manifest["artifacts"]}
    joint = load_json(safe_path(root, paths["leximind_test"]))
    bert = load_json(safe_path(root, paths["bert_combined_test"]))
    if joint.get("_meta", {}).get("split") != "test":
        raise ValueError("LexiMind report must explicitly identify the test split")
    rows = []

    def add(model: str, setting: str, label: str, report: dict, artifact: str, *keys: str):
        rows.append(
            (
                model,
                setting,
                label,
                f"{metric_at(report, keys):.3f}",
                artifact + ":/" + "/".join(keys),
            )
        )

    for key, label in (
        ("rouge1", "ROUGE-1"),
        ("rouge2", "ROUGE-2"),
        ("rougeL", "ROUGE-L"),
        ("bleu4", "BLEU-4"),
        ("bertscore_f1", "BERTScore F1"),
    ):
        add(
            "LexiMind",
            "Summarization, overall",
            label,
            joint,
            "leximind_test",
            "summarization",
            key,
        )
    for domain in ("academic", "literary"):
        add(
            "LexiMind",
            f"Summarization, {domain}",
            "ROUGE-L",
            joint,
            "leximind_test",
            "summarization",
            "per_domain",
            domain,
            "rougeL",
        )
    for prefix, setting in (
        ("", "Emotion, fixed threshold unpinned"),
        ("frozen_tuned_", "Emotion, frozen-tuned"),
    ):
        for key, label in (
            ("sample_avg_f1", "Sample F1"),
            ("macro_f1", "Macro F1"),
            ("micro_f1", "Micro F1"),
        ):
            add("LexiMind", setting, label, joint, "leximind_test", "emotion", prefix + key)
    for key, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro F1")):
        add("LexiMind", "Topic", label, joint, "leximind_test", "topic", key)
    for mode in ("single-topic", "single-emotion", "multitask"):
        if bert[mode].get("split") != "test":
            raise ValueError(f"BERT {mode} report must identify the test split")
        if mode != "single-topic":
            threshold = metric_at(bert, (mode, "evaluation", "emotion", "default_threshold"))
            for prefix, setting in (
                ("", f"Emotion, threshold {threshold:g}"),
                ("frozen_tuned_", "Emotion, frozen-tuned"),
            ):
                for key, label in (
                    ("sample_avg_f1", "Sample F1"),
                    ("macro_f1", "Macro F1"),
                    ("micro_f1", "Micro F1"),
                ):
                    add(
                        f"BERT {mode}",
                        setting,
                        label,
                        bert,
                        "bert_combined_test",
                        mode,
                        "evaluation",
                        "emotion",
                        prefix + key,
                    )
        if mode != "single-emotion":
            for key, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro F1")):
                add(
                    f"BERT {mode}",
                    "Topic",
                    label,
                    bert,
                    "bert_combined_test",
                    mode,
                    "evaluation",
                    "topic",
                    key,
                )
    return rows


def latex_escape(value: str) -> str:
    escapes = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(escapes.get(char, char) for char in value)


def render(rows: list[tuple[str, ...]], output_format: str) -> str:
    if output_format == "latex":
        lines = [
            "% Generated by scripts/build_tables.py; source JSON paths accompany each row.",
            r"\noindent\textit{" + latex_escape(NOTICE) + r"}\par",
            r"\begin{longtable}{p{2.4cm}p{4.5cm}p{2.3cm}r}",
            " & ".join(HEADERS[:4]) + r" \\",
            r"\hline",
        ]
        for row in rows:
            lines.append("% Source: " + row[4])
            lines.append(" & ".join(latex_escape(cell) for cell in row[:4]) + r" \\")
        lines.append(r"\end{longtable}")
    elif output_format == "markdown":
        lines = [
            "<!-- Generated by scripts/build_tables.py; do not edit numbers by hand. -->",
            "# Historical report tables",
            "",
            NOTICE,
            "",
            "| " + " | ".join(HEADERS) + " |",
            "| " + " | ".join(["---"] * 5) + " |",
        ]
        lines.extend(
            "| " + " | ".join(cell.replace("|", r"\|") for cell in row) + " |" for row in rows
        )
        lines.extend(
            [
                "",
                "Source IDs resolve through [manifest.json](manifest.json). "
                "See [RESULTS.md](../../docs/RESULTS.md) for sample counts, intervals and limitations.",
            ]
        )
    else:
        raise ValueError(f"Unknown format: {output_format}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--format", choices=("markdown", "latex"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        manifest_path = safe_path(args.root, str(args.manifest))
        manifest = load_json(manifest_path)
        rendered = render(historical_rows(args.root, manifest), args.format)
        if args.output:
            output = args.output.resolve()
            protected = {safe_path(args.root, item["path"]) for item in manifest["artifacts"]}
            if output in protected | {manifest_path}:
                raise ValueError("Refusing to overwrite source evidence with a generated table")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(1, f"Cannot generate tables: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
