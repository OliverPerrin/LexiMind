# Archived paper drafts (superseded 2026-08-04)

These drafts are retained for history. **Do not cite, link, or distribute them.**
They were withdrawn because several reported numbers were written into tables
before the corresponding experiments were run, and no run artifact exists for them.

| File | Status |
| ---- | ------ |
| `research_paper.tex` / `.pdf` | EMNLP-format draft, "Train Short, Infer Long". Withdrawn. |
| `paper.tex` / `.pdf` | Longer academic-format draft. Withdrawn. |
| `paper_old.tex` | Earlier revision of the above. Withdrawn. |

## Unverified numbers in these drafts

Audited 2026-08-04 against every result file, checkpoint, training log and MLflow
run in the repository. The following table entries have **no supporting run**:

| Draft location | Claim | Why it is unsupported |
| --- | --- | --- |
| `research_paper.tex` Tab. `main_results`, "Single" column | R-1 0.298, Topic 82.0%, Emotion 0.218 | No single-task LexiMind checkpoint or result file exists. `scripts/run_w2_campaign.sh` writes single-task runs to `outputs/w2_baselines/`, which was never created. The `configs/training/single_*.yaml` files were added in commit `c420d7c` — the same commit as the draft — and never executed. |
| `research_paper.tex` Tab. `main_results`, "MTL Base" column | R-1 0.306, Topic 85.2%, Emotion 0.199 | No config selects mean pooling + round-robin; no checkpoint; no result file. |
| `research_paper.tex` Tab. `transfer_ablation` | Random init: R-L 0.098, Topic 45.2%, Emotion 0.082 | No random-init run. `use_pretrained: false` appears only in `configs/model/small.yaml`, which was never trained. |
| `research_paper.tex` Tab. `baselines` | FLAN-T5 zero-shot: R-L 0.121, Topic 58.2%, Emotion 0.089 | No zero-shot evaluation code exists in the repository. |
| `research_paper.tex` §Abstract, §Results (RQ2) | Chunked long-document aggregation comparison | No chunking or aggregation module exists. |
| `research_paper.tex` §Abstract, §Results (RQ4) | Cost/quality comparison vs. zero-shot Claude / GPT-4 | Not implemented, not run. |
| `research_paper.tex` App. `multiseed` | 5-seed mean ± std | Marked `% TODO` in the source. Seed 17 completed; seed 42 terminated at 19% of epoch 1; seeds 123/456/789 never started. |
| `paper.tex` §Abstract | "improving sample-averaged F1 from 0.199 to 0.352 (+77%)" | The 0.199 baseline is the unmeasured MTL-Base figure above. |

Consequently the drafts' central claim — that attention pooling plus temperature
sampling *eliminates negative transfer* relative to single-task training — was
never tested. It may well be true; it was not measured.

## Numbers in these drafts that ARE backed by run artifacts

See [`../RESULTS.md`](../RESULTS.md) for the verified record. In short: the joint-MTL
test results and the BERT-base baseline comparison are real and reproducible from
`outputs/evaluation_report_test.json` and `outputs/bert_baseline/combined_results.json`.
