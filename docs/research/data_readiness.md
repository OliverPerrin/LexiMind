# Current corpus readiness

**September 26, 2026: blocked for research admission.** This is a read-only audit
of the current prepared files, not a training run, model evaluation, or assessment
of model quality. No datasets were downloaded, rewritten, deduplicated, or assigned
to new splits. Historical results retain their original meaning and limitations.

The machine-readable [inventory](../../research/preparation/data_inventory.json)
pins every JSONL and label-vocabulary file by SHA-256. The
[audit](../../research/preparation/data_audit.json) records bounded line references,
record hashes, overlap counts, identity gaps, and the unresolved admission gates.
Both reports are deterministic for unchanged inputs and auditor code.

## Inventory

There are **156,796 records in 12 JSONL files**, totaling **384,924,784 bytes**.
All current records parse successfully; no unknown labels were found against the
saved label vocabularies. Structural readability does not establish source fidelity
or independence between evaluation and training examples.

| Corpus | Train | Validation | Test | Current content |
| --- | ---: | ---: | ---: | --- |
| Books / language modeling | 27,000 | 1,500 | 1,500 | Gutenberg paragraphs |
| Summarization | 61,877 | 3,436 | 3,440 | 50,000 academic + 18,753 literary pairs overall |
| Emotion | 43,410 | 5,426 | 5,427 | Comment text, 28-label vocabulary |
| Topic | 3,402 | 189 | 189 | 3,240 newsgroup + 540 Gutenberg examples overall; 7 labels |

The topic vocabulary is Fiction, Science, Technology, Philosophy, History,
Business, and Arts. Psychology is absent from this data snapshot. Topic source
labels such as `newsgroups` survive, but they do not identify individual source
records or source revisions.

Every one of the **156,796 rows lacks a retained parent `work_id` or `document_id`
and a source URL in the audited provenance fields**. There are no usable author
identities in the current rows, so no title-plus-author grouping can be established.
All **18,753 literary summary pairs** lack the identity/provenance evidence now
required by the repaired converter. This is a quarantine reason, not a finding
that each individual pair is incorrect.

## Split overlap and pairing signals

Exact comparison retains case, punctuation, and diacritics. The normalized check
only applies Unicode NFC and collapses whitespace. It does not use embeddings,
fuzzy matching, title-derived canonical identities, or model outputs.

There are **73 normalized-input groups spanning different splits**; 72 already
match without text normalization. The principal pairwise intersections are:

| Corpus | Split pair | Shared normalized-input groups |
| --- | --- | ---: |
| Emotion | Train / test | 32 |
| Emotion | Train / validation | 41 |
| Emotion | Validation / test | 10 |
| Books | Train / test | 1 |
| Topic | Train / validation | 1 |

Pairwise counts are not additive: one repeated string can occur in all three
splits. Repeated comments can have different annotations, so automatic deletion
or label merging would require a separate, explicit policy. There are also exact
input repetitions within the training partition across tasks; these are recorded
separately from cross-split contamination.

Summarization has **1,175 normalized target-text groups spanning splits**:
638 train/test, 624 train/validation, and 93 validation/test intersections. Repeated
back-cover descriptions across excerpts may describe the same parent document;
these files no longer preserve enough identity to decide reliably. **28 target
texts are attached to more than one distinct title candidate**. That is a pairing
review signal, not proof that the titles refer to different canonical works.

The title-only comparison also finds 340 shared title candidates between books
train/test and 330 between books train/validation. Those signals are consistent
with the historical paragraph-level split risk, but author/edition identity is
missing. Title equality alone cannot prove or repair work-level leakage. No title
candidate is promoted to a `work_id` by this audit.

## Historical report counts have drifted

The archived [test report](../../research/results/historical/leximind_test.json)
contains 2,727 summarization examples: 2,506 academic and 221 literary. The current
test JSONL contains **3,440**: 2,506 academic and **934 literary**, a difference of
713 literary examples. Current emotion and topic test counts equal the report's
5,427 and 189, respectively. Equal counts do not establish equal examples, frozen
preprocessing, or linkage to a particular checkpoint/run. The auditor extracts
only archived sample counts, never metric values or new quality scores.

## Eligibility and quarantine

All four current corpora remain quarantined from a new controlled study. They are
usable as historical artifacts and as inputs to source-reconstruction planning.
They are not approved training/evaluation inputs merely because their bytes are
now hashed.

| Corpus | What can be prepared before training | Remaining admission decision |
| --- | --- | --- |
| Books | Restore provider document IDs; reconcile editions; define the target reading population and a grouped split recipe | Whether this language-model corpus belongs in the next study and what work-level generalization is claimed |
| Literary summarization | Recover source descriptions and parent identities; review legacy joins; keep chapter summaries distinct from descriptive blurbs | Whether to rebuild from verified pairs, use provider-identified chapter data, or exclude this task initially |
| Academic summarization | Pin original provider records and revisions; retain paper/document IDs; document truncation | Whether academic summarization remains part of the general MTL study despite the books-first product |
| Emotion | Restore original comment IDs and source split provenance; define a duplicate/annotation policy; specify separate model-selection and calibration groups | Whether the comment task is retained for research; it is not validated evidence of book mood |
| Topic | Preserve source labels/IDs, freeze the seven-label mapping or define a separately versioned replacement | Whether mixed newsgroup/Gutenberg domains and the label taxonomy answer the intended research question |

The repaired future converters already support **provider-document grouping**:
Gutenberg retains `text_id` or a PG-19 ebook identifier, and BookSum retains its
parent `bid`. Missing Gutenberg identifiers can be replaced with an explicitly
local full-document hash solely for grouping that document's paragraphs. Those
namespaces are not cross-source canonical work authority. Provider-document IDs
can reduce paragraph/chapter leakage while edition and work reconciliation remains
unresolved. See [catalogue identity boundaries](../../data/catalog/README.md).

A future source-preparation change should write a new version, retain its raw
source/revision receipts, document inclusion/exclusion rules, freeze an immutable
partition manifest, and audit it before any model job. It must not retroactively
repair or relabel the archived research reports.

## Reuse and verification

```sh
python3 scripts/audit_research_data.py
python3 scripts/audit_research_data.py --require-ready
python3 -m pytest tests/test_research/test_data_readiness.py -q
```

The normal command completes the audit and exits successfully even when it finds
blockers. `--require-ready` writes the same reports and exits **2**: this tool has no
mechanism to authorize training. Even mechanically clean fixtures are marked
`requires_protocol_review`, never approved. User authorization and the reviewed
research/source protocol remain separate requirements.

The CLI streams each JSONL file and stores only hashes and references in a temporary
SQLite index. It never keeps the entire corpus text in memory or emits raw text,
summaries, titles, or author names into reports. Samples are bounded by
`--sample-limit` (1–10; default 4). Reports cannot overwrite the audited input
folder, one another, or the historical reference. Changes to scanned files during
the audit abort the snapshot. Fixtures exercise these boundaries, including a
clean corpus that still cannot approve its own research admission.

Exact hashes cannot find overlapping excerpts, paraphrases, translated editions,
or other near-duplicates. Source URLs and declared IDs are evidence to review, not
independent verification. These limitations remain explicit even when all
mechanical checks pass.
