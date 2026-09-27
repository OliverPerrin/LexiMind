# Methods for accurate book fields

Use [dataset_decisions.md](dataset_decisions.md) for the current data choice and
[study_decisions.md](study_decisions.md) for the accepted experiment. Topics,
genres and reading atmosphere are different targets; each may need multiple labels.
Prepared news/comment benchmarks are not recommended supervision for product book fields.

## Methods evidence and limits

These four approaches fit the custom encoder/heads. They are recommendations to
validate on books, not measured LexiMind gains or a new model-recipe plan.

1. **Partial multi-label supervision.** Track each label as positive, reviewed
   negative, or unknown; apply loss only to observed labels with explicit
   normalization. [Durand et al., §3.1](https://arxiv.org/pdf/1902.09720) studies this
   setup in vision. Positive-only masking can learn “everything is positive”;
   [Cole et al., §5](https://arxiv.org/pdf/2106.09708) makes that failure explicit.
   Start with reviewed negatives/complete examples alongside source-derived silver
   positives. Missing catalogue subjects are not negatives. Optional parent-label
   closure requires a reviewed ontology: [C-HMCNN](https://proceedings.neurips.cc/paper/2020/file/6dd4e10e3296fa63738371ec0d5df818-Paper.pdf)
   constrains a hierarchy but does not establish a correct book taxonomy.
2. **Fields with inspectable evidence.** Copy explicit catalogue fields
   deterministically. For inferred labels, retain source record/span offsets and
   distinguish the label from its supporting passage; an existing token head can
   support a later span-selection task. [ERASER, §§4–5](https://aclanthology.org/2020.acl-main.408.pdf)
   separates plausible support from model faithfulness. A copied phrase or generated
   explanation does not prove a whole-book genre/mood judgment.
3. **Selective prediction.** Return unknown when evidence is absent; separately
   abstain on uncertain predictions. Calibrate field thresholds on independent
   book-domain examples and measure precision/coverage and risk–coverage by source.
   [Selective QA under domain shift, §§3–5](https://aclanthology.org/2020.acl-main.503.pdf)
   shows why out-of-domain softmax confidence can mislead. Begin with a calibrated
   threshold baseline, not a new rejection architecture or forced tone assignment.
4. **Optional task-domain adaptation.** After the label/data baseline is sound,
   compare teacher-free adaptation on permitted training-book text. [DAPT/TAPT,
   §§3–5](https://aclanthology.org/2020.acl-main.740.pdf) studies continued RoBERTa
   pretraining; custom T5 denoising is an unmeasured adaptation, not a demonstrated
   book improvement. Keep a common initialization across recipe arms and count its
   preparation cost; do not silently consume held-out text or supervision.

## Dataset suitability and terms

The earlier [book evidence register](../../research/preparation/book_literature.json)
retains nine recommendation/narrative/evaluation source groups and provider terms.
Its resources supply metadata, interactions, plot summaries or passage emotions;
none of those targets automatically becomes sustained whole-work mood gold.
Keep source-derived silver labels separate from independent evaluation judgments.

For product relevance, B1 holds out query families and seed groups over a fixed
catalogue; candidate books may recur across partitions. Rank every eligible work.
Primary nDCG requires complete reviewed relevance; pooled scores are secondary.
See [book admission](book_admission_contract.md) and the [mood rubric](../mood_annotation_guide.md)
for evidence scope. Better classification or a convincing rationale alone does not
establish better recommendations or an accurate reader-atmosphere label.
