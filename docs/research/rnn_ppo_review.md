# RNNs and PufferLib PPO: decision review

Reviewed **3 October 2026** against the local implementation and primary sources.
This is a methods and integration review. No PufferLib installation, RNN training,
GPU reservation or recommendation experiment was performed.

The historical review below retains that 3 October scope. A separately verified
8 October CUDA build and engineering smoke are recorded in the dated addendum.

**Decision:** strengthen the native supervised baseline first. A small recurrent
aggregator is a useful later hypothesis for ordered passages or reader sessions.
PufferLib is worth revisiting for a fast, interactive recommendation simulator,
but its current trainer is not a local M5 replacement for LexiMind's PyTorch
training loop. These recommendations are engineering inferences, not measured
LexiMind improvements.

## What LexiMind currently uses

The native [encoder](../../src/models/encoder.py) and
[decoder](../../src/models/decoder.py) are Transformers with T5-compatible
attention, feed-forward blocks and normalization. The
[multitask model](../../src/models/multitask.py) routes shared encoder outputs to
task heads or the decoder. Inspection found no GRU/LSTM/RNN implementation or
PufferLib dependency in `src/models`, `src/training` or `pyproject.toml`.
Autoregressive token generation is not itself an RNN architecture.

The [existing book experiment](../../research/results/book_denoising_20261002.json)
uses 179 training and 36 development examples from disjoint sets of works.
Its [task builder](../../src/training/denoising.py) reveals both sides of a masked
word; the answer is one to three canonical content tokens followed by EOS.
After the common warm start, continued CE reduced work-balanced content NLL more
than sparse-reward Dr. GRPO in both seeds: 6.293 versus 6.371, and 6.158 versus
6.250. Development exact recovery remained 0% and 3.33% respectively across the
two post-warm branches. These are means over works, not raw example percentages.
RL produced only 3 and 5 mixed-reward groups out of 64, hence few updates.
The branches matched prompts, not compute; their comparison does not establish
that RL generally loses to supervised learning.

There is no missing stream of past observations in this short infilling task:
the visible context is already provided. Adding hidden-state memory alone will
not supply missing lexical knowledge, trustworthy labels or successful samples.
The current experiment also supplies no reader-session reward. Keep its outcomes
separate from [book-field evaluation](book_discovery_review.md) and recommendation
quality.

## Four different choices

| Choice | What changes | Plausible LexiMind role |
| --- | --- | --- |
| Conventional GRU/LSTM | A gated hidden state summarizes an ordered sequence; ordinary versions depend on the previous state inside their gates. | A small supervised head over frozen passage embeddings, or a session-history encoder. It need not use RL. |
| Modern recurrent sequence models | minGRU simplifies recurrence to enable parallel scan; RWKV is another recurrent language-model family. | A separately controlled architecture experiment. Neither accepts T5 weights merely because hidden dimensions match. |
| Recurrent RL policy | Memory carries information between environment decisions, with resets at episode boundaries and sequence-aware training. | A reader-session policy when useful preferences/history are absent from the current observation. |
| PPO/PuffeRL | A policy-learning objective and training system, including sampled actions, behavior log probabilities and advantage/value estimation. | A later optimization experiment; it can be combined with memory but does not imply an RNN. |

The recurrence distinctions follow [Feng et al., v3, §§2–3 and appendices][mingru]
and the [RWKV v2 abstract][rwkv]. A fixed-size state is an information bottleneck,
not unlimited recall or a guarantee of full-book understanding. These sources
motivate testing efficiency; they do not validate LexiMind's labels or ranking.

[PPO, v2][ppo] alternates environment sampling with surrogate-objective updates.
PufferAI's [algorithm article][puffer-ppo] describes additional optimizer,
trajectory-sampling and GAE/V-trace changes. Current
[loss/advantage code][algo] contains clipped policy and value losses, entropy and
the combined advantage calculation. Therefore a PufferLib comparison changes
more than the name of LexiMind's existing group-relative objective. Its
hyperparameters and ablations must be explicit.

## Version and hardware distinctions

“PufferLib PPO” is ambiguous because the published Python package and current
repository are different generations. The following were observed on the review
date; branch names and unversioned documentation may change.

| Inspected line | Verified evidence | Consequence |
| --- | --- | --- |
| Current default branch `5.0` | GitHub resolved to `6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2`, committed 13 September 2026. [Build][build], [trainer][trainer] and [algorithm][algo] are C/CUDA. The latest GitHub release is named [5.0 Experiments][release], an experiments/checkpoints release. | Pin the commit for any trial; do not treat the release title as a PyPI package version. |
| Published PyPI package | [PyPI][pypi] reports **3.0.0**, uploaded 23 June 2025, as latest. Its source archive SHA-256 is `7df3a3e3f5f894d78d2a1f5374097890aec01473183e748abefe4f3faa10eaa9`. | `pip install pufferlib` does not select the current 5.0 C/CUDA trainer. The archive was not downloaded or installed in this review. |
| Historical Python branch `3.0` | Inspected commit `3b5c6046bb8b46685d62d151720025507e3418c2`, committed 13 May 2026. [Models][old-models] include `LSTMWrapper`; [emulation][old-env] wraps Gymnasium; [training][old-trainer] includes CPU advantage fallback. | This later branch commit is **not claimed to equal** the June 2025 PyPI archive. Old Python integration examples must name their exact pin. |

Current [official documentation][docs] describes CPU evaluation without CPU
training. The [build script][build] has a macOS CPU path, while trainer builds use
`nvcc` and CUDA libraries. The current default policy is MinGRU with highway
connections; [PufferAI's architecture article][puffer-mingru] attributes its
performance to architecture and custom kernels together. The author's game
benchmarks do not establish equivalent speedups for text models, MPS or Python
implementations.

| Execution target | Assessment |
| --- | --- |
| Native LexiMind on the M5 | The saved infilling receipt records PyTorch 2.14.0, float32 MPS, 16 GiB unified memory and CPU fallback disabled. That verifies the existing workload only. |
| A new PyTorch GRU/LSTM head on CPU/MPS | Reasonable local candidate. [PyTorch 2.14 MPS documentation][mps] establishes the backend, not support/performance for every proposed operation. Check forward/backward, padding, reset behavior, device residency and timing on the actual implementation. |
| PufferLib 5.0 training on M5 CPU/MPS | No supported path found in the inspected source/docs. MPS does not execute CUDA C kernels, and there is no PyTorch `device=mps` switch in this trainer. |
| PufferLib 5.0 CPU evaluation on macOS | Documented/build-supported direction for compatible environments, but unbuilt and unverified here. Environment dependencies may still matter. CPU evaluation support does not imply CPU training. |
| Historical 3.0 Python CPU/MPS training | CPU fallback exists in inspected code and setup supports a C++ extension without CUDA. That is not an end-to-end macOS/MPS compatibility test. Do not promise MPS support; isolate and smoke-test an exact revision if there is a specific reason to use it. |
| PufferLib 5.0 on NVIDIA CUDA | The intended training path. A separate compatible GPU environment and explicit experiment budget are prerequisites; none was used in this review. |

## Integration work a PufferLib experiment would require

The [5.0 environment interface][env] is a shared-buffer C API. `Agent` exposes
observations, actions, rewards, terminals and an optional action mask;
`puf_init`, `puf_reset`, `puf_step`, `puf_close` and logging/render hooks define
the lifecycle. The [template][template] declares fixed observation/action sizes,
writes buffers, performs resets and emits the terminal signal. CPU-hosted
environments can feed the CUDA trainer; a CUDA-native environment is a different
implementation choice. This is not the historical Gymnasium wrapper API.

For a recommendation prototype, materialize a versioned numeric candidate set
and book features, plus a simulator that updates user state and returns feedback.
Keep source text acquisition and neural book encoding outside the simulator's
inner loop. A masked fixed candidate pool is practical for a pilot, but its
ranking claim is conditional on that pool and cannot replace the full eligible
catalogue evaluation required by [book admission](book_admission_contract.md).

Before learning, verify action-mask consistency between sampling and scoring,
reward/terminal buffer clearing, deterministic reset/replay, and hidden-state
isolation between readers. Declare whether the episode limit is a true terminal
objective or a time-limit truncation. The inspected C `Agent` exposes a terminal
buffer, not Gymnasium's separate truncation return; bootstrap semantics need an
explicit design and test. A rollout chunk boundary must not silently become an
episode boundary.

Porting native T5 infilling would additionally need text/token observations,
variable-length masking, EOS handling, a value function and a compatible model
implementation or bridge. The current PufferNet is not a pretrained text model.
Keeping T5 in a separate Python process would need measured transfer/inference
costs and correct behavior-policy accounting. It provides no inherited
million-steps-per-second guarantee.

## Concrete next experiments

1. **Execute the fixed supervised comparison in the existing runtime.** The
   [new protocol](../../configs/research/book_supervision.json) retains the same
   179/36 rows and seeds 17/29, comparing `instruction_word` with `t5_span` at
   fixed 128/512-update development endpoints plus a final training-fit check.
   The second arm changes both input and output format; it does not isolate a
   sentinel-only effect. Report strict exact recovery, within-arm content-NLL
   changes, control/EOS NLL and invalid outputs; equal updates do not equalize
   token or compute budgets. At final weights, the fixed coverage probe samples
   46 training prompts times four responses per arm and seed, with no updates,
   retries, filtering or resampling. Its preregistered coverage gate concerns
   training-reward feasibility only and does not automatically launch RL or
   choose a winner. Keep the previous report immutable and test/quarantine
   narratives excluded. These are protocol commitments, not outcomes of a run
   reported in this review; no extra adaptive overfit run is proposed here.
2. **Test a supervised recurrent passage aggregator when matching book labels
   are admitted.** Freeze/cache the same native encoder representations for all
   arms. Compare mean pooling plus a small head, a parameter-accounted GRU head,
   and optionally minGRU over the same ordered passages. Match available text,
   targets, work splits, optimizer budget and selection allowance. Include a
   shuffled-passage control to test whether order helps, and evaluate partial
   label masks, per-field precision/coverage, latency and peak memory. A
   forward GRU is naturally streamable; a bidirectional GRU requires the whole
   observed sequence and must not be used as if it were an online policy.
   Unknown labels remain unknown. No independent book-field gain can be measured
   before appropriate evaluation judgments exist.
3. **For the website, test supervised session ranking before sequential RL.**
   [GRU4Rec, v4][gru4rec] is primary precedent for learning session-based ranking
   from item sequences without PPO. Compare a GRU with content similarity,
   popularity, last-item and pooled-history controls using the same candidates.
   Define impression, click, save and explicit preference events separately;
   missing interaction is not a reviewed dislike. Use chronological splits,
   session/user separation appropriate to the claim, and metrics such as
   Recall/nDCG alongside coverage and repetition. This requires usable session
   data; the current narrative reconstruction dataset cannot supply it.
4. **Admit a bounded PufferLib simulator trial only for a sequential question.**
   Example: can remembering earlier preference feedback improve recommendations
   over a short session when the current observation omits that history? First
   validate a deterministic small simulator with scripted and random policies.
   Then compare a memoryless policy, explicit-history control and recurrent
   policy under the same reward, candidates, step budget and seeds. If evaluating
   PufferNet versus an MLP requires altering the pinned 5.0 architecture, record
   that change as part of the experiment. Freeze simulator dynamics and evaluation
   seeds before model selection; report total steps, wall time, tuning cost,
   reward, diversity and repeat rate. A one-decision ranking task is a contextual
   bandit, so it does not test long-term credit assignment. Simulator gains alone
   do not establish reader satisfaction or causal online benefit.

If the question becomes **PPO versus GRPO for infilling**, first compare the
objectives around the same native T5, samples/reward contract and declared compute
accounting, adding an explicitly budgeted value head and checking its returns.
Switching architecture, optimizer, device and environment simultaneously would
not isolate the algorithm. A critic can supply a baseline for advantages; it
cannot manufacture correct sampled completions from an uninformative reward.

## Evidence boundaries and source pins

Do not claim an RNN/PPO improvement, successful PufferLib installation, current
MPS support for PufferLib, whole-book comprehension, matched-compute superiority
or recommendation quality from this review. The accepted
[MTL/merging study](study_decisions.md) remains separate. Changing to a recurrent
backbone would require fresh adapter/head/merge compatibility controls, not a
silent substitution into the T5 experiment.

Primary-source reading scope was the current docs/FAQ; listed C/CUDA interface,
build, architecture/loss sections; historical Python wrapper/trainer/setup
sections; PufferAI's MinGRU and PPO articles; minGRU method/implementation sections;
GRU4Rec's method/evaluation; and PPO/RWKV version metadata and abstracts. This is
not a replication or an exhaustive recurrent-language-model survey.

The website documentation and both cited PufferAI articles were also checked in
the [website repository at `99b93b0926fe11540267edfe1f6c359225cd8e41`][website],
committed 14 September 2026. Source SHA-256 receipts for key inspected files:

| Pinned source | SHA-256 |
| --- | --- |
| Website `docs/docs.html` | `8983e6f77fbdf42d91ab27c2ca0bbef225a05ae53c40a4608366b5b706885858` |
| Website `docs/blog/mingru/index.html` | `ab38c03dddb82aca4f22e2d430739be7a4ba5a356f64ee4da3cf1a37bd102081` |
| Website `docs/blog/ppo/index.html` | `495dce86fa13c64e82b46a414c3c69973a680d7421566755a12a640f6afe01e6` |
| PufferLib 5.0 `build.sh` | `3682a835ef989391b11ab53cc02bede0551b46d9302e32e2f98758d598cb879a` |
| PufferLib 5.0 `src/pufferenv.h` | `3324a670791e2b3e46cbe7d10d9233d52fbc258e0d7e37d5b0c470e580de880c` |
| PufferLib 5.0 `src/algo.cu` | `beb55f848076b6e089b015cf90970bb0d7a38290aa6e1ff74c961d9d4e1471f1` |
| PufferLib historical 3.0 `pufferlib/models.py` | `eb8e0edb5f7f270a5cfe847288acac300223130503a94a4fb4ec071c60e131b9` |
| PufferLib historical 3.0 `pufferlib/pufferl.py` | `ebe89e96a2515e7b0370d7b4833ae64bb46c1a68c20d13fd014a10abc4e33323` |

## Addendum, 8 October 2026: verified native CUDA engineering smoke

A private CUDA 13.0.2 toolkit (nvcc 13.0.88) built the pinned PufferLib 5.0
source `6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2` for
`breakout --cu sm_89` on the user-owned RTX4070/WSL host. The
[compact observation](../../research/results/puffer_cuda_smoke_20261008.json)
binds the build, binary, configuration, logs, checkpoint and transfer hashes.
The build-stage receipt's “not GPU executed” status precedes the separately
authorized smoke; it is preserved as historical evidence.

The single fixed seed-17 smoke exited 0 after **4,194,304 agent steps / 32 rollout
epochs**, saving four checkpoints at intervals of eight epochs. All checkpoint
FP32 master weights and the 24 displayed loss values were finite. Those values
include repeated dashboard displays and do not provide continuous loss
monitoring. Evaluation
used a stopping threshold of 128 episodes; upstream vectorization actually
completed **255 episodes**, with recorded mean score **0.6039215922355652** and
mean episode length 514.7254638671875. End-to-end wall time was
**1.194140217 seconds**, while upstream reported internal uptime
**0.22890925407409668 seconds**. These timings measure different scopes.

One-second GPU polling yielded only two samples, with sampled maximum memory
1829 MiB; upstream separately reported 1.59912109375 GiB. Neither value proves
true continuous peak memory. The saved post-run query listed no compute
processes, but its exit status and stderr were not recorded; GPU release is
therefore not proven. No tuning or
retry was performed. This verifies this native build and upstream Breakout
execution path. It establishes no LexiMind PPO integration, PPO quality gain,
recommendation relevance, RNN advantage or semantic-task performance. Binary
provenance at execution remains receipt-based because no execution-time binary
hash was saved. A later comparison of the current binary with the build hash
cannot retrospectively prove the exact bytes executed. The current saved binary
and raylib archive were subsequently checked independently against the build
receipt and match its byte counts and SHA-256 hashes.

Key SHA-256 receipts:

| Evidence | SHA-256 |
| --- | --- |
| Build receipt | `ac00093bafda6d6f5f90f5b1e27a7a1b66978d3fdc2039da56cbd0704deaf1d6` |
| Smoke receipt | `4eb8497d62147e47add497b002afe13874b3dcf3815acb88886a2e1bf27330c9` |
| Native binary | `6773b2ed4f41edd8e3cbf1770b094633fe007dcd3dda81e0c782dc9e11a903ae` |
| Transferred smoke evidence | `cb870290726c25865467af8a708a7b57ac2bbec81184cfcde67746d1806936f3` |

The prior review’s LexiMind task/environment and supervised-control gates remain
open. A future simulator study needs its own fixed observations, rewards,
baselines, seeds and evaluation contract; this smoke supplies no such evidence.

[mingru]: https://arxiv.org/html/2410.01201v3
[rwkv]: https://arxiv.org/abs/2305.13048v2
[ppo]: https://arxiv.org/abs/1707.06347v2
[gru4rec]: https://arxiv.org/html/1511.06939v4
[docs]: https://puffer.ai/docs.html
[puffer-mingru]: https://puffer.ai/blog/mingru/
[puffer-ppo]: https://puffer.ai/blog/ppo/
[pypi]: https://pypi.org/project/pufferlib/3.0.0/
[release]: https://github.com/PufferAI/PufferLib/releases/tag/5.0-experiments
[build]: https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/build.sh
[trainer]: https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/src/pufferl.cu
[algo]: https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/src/algo.cu
[env]: https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/src/pufferenv.h
[template]: https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/ocean/template/template.h
[old-models]: https://github.com/PufferAI/PufferLib/blob/3b5c6046bb8b46685d62d151720025507e3418c2/pufferlib/models.py
[old-env]: https://github.com/PufferAI/PufferLib/blob/3b5c6046bb8b46685d62d151720025507e3418c2/pufferlib/emulation.py
[old-trainer]: https://github.com/PufferAI/PufferLib/blob/3b5c6046bb8b46685d62d151720025507e3418c2/pufferlib/pufferl.py
[mps]: https://docs.pytorch.org/docs/2.14/mps.html
[website]: https://github.com/PufferAI/puffer.ai/tree/99b93b0926fe11540267edfe1f6c359225cd8e41
