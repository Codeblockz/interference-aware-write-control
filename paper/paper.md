---
title: "Interference-Aware Write Control for Bounded-Memory Recurrent Models"
subtitle: "EN.705.742 Advanced Applied Machine Learning, Research Paper, Midpoint Draft"
author: "Ryan Ellis"
date: "2026-10-25"
abstract: |
  Recurrent sequence models keep a fixed-size state, so when many key-value associations
  must be remembered, new writes overwrite old ones. Recent work adds bounded memories
  with allocation rules driven by input novelty, prediction residuals, or position. This
  paper asks whether a write controller that also estimates *interference*, the conflict
  between an incoming association and the associations already stored, allocates a fixed
  memory budget better. We build a gated-linear-attention backbone with a 32-slot
  external memory, a small learned controller that produces per-slot write gates from
  novelty and interference features, and four allocation baselines that share the same
  backbone and budget. On an extended multi-query associative recall task with 64
  associations, key updates, and distractors, we test (H1) whether the controller beats
  the strongest baseline by at least five percentage points of exact recall and (H2)
  whether removing the interference features removes that gain. This draft reports the
  method, a systematic survey of allocation mechanisms, the experimental protocol, and a
  single-seed pilot run that establishes the pipeline and its compute cost. Full
  multi-seed results with confidence intervals follow in the final paper.
---

# 1. Introduction and Goals

## 1.1 The problem

A Transformer remembers everything it has seen because its key-value cache grows with the
sequence. Recurrent alternatives, including linear attention [@katharopoulos2020linear],
gated linear attention [@yang2023gla], DeltaNet [@yang2024deltanet], and selective state
space models [@gu2023mamba], replace the cache with a state of fixed size. That makes
inference cost constant per token, but it also makes memory a scarce resource. When a
sequence contains more associations than the state can hold, a new write lands on
directions already in use and the old association degrades. Zoology [@arora2023zoology]
made this concrete with the multi-query associative recall (MQAR) task, where recall
accuracy of fixed-state models falls off as the number of key-value pairs grows past the
state's capacity. A recent mechanistic study [@lee2026howlinear] shows that the degradation
tracks *memory load*, the number of overlapping associations stored, far more than
elapsed context.

Three 2026 papers attack this with explicit bounded memories beside the recurrent state
and a rule for what to write into them. HOLA [@cui2026hola] keeps an exact key-value cache
and admits a token when the recurrent state's prediction residual for it is large. Raven
[@afzal2026raven] keeps a fixed set of slots and routes each token to a subset by a
learned, input-dependent router. Proteus [@bayat2026proteus] activates memory capacity on
a position schedule so early tokens cannot pollute the whole memory. Each rule looks at
the *input* (its novelty, its residual, its routing logits) or at *position*. None looks
directly at the *conflict* between the incoming association and what is already stored.

## 1.2 Goal and hypotheses

The goal of this project is to test whether an explicit interference estimate improves
allocation at a fixed memory budget. The candidate solution is a small learned write
controller that receives, for every memory slot, the similarity of the incoming key to the
slot's key and the disagreement between the incoming value and the slot's value, alongside
input novelty and slot age. It outputs differentiable per-slot write gates. Backbone,
memory, and controller train jointly on query-answer cross-entropy.

**H1.** At matched total memory and training compute, the controller improves exact recall
by at least five percentage points over the strongest validation-selected baseline on
sequences with 64 to 128 distinct associations and a 32-slot budget.

**H2.** Removing the interference features while holding the controller's size fixed
reduces that gain, showing that the interference signal carries information the input
alone does not.

## 1.3 Approach and what this draft contains

The applied ML problem is supervised sequence-to-token prediction on a synthetic recall
task. The method of analysis is a controlled architecture comparison: five writers
(the controller, its ablation, and three baselines modeled on the cited papers) on one
backbone, one memory, and one data distribution, with seeds, bootstrap confidence intervals,
and explicit state-byte accounting. This midpoint draft contains the complete method
(Section 2), the related-work survey (Section 3), the experimental protocol and a pilot
run that validates the pipeline (Section 4), and what we expect to learn (Section 5).
Multi-seed results arrive in the final paper.

# 2. Hypotheses and Method

## 2.1 Task: multi-query associative recall with updates and distractors

Each sequence has three regions (Figure 1). The *context block* holds $N$ queried pairs
and $D$ distractor pairs, each a key token followed by a value token, in random order. The
*update block* repeats $U$ of the queried keys with a new value; the correct answer for an
updated key is its latest value. The *query region* presents each queried key once more;
the label at that position is the final value. Query positions follow Zoology's power-law
gap distribution, shifted by a minimum delay $\delta$ so delay is a separate knob. The
vocabulary is split into key tokens and value tokens. Loss and metrics use only the $N$
answer positions per sequence. The generator is adapted from Zoology's `multiquery_ar`
and `forgetting_mqar` (Apache-2.0) with the distractor and delay extensions added.

Pilot setting: vocabulary 2048, sequence length 128, $N = 16$, $U = 4$, $D = 4$,
$\delta = 0$, so 24 associations compete for $S = 16$ slots. This is smaller than the
proposal's 64-association, 32-slot target; Section 4.2 reports what happened at the target
and why the pilot moved. Splits: 100,000 train, 5,000 validation, 10,000 test, generated
from distinct seeds with exact-duplicate sequences removed from validation and test.

## 2.2 Architecture

![Figure 1. System block diagram. Tokens pass through a gated-linear-attention backbone;
at each step the backbone output is projected to a query, a key, and a value. The memory
is read with the query, then the writer (learned controller or a fixed baseline rule)
produces per-slot gates from the incoming association and the memory's state, and the
memory is updated. The answer head sees the backbone output and the readout.](figures/block-diagram-proposal.png)

**Backbone.** $L$ pre-norm blocks of gated linear attention (GLA) [@yang2023gla] with an
MLP, using the `flash-linear-attention` implementation. GLA keeps, per head and layer, a
$d_k \times d_v$ recurrent state with a learned data-dependent decay. Pilot: $L = 4$,
$d_{\text{model}} = 256$, 4 heads; the full pilot model has 7.6M parameters including the 8192-token embedding and the output head. The final paper
scales to the proposal's 20M.

**Projections.** From the backbone output $h_t \in \mathbb{R}^{d_{\text{model}}}$:
$q_t = W_q h_t$, $v_t = W_v h_t$, and $k_t = W_k h_{t-1}$, all in $\mathbb{R}^{d}$ with
$d = 128$. The key is projected from the *preceding* position because an association is a
(key, value) token pair: the value token's own representation carries the value, and the
token before it carries the key. Without this one-step shift the backbone must first learn
to copy each key one position forward before the memory can store anything retrievable;
in a pilot diagnostic that cost a four-fold difference in recall at equal training steps
(Section 4.2). All writers share this projection, so it does not favor the controller.

**Slot memory.** $S$ slots, each a key $k_j$ and value $v_j$ in $\mathbb{R}^{d}$, plus
scalar age $a_j$, usage $c_j$, and occupancy $f_j$. Read is attention over occupied slots:
$$
\alpha_{t,j} = \operatorname{softmax}_j\!\left(\frac{q_t^\top k_j}{\sqrt{d}} + \log f_j\right),
\qquad r_t = \sum_j \alpha_{t,j} v_j .
$$
Write is a convex update of every slot, as in the proposal:
$$
m_{t,j} = (1 - g_{t,j})\, m_{t-1,j} + g_{t,j}\, u_t, \qquad m_j = (k_j, v_j),\; u_t = (k_t, v_t),
$$
with $a_j \leftarrow (a_j + 1)(1 - g_{t,j})$, $c_j \leftarrow c_j + g_{t,j}$, and
$f_j \leftarrow (1 - g_{t,j}) f_j + g_{t,j}$. At each position the memory is read *before*
it is written, so a query token sees what was stored before it.

**Answer head.** A two-layer MLP on $[h_t ; r_t]$ produces logits over the vocabulary.
Cross-entropy is computed at answer positions only.

## 2.3 The interference-aware controller

For each slot $j$ the controller builds a six-dimensional feature vector
$$
\phi_{t,j} = \big[\, \underbrace{\cos(k_t, k_j),\; \tfrac{1}{\sqrt{d}}\lVert v_t - v_j \rVert}_{\text{interference}},\;
\underbrace{1 - \max_{j'} \cos(k_t, k_{j'})}_{\text{novelty}},\; a_j/T,\; c_j/T,\; f_j \,\big],
$$
where $T$ is the sequence length. The first feature is high when the incoming key would
land on a slot that already holds a similar key, which is the condition under which a
write *overwrites* rather than *adds*. The second says whether that overwrite would change
the stored answer. Together they distinguish a harmless duplicate (similar key, same
value) from a genuine key update (similar key, new value) and from a collision between
unrelated associations whose keys happen to be close.

A two-layer MLP maps $\phi_{t,j}$ to a logit $\ell_{t,j}$: this decides *where* a write
goes, from interference and slot age alone. A second small MLP decides *whether* to write
at all, producing a scalar logit $w_t$ from a pooled memory summary (novelty, mean
occupancy, mean value disagreement) concatenated with $c_t = W_c h_t \in \mathbb{R}^{16}$,
a low-dimensional projection of the current token's representation. The content input is
what lets the controller learn that a value token following a key is worth storing and a
filler token is not; without it, the decision to write could depend only on memory
statistics, and in the pilot diagnostics learning then depended on luck (Section 4.2). The gate is
$$
g_{t,j} = \sigma(w_t)\, \operatorname{onehot}\!\big(\arg\max_j \ell_{t,j}\big),
$$
with the straight-through estimator: the forward pass writes exactly one slot, and the
backward pass uses the gradient of $\operatorname{softmax}_j(\ell_{t,j})$ in its place.
The write can still be suppressed entirely through $\sigma(w_t)$. A fully soft gate was
tried first and did not learn: at initialization it spreads each association across all
slots, the memory never holds a retrievable entry, and so no gradient arrives to sharpen
it. Hard forward selection with a soft backward path is the same device ARM uses for its
slot router [@zeng2026arm]. The controller has 1,858 parameters in the pilot, plus 4,112 in the content projection; the full pilot model has 7.6M. Its decision
depends only on the current association and the memory, so it adds no per-sequence state
beyond the memory's own age, usage, and occupancy scalars, which are counted in the budget.

**H2 ablation.** `interference=False` zeroes the two interference features and leaves
everything else identical, including parameter count. This is also the input-only routing
baseline: a learned router that sees novelty and slot age but not conflict, which is the
simplified form of Raven's routing signal.

## 2.4 Baselines

All writers share the backbone, projections, memory, and answer head; only the gate rule
differs. Each is a deliberate simplification of the cited mechanism, re-implemented from
the paper's description because none of the three has public code at the time of writing.

| Writer | Rule | Models |
|---|---|---|
| Recency | $g_{t,j} = 1$ for the oldest (or first empty) slot, else 0 | trivial control |
| Input-only routing | controller with interference features zeroed | Raven [@afzal2026raven] |
| Residual cache | write to the oldest slot with gate $\sigma(\beta(\lVert r_t(k_t) - v_t\rVert/\sqrt{d} - \tau))$, learned $\tau, \beta$ | HOLA [@cui2026hola] |
| Position activation | slot $j$ unlocked at position $\lceil jT/S \rceil$; recency among unlocked slots | Proteus [@bayat2026proteus] |

## 2.5 Training and accounting

AdamW ($\beta = (0.9, 0.95)$, weight decay 0.1 on matrices), linear warmup then cosine
decay, gradient clipping at 1.0, bf16 autocast for the backbone with the memory loop in
fp32. Hyperparameters (learning rate, controller hidden size, gate temperature) are tuned
on validation only. State bytes per sequence are reported as backbone recurrent state
plus $S(2d + 3)$ floats of memory plus any writer statistics; for the pilot model that is
131 KB backbone, 33 KB memory, 0 writer.

# 3. Related Work

## 3.1 Survey method

Following the PRISMA 2009 structure [@moher2009prisma] in a light form suited to a course
paper: a stated question, explicit eligibility criteria, documented searches, and a flow
count. The question was *which mechanisms decide what a bounded-memory sequence model
writes, and has interference between stored associations been used as a write signal?*
Three arXiv searches on 2026-10-15, the proposal's references, and eleven foundational
papers identified 44 records; 31 were unique; 9 were screened out by title and abstract
(retrieval augmentation, agent memory, video, serving systems, surveys); 22 were assessed
and included. The full log with queries and per-record exclusion reasons is in the project
repository (`paper/prisma/search-log.md`).

## 3.2 Bounded-state backbones and their write rules

Linear attention [@katharopoulos2020linear] stores associations as a sum of outer
products $\sum_t v_t k_t^\top$; every token writes with equal weight. Fast-weight
programmers [@schlag2021fastweight] introduced the delta rule, which subtracts the state's
current prediction before adding, so a write *corrects* rather than accumulates. GLA
[@yang2023gla] adds a data-dependent decay gate; DeltaNet [@yang2024deltanet] and Gated
DeltaNet [@yang2024gateddeltanet] make the delta rule and its decay trainable at scale;
Mamba [@gu2023mamba] reaches a similar place through selective state-space parameters.
Based [@arora2024based] combines a small linear-attention state with sliding-window
attention and shows the recall-throughput trade-off directly. Gated Slot Attention
[@zhang2024gsa] and its successor GSA-2 [@li2026gsa2] organize the state as shared latent
slots with decoupled erase and write gates on both the key and value side. Variational
Linear Attention [@pandey2026vla] casts each write as a regularized least-squares step
and bounds the state norm. In all of these the write decision is a function of the token
and a learned gate; the state is addressed implicitly through key overlap, and the
mechanistic study by Lee et al. [@lee2026howlinear] shows that recall degrades with the
number of overlapping writes.

## 3.3 Allocation, routing, and eviction into explicit memory

Neural Turing Machines [@graves2014ntm] and the Differentiable Neural Computer
[@graves2016dnc] are the classical learned-allocation designs: NTM addresses by content
and location, DNC adds a usage vector and allocates the least-used slot. Our age and usage
features are DNC-style; our interference features are not in either model. Titans
[@behrouz2024titans] gates memory updates by *surprise*, the gradient of a reconstruction
loss, which is an input-side novelty signal. HOLA [@cui2026hola] admits tokens to an exact
cache when the delta-rule residual is large, again a property of the token against the
state's prediction rather than a per-slot conflict. Raven [@afzal2026raven] learns an
input-dependent router over a fixed slot set. ARM [@zeng2026arm] does the same for a
Transformer KV cache with a Gumbel-softmax router and sigmoid-gated blends. IndexMem
[@yang2026indexmem] trains an indexer to score KV entries for eviction. Fast-weight
Product Key Memory [@zhao2026fwpkm] updates only activated slots of a large sparse
memory. Proteus [@bayat2026proteus] sidesteps allocation by unlocking capacity on a
position schedule, and Memory Caching [@behrouz2026memcache] sidesteps it by letting
memory grow. Zoology [@arora2023zoology] and RULER [@hsieh2024ruler] supply the recall
evaluations the field uses.

## 3.4 The gap

Every allocation rule above conditions on the input (novelty, surprise, residual, router
logits) or on position (schedule, recency, usage). None computes, for each candidate slot,
whether the incoming association conflicts with the one stored there, even though the
capacity analyses say that this conflict is what limits recall. The controller in
Section 2.3 adds exactly that signal and nothing else, so the comparison isolates its
value.

# 4. Application and Results

## 4.1 Experimental protocol

*Factors.* Memory budget $S \in \{32, 64, 128\}$; associations $N \in \{64, 128\}$;
writer $\in$ {controller, no-interference, recency, residual, position}. Updates and
distractors fixed at $N/4$ each; delay swept separately at $N = 64$. Train at length 512,
test at 512, 1024, and 2048 with freshly generated sequences.

*Seeds and statistics.* Three seeds per cell. Primary metric is exact recall: the fraction
of answer positions predicted exactly, averaged over the 10,000 test sequences. Each cell
reports mean over seeds and a 95% bootstrap confidence interval over test sequences
[@efron1979bootstrap]; H1 is tested with a *paired* bootstrap on the per-sequence
difference between the controller and the best baseline, since all writers see the same
test sequences. Secondary metrics: recall by association density ($N/S$), state bytes,
training and evaluation throughput, and the write-gate statistics (mean gate mass, slot
entropy) that show *how* each writer uses its budget.

*Selection.* The learning rate for each writer and the controller's hidden size are chosen
on validation recall at $N = 64$, $S = 32$ and then frozen.

## 4.2 Pilot run

*What the pilot is for.* A single seed, run to validate the pipeline end to end, to find
the task size at which learning happens within a short budget, and to measure cost. The
numbers below are one seed and carry no statistical claim about H1 or H2.

*The proposal's setting did not learn.* The first pilot used the proposal's target:
$N = 64$ with 16 updates and 16 distractors, 32 slots, length 512, vocabulary 8192, 8000
steps at batch 64. The recency baseline finished at chance recall (0.01%, where chance is
1/4096). The controller was at chance after 2600 steps with the loss still at the unigram
floor, and was stopped. Diagnostics on smaller tasks then isolated three separate causes,
fixed one at a time with the smallest change that tested each hypothesis:

1. *Key alignment.* Projecting the stored key from the value token's position made the
   memory path useless until the backbone learned to copy keys forward. The one-step shift
   of Section 2.2 took the recency writer from 12% to 47% recall on an 8-pair task at equal
   steps.
2. *Diffuse gates.* The soft sigmoid-times-softmax gate smeared each write over all slots.
   The straight-through gate of Section 2.3 took the controller from 11% to 81% on the same
   task.
3. *Content-blind write decision.* With only memory statistics as input, whether to write
   could not depend on what the token was, so storing value tokens rather than filler was
   learned only when the backbone happened to bootstrap it. Adding the 16-dimensional
   content projection took a 16-pair task from 3% to 92% at 1400 steps, and a setting that
   the content-blind controller never learned in 5000 steps started learning within 1400.

After these fixes a one-factor scan from the smallest learnable task showed that vocabulary
(512 to 8192) and length (64 to 128) did not block learning, but the ratio of associations
to slots did, and the 64-association target was still at chance after 1400 steps, with one
run producing a NaN loss. The pilot therefore moved to the 24-association, 16-slot setting
of Section 2.1, which is the largest that learned within a 4000-step budget. Reaching the
proposal's target is the first item of future work (Section 5.3).

<!-- PILOT_TABLE -->

Table 1. Pilot run, seed 0, 24 associations and 16 slots, trained at length 128. Exact
recall on 10,000 test sequences with 95% bootstrap CI; cost columns measured on one RTX 5090.

<!-- PILOT_FIGURE -->

*Pipeline checks that passed.* Unit tests cover split disjointness, answer-position
masking, the slot update's identity at zero gate, gate range and normalization, and
gradient flow into the controller and into both memory projections. A 64-sequence
overfit check drives the loss below 0.1 in 300 steps, confirming that the read path can
carry the answer.

<!-- PILOT_DISCUSSION -->

# 5. Conclusions

## 5.1 What this draft establishes

The method is fully specified and implemented: a GLA backbone, a bounded slot memory with
a convex gated update, a 6-feature controller whose interference features can be switched
off without changing its size, and three baselines that isolate the novelty, residual, and
position signals used by the cited work. The survey shows that the interference signal is
absent from existing allocation rules, which is what makes H1 and H2 worth testing. The
pilot shows the pipeline trains and evaluates at a known cost, so the full sweep is
feasible on one GPU before the final deadline.

## 5.2 What the final paper will answer

Whether H1 holds, and at which densities; whether H2 holds, which separates "a learned
router helps" from "the interference signal helps"; how each writer's gate statistics
explain its errors; and where a 32-slot budget fails for every writer. Because the task is
synthetic, the findings bound what allocation can do on associative recall; they do not
establish language-model gains.

## 5.3 Risks identified by the pilot, and what changes before the final paper

1. **Training budget at the proposal's scale.** The 64-association target did not leave
   chance within the budgets tried. Recurrent models are known to need long training on
   MQAR, and Zoology's reference runs use tens of thousands of steps. The full sweep will
   use a longer schedule and a curriculum that starts from the learnable 16-pair setting
   and grows $N$, which also matches how the proposal varies density.
2. **Cold start of learned allocation.** With many more associations than slots, a random
   initial policy thrashes the memory and no retrieval signal reaches the controller. The
   content projection reduced this; the curriculum above is the remaining mitigation, and
   a recency prior on the slot logits is a cheap alternative to test.
3. **Write-loop cost.** The memory update is a sequential loop over tokens and takes 98% of
   step time (0.65 s per step at batch 64, length 512). A compiled step body gave 2x; the
   full sweep needs a vectorized or fused update, which is scheduled before the sweep.
4. **Numerical stability.** One 64-association run reached a NaN loss at step 1400 under
   bf16 autocast. The cause has not been isolated; the memory loop runs in fp32, so the
   backbone or the head is the suspect. A gradient-norm and loss guard with fp32 fallback
   will be added before the sweep.
5. **Baselines not yet trained.** The residual-cache and position-activation writers are
   implemented and unit-tested but have no training runs; their learned threshold and
   schedule need validation-set tuning before they count as fair comparisons.

<!-- PILOT_RISKS -->

# References
