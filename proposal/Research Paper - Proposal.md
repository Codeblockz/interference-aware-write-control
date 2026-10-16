---
course: EN.705.742.81.FA26
type: research-paper
title: "Interference-aware write control for bounded-memory recurrent models"
status: proposal
filed: 2026-10-03
tags: [research-paper, aml]
---

# 742 Research Paper — Proposal

**The idea in one line:** a learned write controller that uses *interference between stored
associations* (not just input novelty) to decide what a recurrent model writes into a fixed
memory budget.

**Original file:** `Ellis - Research Proposal - Bounded Memory.docx` (in this folder)
**Assignments:** [[RP1 - Research Paper Pick Topic/assignment|Pick Topic]] (09/13, graded 100/100) ·
[[RP2 - Research Paper Midpoint Check/assignment|Midpoint Check]] (**due 10/25**) ·
[[RP3 - Research Paper Final/assignment|Final]] (**due 12/13**)
**Rubric:** [[Research Project Rubric]]

> Filed from the .docx on 2026-10-03 so it is searchable in the vault. The .docx is the
> original; if you revise the proposal, edit there and re-file, or edit this note directly.

---

**Paper outline:** Introduction and Goals; Hypotheses and Method; Related Work; Application and Results; Conclusions.

## Problem and candidate solution

Recurrent sequence models use bounded memory but can overwrite information needed later. HOLA adds an exact cache [1], Raven learns sparse memory routing [2], and Proteus progressively activates capacity [3]. I will investigate whether explicitly estimating interference between stored associations improves allocation at a fixed memory budget. My candidate solution is a learned write controller that considers input novelty and conflicts among similar stored keys. Its advantage over existing allocation mechanisms is an untested hypothesis.

## Research hypotheses

**H1:** At matched total memory and training compute, the controller will improve exact recall by at least five percentage points over the strongest validation-selected baseline on tests with 64-128 distinct associations and a 32-slot budget.

**H2:** Removing the interference signal while holding controller size fixed will reduce this gain, indicating that the signal supplies useful information.

## Applied method and dataset

I will train a 20-million-parameter recurrent model in PyTorch with 32, 64, or 128 memory slots. A small neural controller will produce differentiable write gates from the current input and memory, using key similarity and value disagreement as interference features. A simplified slot update is:

$m_{t,j}  =  (1  -  g_{t,j})m_{t - 1,j}  +  g_{t,j}u_{t}$

Here m is a slot vector, u is the incoming representation, and g is a learned gate between zero and one. The model and controller will learn jointly through query-answer cross-entropy.

Data will use Zoology's public multi-query associative recall generator [4], extended with key updates and distractors. I will generate 100,000 training, 5,000 validation, and 10,000 test sequences with disjoint examples and new key-value assignments. I will vary distinct associations, redundancy, and query delay independently; train at 128-512 tokens and test up to 2,048 tokens.

![[Research Paper - Proposal - architecture.png]]

*Figure: The input sequence passes through an encoder and a learned write controller into bounded memory. The answer readout receives both an encoded query and the stored memory. Memory supplies novelty and interference estimates to the controller for subsequent writes.*

## Evaluation and expected learning

Controls will use the same backbone with recency retention, residual-based caching [1], input-only routing [2], or position-based activation [3]. I will match parameter counts and training compute, count all recurrent state, caches, and controller statistics, and tune learning rates and gate settings only on validation data. Across three seeds, I will report exact recall, errors by association density, state bytes, latency, and paired bootstrap 95% confidence intervals. An initial small run will establish GPU memory and runtime. The conclusions will identify when allocation helps, whether the hypotheses hold, and where limited capacity still causes failure; synthetic-task findings will not establish general language-model gains.

## References

[1] Cui (2026). [A Hippocampus for Linear Attention: An Exact Memory for What the Recurrent State Forgets.](https://arxiv.org/abs/2607.02303) arXiv:2607.02303.

[2] Afzal et al. (2026). [Raven: High-Recall Sequence Modeling with Sparse Memory Routing.](https://arxiv.org/abs/2607.25357) arXiv:2607.25357.

[3] Bayat et al. (2026). [Proteus: Incremental Memory Activation for Long-Context Sequence Modeling.](https://arxiv.org/abs/2608.16844) arXiv:2608.16844.

[4] Arora et al. (2023). [Zoology: Measuring and Improving Recall in Efficient Language Models.](https://arxiv.org/abs/2312.04927) arXiv:2312.04927.
