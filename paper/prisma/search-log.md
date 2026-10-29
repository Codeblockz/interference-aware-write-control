# Related-work search log (PRISMA 2009, light form)

Follows the identification / screening / eligibility / included structure of Moher et al.
(2009). Scope is deliberately narrow: this is a course paper, not a full systematic review.

## Research question

Which mechanisms decide *what* a bounded-memory recurrent sequence model writes, and has
interference between stored associations been used as a write signal?

## Eligibility criteria

Included if all hold:

1. Sequence model with a fixed-size (bounded) memory or recurrent state.
2. Describes a mechanism for selecting, gating, routing, or evicting writes into that memory.
3. Reports recall-style evaluation (associative recall, needle-in-a-haystack, RULER) or
   an explicit capacity analysis.
4. 2020 or later, English, arXiv preprint or peer-reviewed venue.

Excluded: retrieval-augmented or external-database memory (RAG), KV-cache compression for
softmax Transformers unless the eviction policy is learned, agent memory frameworks, and
vision/video memory.

## Identification (2026-10-15)

| # | Source | Query | Hits reviewed |
|---|--------|-------|---------------|
| S1 | arXiv via web search | linear attention bounded memory write gate interference associative recall | 9 |
| S2 | arXiv via web search | learned memory allocation eviction policy recurrent sequence model fixed-size state recall | 10 |
| S3 | arXiv via web search | key-value memory slots sequence model routing capacity long-context retrieval | 10 |
| S4 | Proposal references and their citations | HOLA, Raven, Proteus, Zoology | 4 |
| S5 | Known foundational work (no search) | NTM, DNC, linear attention, fast weights, GLA, DeltaNet, Gated DeltaNet, Mamba, Based, GSA, Titans | 11 |
| S6 | Citation chasing (2026-10-24), web search for title candidates surfaced GSA's parent and a 2026 addressable-state paper | ABC (2110.02488), RAM-Net (2602.11958) | 2 |

Records identified: 44 on 2026-10-15 plus 2 on 2026-10-24. After removing duplicates (Raven and HOLA appeared in several
searches; survey and video results dropped): 33 unique.

## Screening (title + abstract)

Excluded at screening, with reason:

- LiveMem (2608.02515): inference-serving system, not a write mechanism. (criterion 2)
- Memory for Large Language Models (2607.25380): survey. (background only)
- MemoRAG (2409.05591): retrieval augmentation. (exclusion: RAG)
- SlotMemory (2605.31033): video generation. (exclusion: video)
- MKA (2603.20586): keeps full cache, routes attention. (criterion 1)
- Memory-Managed Long-Context Attention (2606.28876): preliminary study, editable table for agents. (exclusion: agent memory)
- Learning What Not to Forget (2606.20954): agent memory eviction. (exclusion: agent memory)
- Chain-of-Thought and Compressed Looped Transformers (2605.30757): theory of CoT budget, cited only for the capacity-scaling observation. (criterion 2)
- GatedFWA (2512.07782): windowed attention with gated memory; abstract verified 2026-10-28: windowed attention with a gated memory, no admission rule over stored content; excluded (criterion 2).

Screened out: 9. Passed to eligibility: 24.

## Eligibility (full abstract read; full text where available)

Included (24):

**Bounded-state backbones and their write rules**
Katharopoulos 2020 (linear attention); Schlag 2021 (fast weights, delta rule);
Yang 2023 (GLA, gated decay); Yang 2024a (DeltaNet); Yang 2024b (Gated DeltaNet);
Gu 2023 (Mamba, selective state); Arora 2024 (Based); Peng 2021 (ABC, bounded memory with a
learned control function; GSA's parent); Zhang 2024 (Gated Slot Attention);
Li 2026 (GSA-2, two-sided key/value correction); Pandey 2026 (VLA, regularized writes).

**Allocation, routing, and eviction into bounded memory**
Graves 2014 (NTM, content + location addressing); Graves 2016 (DNC, usage-based allocation);
Cui 2026 (HOLA, residual-triggered exact cache); Afzal 2026 (Raven, input-dependent slot
routing); Bayat 2026 (Proteus, position-scheduled activation); Zeng 2026 (ARM, Gumbel
router over slots); Yang 2026 (IndexMem, learned KV eviction); Zhao 2026 (FwPKM, sparse
fast-weight slots); Xiao 2026 (RAM-Net, key-addressed sparse slots, interference named as the
target but addressed from the key alone; NeurIPS 2026); Behrouz 2025 (Titans, surprise-gated memory); Behrouz 2026 (Memory
Caching, growing memory as the alternative to allocation).

**Capacity and interference analyses; evaluation protocols**
Lee 2026 (How Linear Attention Remembers: interference grows with load, not elapsed
context); Arora 2023 (Zoology, MQAR); Hsieh 2024 (RULER).

## Flow summary

| Stage | Count |
|-------|-------|
| Identified | 46 (+5 online-caching papers hand-added 2026-10-27 from the discussion) |
| After duplicate removal | 33 |
| Screened out (title/abstract) | 9 |
| Assessed for eligibility | 24 |
| Included in synthesis | 29 (24 + 5) |

## Gap this paper targets

Every included allocation mechanism conditions writes on the *input* (novelty, surprise,
residual, router logits from the token) or on *position* (schedule, recency). None uses a
direct estimate of conflict between the incoming association and the associations already
stored, although Lee 2026 shows that such conflict is what limits recall. That is the
hypothesis under test.

## Limitations of this search

Single reviewer, web search over arXiv rather than database export, citation chasing done
only once (2026-10-24, two additions), abstracts only for most 2026 entries. The 2026 papers were verified against their
arXiv abstract pages on 2026-10-15; their reported numbers have not been checked against
full text.
