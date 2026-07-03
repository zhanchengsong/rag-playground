# RAG & LLM Knowledge Base — Learning Curriculum

A working plan for learning how RAG works from first principles, then how to
optimize it. Two tracks:

- **Track A — Understand the mechanism.** Build each piece from scratch and
  watch it behave. No frameworks; nothing stays a black box.
- **Track B — Optimize.** Eval-first playground where every technique is a
  hypothesis you confirm or kill with numbers.

Do A first to build intuition, then B so every optimization has an obvious
"this fixes failure mode X" story behind it. Check items off as you go.

**Guiding rule:** avoid LangChain / LlamaIndex here — they hide the exact
mechanics you're trying to learn. Build thin wrappers yourself; use libraries
only for the boring parts (BM25, vector store, metrics).

---

## Track A — Understand the mechanism (build from scratch)

- [x] **M1 — See an embedding space.** Embed a few hundred sentences, reduce
  with UMAP, plot. Watch semantically related text cluster. Then implement
  brute-force cosine kNN yourself in ~10 lines of numpy.
  *Learn:* "retrieval" is nearest-neighbor search in a vector space — no
  database magic. Use a labeled set (20 Newsgroups / AG News) so clusters are
  checkable by eye. *Lab scaffolded: `lab1_embedding_space.py` — run the
  neural-vs-`--tfidf` comparison and the zero-word-overlap query.*

- [x] **M2 — RAG in ~100 lines, no framework.** Chunk → embed → cosine top-k →
  prompt template → local chat model, all in one readable file. Print the exact
  prompt. *Scaffolded in this repo (`rag.py`).*
  *Learn:* RAG is a plain data transformation — string assembly around a kNN
  lookup.

- [ ] **M3 — Watch the LLM consume the context.** Vary `k`. Reorder chunks.
  Remove the gold chunk and watch it hallucinate. Inject a plausible distractor.
  If you can, inspect attention over the prompt.
  *Learn:* "lost in the middle," why ordering matters, how faithfulness
  degrades as `k` grows. The single most clarifying experiment here.

- [ ] **M4 — Tokens & the context budget.** Swap word-chunking for
  token-chunking; measure tokens-per-chunk; watch chunk size trade retrieval
  granularity against context spend. Identify which part of a RAG prompt is a
  stable cacheable prefix and how the retrieved chunks bust it.
  *Learn:* context economy made concrete — ties straight into KV-cache
  prefix-reuse.

- [ ] **M5 — Break it on purpose.** Query "heart attack" against docs that say
  "myocardial infarction" (dense retrieval whiffs → why hybrid/BM25 exists).
  Make a chunk huge and watch its embedding average into mush. Flood the corpus
  with near-duplicates and watch top-k fill with redundancy.
  *Learn:* you derive the Track B optimizations from first principles instead of
  cargo-culting them.

- [ ] **M6 — Move to the application layer.** With the mechanism intuitive,
  Track B is the natural continuation.

- [ ] *Optional detour — generation internals.* Run nanoGPT / hand-implement a
  single attention block, to close the loop between "I understand attention" and
  "I built the thing that generates the answer."

---

## Track B — Optimize (eval-first playground)

- [ ] **P0 — Eval harness first.** Pick one dataset with relevance judgments
  (qrels) and one with QA pairs. Two scoreboards: pure-IR (Recall@k, nDCG@k,
  MRR) and end-to-end (faithfulness, answer relevancy, context precision/recall).
  Target: `python eval.py --config baseline.yaml` appends one row to a results
  table. Everything after this is a new config row.

- [ ] **P1 — Naive baseline.** Fixed chunking, dense retrieval, stuff top-k,
  single generation. Deliberately mediocre — this is the control group.

- [ ] **P2 — Retrieval quality.** Sweep chunk size/overlap; try
  structure/AST-aware chunking; add BM25 and hybrid via reciprocal rank fusion;
  add a reranker over fused candidates.
  *Test:* does hybrid beat dense on keyword queries? How much does the reranker
  recover from a deliberately bad retriever?

- [ ] **P3 — Query rewriting** (headline interest). Implement and benchmark one
  at a time — see the reference checklist below. Plot quality **vs latency**;
  most cost a generation call per query and won't justify it on easy queries.

- [ ] **P4 — Context economy & generation.** Contextual retrieval (LLM-generated
  chunk summary prepended before embedding), context compression / sentence
  filtering, citation-forced generation, "lost in the middle" ordering.

- [ ] **P5 — Advanced / agentic.** Iterative retrieve→reason→retrieve, Self-RAG
  style "do I have enough?" gating, and **DSPy** to *optimize* the rewriting
  prompts/few-shots against your eval metric instead of hand-tuning.

### Query rewriting techniques (P3 reference)

- [ ] **Query expansion / rewriting** — LLM rewrites the raw query before embedding.
- [ ] **HyDE** — generate a hypothetical answer, embed *that* (closes the
  question↔document gap).
- [ ] **Multi-query / RAG-Fusion** — N paraphrases, retrieve each, fuse with RRF.
- [ ] **Step-back prompting** — derive a more general question, retrieve
  background, then answer.
- [ ] **Query decomposition** — split multi-hop questions into sub-questions
  (needs a multi-hop dataset).
- [ ] **Self-query** — LLM extracts metadata filters from natural language.
- [ ] **Routing** — classify whether to retrieve at all, and which index to hit.

---

## Datasets

### For *understanding* (small + readable — verify retrieval by eye)

- [ ] **Paul Graham essays** (~200) — canonical RAG toy corpus; already wired
  into this repo. Best for M1–M3.
- [ ] **Wikipedia subset** (SQuAD / NQ passage set) — inspectable, ships with
  natural questions.
- [ ] **SciFact** (~5k docs) — tiny *and* has qrels; lightweight ground truth.
- [ ] **20 Newsgroups / AG News** — labeled categories, ideal for the M1
  embedding-space visualization.
- [ ] **Your own corpus** — arXiv LLM-infra papers, your notes, your codebase
  (the AST-chunking angle). Most motivating: you already know the right answers.

### For *measuring* (benchmark-grade)

- [ ] **BEIR** — start with small ones: SciFact, FiQA, NFCorpus, TREC-COVID,
  ArguAna. Gold standard for the P2/P3 retrieval numbers.
- [ ] **MS MARCO** passage ranking — canonical; sample a subset for a laptop.
- [ ] **Multi-hop QA** — HotpotQA, 2WikiMultiHopQA, MuSiQue (essential for the
  decomposition work — these *require* rewriting to do well).
- [ ] **Open QA** — Natural Questions, TriviaQA, KILT suite (one Wikipedia
  snapshot serves many task types).
- [ ] **RAG-specific** — TREC RAG / MS MARCO V2.1, Researchy Questions, and
  RAGTruth (hallucination-annotated, for P4 faithfulness stress tests).

### Access

- **`ir_datasets`** — unified access to most IR collections *with qrels
  attached*; don't hand-assemble relevance judgments.
- **HuggingFace `datasets`** — the QA corpora.
- **Ragas synthetic generation** — manufacture question/answer/context triples
  from any unlabeled corpus (for the bring-your-own case).

**Suggested minimal set:** SciFact or FiQA (clean retrieval numbers) + HotpotQA
(query-rewriting story) + one BYO corpus (vibes). Covers every phase.

---

## Stack (fitted to the RTX 5090 + 128GB Mac)

- **Embeddings:** Qwen3-Embedding (0.6B fast iter / 8B quality; tops the
  open-source MTEB leaderboard, instruction-aware). Keep **BGE-M3** around — it
  emits dense + sparse + multi-vector from one model, ideal for the hybrid phase.
- **Reranker:** Qwen3-Reranker or BGE-reranker-v2-m3.
- **Vector store:** LanceDB (embedded, zero-server — great for a playground) or
  Qdrant (native hybrid + filtering). BM25 via `bm25s` or Tantivy.
- **Generation:** local Qwen3.6 on the 5090 via llama.cpp, or the 30B MoE on the
  Mac via MLX — everything through the OpenAI-compatible endpoint.
- **Eval:** Ragas (the open-source standard; can use your local LLM as judge) +
  `ir_measures` for clean nDCG/MRR/Recall. DeepEval for pytest-style component
  tests.
- **Optimizer:** DSPy (P5).

---

## Status / next action

- [x] Repo scaffolded — M2 running (`rag.py`, no-framework RAG loop).
- [x] M1 lab scaffolded — `lab1_embedding_space.py` (UMAP plot, kNN purity
  metric, hand-rolled cosine kNN, neural vs TF-IDF comparison).
- [ ] **Run M1:** both modes, compare PNGs + purity; try the
  "kicking a ball into a net" zero-overlap query.
- [ ] **Next build:** M3 experiment — same query at `k = 1, 4, 12`, find the
  "lost in the middle" failure, then add a small Q/A set + Recall@k so knob
  changes produce numbers instead of vibes (bridges into P0).

## Repo contents

| File | Purpose |
|---|---|
| `PLAN.md` | this curriculum (kept current as milestones land) |
| `config.py` | every knob: endpoints, models, k, chunking |
| `corpus.py` | PG-essay loader + word-window chunking |
| `rag.py` | M2: the ~100-line no-framework RAG loop |
| `lab1_embedding_space.py` | M1: embedding-space plot + hand-rolled kNN |
