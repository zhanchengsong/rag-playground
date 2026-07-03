# rag-playground

A from-scratch RAG loop with no orchestration framework, so the mechanism stays
visible. Corpus → chunk → embed → cosine kNN → stuff into a prompt → generate.
Runs against your local OpenAI-compatible endpoints (llama.cpp / MLX).

## Setup

```bash
pip install -r requirements.txt
```

Point it at your servers (defaults assume both on `localhost:8080`):

```bash
export CHAT_BASE_URL=http://localhost:8080/v1     # your Qwen3 chat model
export EMBED_BASE_URL=http://localhost:8081/v1    # your Qwen3-Embedding model
export CHAT_MODEL=qwen3
export EMBED_MODEL=qwen3-embedding
```

If a server isn't reachable or you'd rather use your own corpus, drop `.txt`
files in `data/essays/` and they're used instead of the HuggingFace download.

## Run

```bash
python rag.py                                   # interactive REPL
python rag.py -q "What does PG say about cofounders?"
python rag.py -q "..." -k 8                      # retrieve more chunks
python rag.py --rebuild                          # re-embed after changing knobs
```

First run downloads ~200 essays and embeds every chunk (one-time; cached to
`data/embeddings.npy`). Every later run loads the cache instantly unless the
corpus or chunking knobs change.

## What to actually do with it (the experiment ladder)

The code prints the retrieved chunks, their scores, and the **exact prompt**.
The learning is in perturbing one knob and watching the behavior shift.

- **See retrieval is just kNN.** Read `retrieve()` — three lines of numpy. There
  is no database magic.
- **Watch context consumption (Milestone 3).** Run the same query at `-k 1`,
  `4`, `12`. Find a query where the gold chunk lands at position ~6 and see the
  model ignore it — "lost in the middle," live. Try reordering `retrieved` in
  `answer()` and re-asking.
- **Break dense retrieval on purpose (Milestone 5).** Ask about "heart attack"
  when the essays only say "myocardial infarction"-style synonyms — watch pure
  vector search whiff, which is *why* BM25/hybrid exists. Crank `CHUNK_SIZE` to
  800 and watch each chunk's embedding turn to mush as it averages too many
  topics.
- **Toggle the Qwen3 instruction prefix.** `USE_QUERY_INSTRUCTION=0 python
  rag.py --rebuild ...` and measure whether it actually moves recall on your
  queries.

## Where this goes next

This is Milestone 2 of the build-from-scratch track. Natural next steps, each a
small diff on this base:

1. **Tokens & cache (M4):** swap word-chunking for token-chunking; print
   tokens-per-chunk; reason about which part of the prompt is a cacheable prefix
   and how the retrieved chunks bust it.
2. **Eval harness:** wire in a handful of question/answer pairs + Recall@k so
   every knob change produces a number, not a vibe.
3. **Hybrid + rerank:** add BM25, fuse with the dense scores (RRF), add a
   Qwen3-Reranker pass — then graduate the in-memory matrix to LanceDB.
4. **Query rewriting:** HyDE, multi-query/RAG-Fusion, decomposition — the
   headline interest, now with an eval to tell you which ones earn their latency.
```
