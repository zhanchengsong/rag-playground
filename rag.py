"""
A whole RAG pipeline you can read top to bottom. No framework.

    corpus -> chunk -> embed -> [cosine kNN] -> stuff into prompt -> generate

Run it:
    python rag.py                       # interactive REPL
    python rag.py -q "What does PG say about cofounders?"
    python rag.py -q "..." -k 8         # change how many chunks are retrieved
    python rag.py --rebuild             # re-embed the corpus from scratch

Everything that matters -- the retrieved chunks, their scores, and the EXACT
prompt sent to the model -- is printed. Watching the prompt is the point.
"""
import os
import json
import argparse
import numpy as np
from openai import OpenAI

import config
import corpus

# Two clients: the generation endpoint and the embedding endpoint may differ.
_chat  = OpenAI(base_url=config.CHAT_BASE_URL,  api_key=config.API_KEY)
_embed = OpenAI(base_url=config.EMBED_BASE_URL, api_key=config.API_KEY)

VEC_PATH  = os.path.join(config.CACHE_DIR, "embeddings.npy")
META_PATH = os.path.join(config.CACHE_DIR, "index_meta.json")


# --- Embedding -------------------------------------------------------------
def embed_texts(texts, is_query=False):
    """Embed a list of strings -> (n, d) float32 matrix, L2-normalized.

    Normalizing here means cosine similarity is just a dot product later.
    The query gets the Qwen3 instruction prefix; documents never do.
    """
    if is_query and config.USE_QUERY_INSTRUCTION:
        texts = [config.QUERY_INSTRUCTION + t for t in texts]

    out = []
    for i in range(0, len(texts), config.EMBED_BATCH):
        batch = texts[i:i + config.EMBED_BATCH]
        resp = _embed.embeddings.create(model=config.EMBED_MODEL, input=batch)
        out.extend(d.embedding for d in resp.data)

    mat = np.asarray(out, dtype=np.float32)
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return mat


# --- Index (build once, cache to disk) -------------------------------------
def build_index(rebuild=False):
    """Return (chunks, matrix). Re-embeds only when the corpus/knobs change."""
    chunks = corpus.get_chunks()
    meta = {
        "n": len(chunks),
        "chunk_size": config.CHUNK_SIZE,
        "overlap": config.CHUNK_OVERLAP,
        "embed_model": config.EMBED_MODEL,
    }

    if not rebuild and os.path.exists(VEC_PATH) and os.path.exists(META_PATH):
        with open(META_PATH) as f:
            cached = json.load(f)
        if cached.get("meta") == meta:
            mat = np.load(VEC_PATH)
            print(f"[index] loaded cached embeddings {mat.shape}")
            return cached["chunks"], mat
        print("[index] config changed -> re-embedding")

    print(f"[index] embedding {len(chunks)} chunks via {config.EMBED_BASE_URL} ...")
    mat = embed_texts([c["text"] for c in chunks], is_query=False)
    np.save(VEC_PATH, mat)
    with open(META_PATH, "w") as f:
        json.dump({"meta": meta, "chunks": chunks}, f)
    print(f"[index] done, embeddings {mat.shape}")
    return chunks, mat


# --- Retrieval: brute-force cosine kNN, ~3 lines ---------------------------
def retrieve(query, chunks, mat, k):
    q = embed_texts([query], is_query=True)[0]   # (d,)
    scores = mat @ q                              # cosine sim, since all normalized
    top = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top]


# --- Prompt assembly: a string-building problem ----------------------------
def build_prompt(query, retrieved):
    context = "\n\n".join(
        f"[{n+1}] (from \"{c['title']}\")\n{c['text']}"
        for n, (c, _) in enumerate(retrieved)
    )
    system = (
        "Answer the question using ONLY the numbered context passages below. "
        "Cite the passages you use like [1], [2]. If the answer isn't in the "
        "context, say so plainly."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}"
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# --- Glue ------------------------------------------------------------------
def answer(query, chunks, mat, k, show_prompt=True):
    retrieved = retrieve(query, chunks, mat, k)

    print("\n--- retrieved ---")
    for n, (c, s) in enumerate(retrieved):
        preview = c["text"][:110].replace("\n", " ")
        print(f"  [{n+1}] {s:.3f}  {c['id']:<28} {preview}...")

    messages = build_prompt(query, retrieved)
    if show_prompt:
        print("\n--- prompt sent to model ---")
        for m in messages:
            print(f"<{m['role']}>\n{m['content']}\n")

    resp = _chat.chat.completions.create(
        model=config.CHAT_MODEL, messages=messages, temperature=0.2,
    )
    print("--- answer ---")
    print(resp.choices[0].message.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", help="one-shot query; omit for REPL")
    ap.add_argument("-k", type=int, default=config.TOP_K, help="chunks to retrieve")
    ap.add_argument("--rebuild", action="store_true", help="re-embed the corpus")
    ap.add_argument("--no-prompt", action="store_true", help="hide the assembled prompt")
    args = ap.parse_args()

    chunks, mat = build_index(rebuild=args.rebuild)

    if args.query:
        answer(args.query, chunks, mat, args.k, show_prompt=not args.no_prompt)
        return

    print("\nType a question (empty line / Ctrl-C to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        answer(q, chunks, mat, args.k, show_prompt=not args.no_prompt)


if __name__ == "__main__":
    main()
