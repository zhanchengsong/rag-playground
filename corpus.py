"""
Corpus loading + chunking. Deliberately dumb and inspectable:
chunking is word-windowed (not token-based) so you can eyeball every chunk.
Token-based chunking + the tokens/context-budget experiment is Milestone 4.
"""
import os
import glob
import config


def load_essays():
    """Return a list of {"title", "text"}.

    Order of preference:
      1. Local .txt files in data/essays/  (bring-your-own corpus)
      2. The HuggingFace dataset in config.HF_DATASET
    """
    # 1) Local files win, so you can swap in your own corpus (your notes, arXiv
    #    papers, your codebase) just by dropping .txt files in the folder.
    if os.path.isdir(config.LOCAL_ESSAYS):
        paths = sorted(glob.glob(os.path.join(config.LOCAL_ESSAYS, "*.txt")))
        if paths:
            essays = []
            for p in paths:
                with open(p, encoding="utf-8") as f:
                    essays.append({"title": os.path.basename(p)[:-4], "text": f.read()})
            print(f"[corpus] loaded {len(essays)} local essays from {config.LOCAL_ESSAYS}")
            return essays

    # 2) HuggingFace dataset.
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "Install deps (`pip install -r requirements.txt`) or drop .txt files "
            f"in {config.LOCAL_ESSAYS}/"
        )

    ds = load_dataset(config.HF_DATASET, split="train")
    # Be tolerant about column names across dataset versions.
    cols = ds.column_names
    text_col  = next((c for c in ("text", "content", "essay", "body") if c in cols), None)
    title_col = next((c for c in ("title", "name", "subject") if c in cols), None)
    if text_col is None:
        raise SystemExit(f"Couldn't find a text column in {config.HF_DATASET}; got {cols}")

    essays = []
    for i, row in enumerate(ds):
        text = (row[text_col] or "").strip()
        if not text:
            continue
        title = row[title_col] if title_col else f"essay_{i}"
        essays.append({"title": str(title), "text": text})
    print(f"[corpus] loaded {len(essays)} essays from HF dataset '{config.HF_DATASET}'")
    return essays


def chunk_text(text, size, overlap):
    """Split text into overlapping windows of `size` words."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        if window:
            chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def get_chunks():
    """Flatten the corpus into a list of {"id", "title", "text"} chunks."""
    chunks = []
    for essay in load_essays():
        for j, piece in enumerate(chunk_text(essay["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            chunks.append({
                "id": f"{essay['title']}#{j}",
                "title": essay["title"],
                "text": piece,
            })
    print(f"[corpus] {len(chunks)} chunks "
          f"(size={config.CHUNK_SIZE}w, overlap={config.CHUNK_OVERLAP}w)")
    return chunks
