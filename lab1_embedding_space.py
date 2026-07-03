"""
Lab 1 -- See an embedding space (Track A, M1)

Goal: make "retrieval is just nearest-neighbor search in a vector space"
something you've SEEN, not read.

    sentences -> embed -> (a) UMAP to 2D and plot: labels emerge as clusters
                          (b) cosine kNN by hand (~10 lines of numpy)

Run:
    python lab1_embedding_space.py                 # AG News, local embedder
    python lab1_embedding_space.py --n 400         # fewer sentences (faster)
    python lab1_embedding_space.py --tfidf         # no LLM: lexical baseline
    python lab1_embedding_space.py --query "the fed raised interest rates"

Uses the same config.py / endpoints as rag.py. If your embedding server is
down (or you pass --tfidf), it falls back to TF-IDF + SVD -- a purely LEXICAL
embedding. Comparing the two plots is itself the first experiment: TF-IDF
clusters on shared words, the neural embedder clusters on meaning.

Outputs: embedding_space.png (the plot) + printed kNN results.
"""
import argparse
import numpy as np

import config

# ----------------------------------------------------------------------------
# 1. A few hundred labeled sentences.
#    AG News: 4 classes (World / Sports / Business / Sci-Tech). Labels are the
#    ground truth we'll check the clusters against -- the model never sees them.
# ----------------------------------------------------------------------------
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# Tiny built-in corpus so the lab still runs with zero downloads.
_FALLBACK = [
    ("UN peacekeepers deployed to the border region after ceasefire talks stalled", 0),
    ("The prime minister dissolved parliament and called early elections", 0),
    ("Diplomats meet in Geneva to negotiate the trade embargo", 0),
    ("Rebel forces seized the provincial capital overnight", 0),
    ("The striker scored a hat-trick in the cup final", 1),
    ("The champion defended her title in straight sets at the open", 1),
    ("Coach fired after the team's seventh straight loss", 1),
    ("Olympic committee announces host city for the winter games", 1),
    ("Shares tumbled after the company missed quarterly earnings estimates", 2),
    ("The central bank raised interest rates to fight inflation", 2),
    ("The merger creates the largest retailer in the region", 2),
    ("Oil prices surged as supply concerns mounted", 2),
    ("Researchers unveiled a faster chip built on a 2nm process", 3),
    ("The startup released an open-source framework for training models", 3),
    ("Astronomers detected water vapor on a distant exoplanet", 3),
    ("A software flaw exposed millions of user records", 3),
] * 12  # ~192 rows; enough to see structure


def load_sentences(n):
    """Return (texts, labels). Tries AG News via HuggingFace, else fallback."""
    try:
        from datasets import load_dataset
        ds = load_dataset("ag_news", split="train").shuffle(seed=42).select(range(n))
        texts = [r["text"] for r in ds]
        labels = np.array([r["label"] for r in ds])
        print(f"[data] {n} AG News sentences (4 classes)")
        return texts, labels
    except Exception as e:
        print(f"[data] AG News unavailable ({type(e).__name__}); using built-in mini corpus")
        texts = [t for t, _ in _FALLBACK]
        labels = np.array([l for _, l in _FALLBACK])
        return texts, labels


# ----------------------------------------------------------------------------
# 2. Embed. Neural (your local Qwen3-Embedding) or lexical (TF-IDF + SVD).
# ----------------------------------------------------------------------------
def embed_neural(texts):
    from openai import OpenAI
    client = OpenAI(base_url=config.EMBED_BASE_URL, api_key=config.API_KEY)
    out = []
    for i in range(0, len(texts), config.EMBED_BATCH):
        resp = client.embeddings.create(model=config.EMBED_MODEL,
                                        input=texts[i:i + config.EMBED_BATCH])
        out.extend(d.embedding for d in resp.data)
        print(f"\r[embed] {min(i+config.EMBED_BATCH, len(texts))}/{len(texts)}", end="")
    print()
    return np.asarray(out, dtype=np.float32)


def embed_tfidf(texts, dim=256):
    """Lexical baseline: bag-of-words -> SVD. No neural net anywhere."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    X = TfidfVectorizer(max_features=20000, stop_words="english").fit_transform(texts)
    dim = min(dim, X.shape[1] - 1, len(texts) - 1)
    return TruncatedSVD(n_components=dim, random_state=42).fit_transform(X).astype(np.float32)


def normalize(mat):
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)


# ----------------------------------------------------------------------------
# 3. THE POINT: cosine kNN by hand. This is all "retrieval" is.
# ----------------------------------------------------------------------------
def knn(query_vec, mat, k=5):
    """mat: (n, d) L2-normalized. query_vec: (d,) L2-normalized."""
    scores = mat @ query_vec          # cosine similarity == dot product
    top = np.argsort(-scores)[:k]     # indices of the k most similar rows
    return top, scores[top]


# ----------------------------------------------------------------------------
# 4. Project to 2D and plot. UMAP if installed, else PCA.
# ----------------------------------------------------------------------------
def project_2d(mat):
    try:
        import umap
        print("[plot] reducing with UMAP")
        return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(mat)
    except ImportError:
        from sklearn.decomposition import PCA
        print("[plot] umap-learn not installed -> PCA (pip install umap-learn for nicer plots)")
        return PCA(n_components=2, random_state=42).fit_transform(mat)


def plot(xy, labels, title, path="embedding_space.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7))
    for lab in np.unique(labels):
        m = labels == lab
        ax.scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.7,
                   label=LABEL_NAMES[lab] if lab < len(LABEL_NAMES) else str(lab))
    ax.set_title(title)
    ax.legend()
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[plot] saved {path}")


# ----------------------------------------------------------------------------
# 5. A number, not just a picture: do nearest neighbors share the label?
#    (kNN label agreement -- crude but honest measure of cluster quality.)
# ----------------------------------------------------------------------------
def neighbor_purity(mat, labels, k=5):
    agree = 0
    for i in range(len(mat)):
        scores = mat @ mat[i]
        scores[i] = -np.inf                       # exclude self
        nn = np.argsort(-scores)[:k]
        agree += np.mean(labels[nn] == labels[i])
    return agree / len(mat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="number of sentences")
    ap.add_argument("--tfidf", action="store_true", help="force the lexical baseline")
    ap.add_argument("--query", default="the central bank cut rates amid recession fears")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    texts, labels = load_sentences(args.n)

    mode = "tfidf"
    if not args.tfidf:
        try:
            mat = embed_neural(texts)
            mode = "neural"
        except Exception as e:
            print(f"[embed] endpoint failed ({type(e).__name__}: {e}) -> TF-IDF fallback")
    if mode == "tfidf":
        mat = embed_tfidf(texts)
    mat = normalize(mat)
    print(f"[embed] matrix {mat.shape} ({mode})")

    # --- the picture ---
    xy = project_2d(mat)
    plot(xy, labels, f"{len(texts)} sentences, {mode} embeddings "
                     f"(colors = true labels, never shown to the model)")

    # --- the number ---
    purity = neighbor_purity(mat, labels, k=args.k)
    print(f"[metric] {args.k}-NN label agreement: {purity:.1%}  "
          f"(random would be ~{1/len(np.unique(labels)):.0%})")

    # --- the mechanism: retrieve for a query ---
    print(f"\n[knn] query: {args.query!r}")
    if mode == "neural":
        q = embed_neural([args.query])[0]
    else:
        # NOTE the classic TF-IDF trap: the vectorizer was fit on the corpus, so
        # we re-fit including the query. Fine for a demo; in rag.py the neural
        # embedder maps ANY text into the same space -- that's the upgrade.
        mat2 = normalize(embed_tfidf(texts + [args.query]))
        mat, q = mat2[:-1], mat2[-1]
    q = q / (np.linalg.norm(q) + 1e-12)

    idx, scores = knn(q, mat, k=args.k)
    for rank, (i, s) in enumerate(zip(idx, scores), 1):
        print(f"  {rank}. {s:.3f} [{LABEL_NAMES[labels[i]]:<8}] {texts[i][:90]}")

    print("\nThings to try:")
    print("  * run once with --tfidf and once against your endpoint; compare the")
    print("    two PNGs and purity numbers -- lexical vs semantic clustering, visible")
    print("  * query with a paraphrase that shares NO words with its topic")
    print("    (e.g. 'kicking a ball into a net') and see which mode still finds Sports")
    print("  * bump --n to 2000 and watch cluster boundaries sharpen")


if __name__ == "__main__":
    main()
