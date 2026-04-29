from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from typing import Iterable

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_ACTIVE_BACKEND = "hash"


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}



def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())



def _hash_to_index_sign(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    value = int(digest[:16], 16)
    index = value % dim
    sign = 1.0 if ((value >> 1) & 1) else -1.0
    return index, sign



def _hash_text_embedding(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    tokens = _tokenize(text)
    for token in tokens:
        i, s = _hash_to_index_sign(f"tok::{token}", dim)
        vec[i] += s

    compact = " ".join(tokens)
    for i in range(max(0, len(compact) - 2)):
        tri = compact[i : i + 3]
        j, s = _hash_to_index_sign(f"tri::{tri}", dim)
        vec[j] += 0.3 * s

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _sentence_transformer_embedding(text: str) -> list[float] | None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from transformers import logging as hf_logging  # type: ignore

        hf_logging.set_verbosity_error()
    except Exception:
        pass

    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None

    model_name = os.getenv("CREATIVEAI_SENTENCE_MODEL", "all-MiniLM-L6-v2")
    device = os.getenv("CREATIVEAI_SENTENCE_DEVICE", "").strip()
    if not device:
        try:
            import torch  # type: ignore

            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"

    cache_key = f"_creativeai_st_model::{model_name}::{device}"
    model = getattr(_sentence_transformer_embedding, cache_key, None)
    if model is None:
        model = SentenceTransformer(model_name, device=device)
        setattr(_sentence_transformer_embedding, cache_key, model)

    vec = model.encode([text], normalize_embeddings=True)[0]
    return [float(x) for x in vec]


def text_embedding(text: str, dim: int = 256, backend: str | None = None) -> list[float]:
    global _ACTIVE_BACKEND
    preferred = (backend or os.getenv("CREATIVEAI_EMBEDDING_BACKEND", "sentence_transformer")).strip().lower()
    require_semantic = _truthy(os.getenv("CREATIVEAI_REQUIRE_SEMANTIC", "false"))
    if preferred in {"sentence_transformer", "sentence-transformer", "semantic"}:
        vec = _sentence_transformer_embedding(text)
        if vec is not None:
            _ACTIVE_BACKEND = "sentence_transformer"
            return _normalize(vec)
        if require_semantic:
            raise RuntimeError(
                "Semantic embeddings required but sentence-transformers backend is unavailable. "
                "Install sentence-transformers and torch, or set CREATIVEAI_REQUIRE_SEMANTIC=false."
            )

    _ACTIVE_BACKEND = "hash"
    return _hash_text_embedding(text, dim=dim)


def active_embedding_backend() -> str:
    return _ACTIVE_BACKEND



def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        raise ValueError("Embedding vectors must be non-empty and same length")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)



def mean_pairwise_cosine_distance(texts: Iterable[str], dim: int = 256) -> float:
    seq = list(texts)
    if len(seq) < 2:
        return 0.0
    embeds = [text_embedding(t, dim=dim) for t in seq]
    total = 0.0
    count = 0
    for i in range(len(embeds)):
        for j in range(i + 1, len(embeds)):
            total += 1.0 - cosine_similarity(embeds[i], embeds[j])
            count += 1
    return total / max(count, 1)



def centroid(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []
    dim = len(embeddings[0])
    out = [0.0] * dim
    for emb in embeddings:
        for i, value in enumerate(emb):
            out[i] += value
    scale = 1.0 / len(embeddings)
    out = [v * scale for v in out]
    norm = math.sqrt(sum(v * v for v in out))
    if norm == 0:
        return out
    return [v / norm for v in out]



def nearest_neighbor_similarity(texts: list[str], dim: int = 256) -> float:
    if len(texts) < 2:
        return 0.0
    embeds = [text_embedding(t, dim=dim) for t in texts]
    sims: list[float] = []
    for i, emb in enumerate(embeds):
        best = -1.0
        for j, other in enumerate(embeds):
            if i == j:
                continue
            sim = cosine_similarity(emb, other)
            if sim > best:
                best = sim
        sims.append(best)
    return sum(sims) / len(sims)



def compactness_similarity(texts: list[str], dim: int = 256) -> float:
    if not texts:
        return 0.0
    embeds = [text_embedding(t, dim=dim) for t in texts]
    center = centroid(embeds)
    if not center:
        return 0.0
    sims = [cosine_similarity(emb, center) for emb in embeds]
    return sum(sims) / len(sims)
