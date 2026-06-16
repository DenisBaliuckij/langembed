"""Phase 7: embedding inference service (shares preprocess.normalize with training)."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from langembed.preprocess import normalize

app = FastAPI(title="langembed embed")
_model: Any = None


def _get_model() -> Any:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("artifacts/embed_gu_v1")
    return _model


class EmbedIn(BaseModel):
    texts: list[str]


@app.post("/embed")
def embed(payload: EmbedIn) -> dict[str, Any]:
    model = _get_model()
    vecs = model.encode([normalize(t) for t in payload.texts], normalize_embeddings=True)
    return {"embeddings": vecs.tolist(), "dim": int(vecs.shape[1])}
