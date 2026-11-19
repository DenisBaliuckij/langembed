"""Split a sentence into short meaningful phrase-level semantic units --
noun phrases, verb phrases, and fixed idioms/collocations -- instead of
whole-sentence granularity. Used for the per-article semantic graph and its
unit-level embeddings (see semantic_graph/ tooling and its Airflow DAG).

Two interchangeable methods, selected via `method`:
- "spacy" (default): real syntactic chunking via spaCy's parser + a small
  built-in idiom list. Higher-quality phrase boundaries.
- "regex": a lightweight content-word-run heuristic with no syntactic
  awareness. Cruder boundaries, kept for later use per explicit decision to
  build both now and switch later.
"""

from __future__ import annotations

import functools
import re

# Small built-in list of common fixed English idioms/collocations, matched
# greedily before syntactic chunking so they survive as single units instead
# of being split across noun/verb-phrase boundaries.
IDIOMS: list[str] = [
    "state of the art",
    "as well as",
    "in order to",
    "on the other hand",
    "due to the fact that",
    "as a result",
    "in terms of",
    "with respect to",
    "a wide range of",
    "on the basis of",
    "at the same time",
    "in addition to",
    "as a whole",
    "for the sake of",
    "by means of",
    "in spite of",
    "rather than",
    "so that",
    "such as",
    "as long as",
    "as soon as",
    "kick the bucket",
    "break the ice",
    "under the weather",
]

# Content POS tags: a leftover token in one of these categories is meaningful
# enough to stand alone as a unit even outside any phrase/idiom span.
_CONTENT_POS = {"VERB", "NOUN", "PROPN", "ADJ", "ADV", "NUM"}

_STOPWORDS = frozenset(
    """
    the a an and or of to in on for with is are was were be been being this that
    these those as by at from into over under between among it its it's we our
    us they their them he she his her you your i not no but if then than so such
    can could will would should may might must do does did done have has had
    which who whom what when where why how all each both more most other some
    """.split()
)


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)


def regex_units(text: str) -> list[str]:
    """Split `text` into runs of consecutive non-stopword words, dropping
    stopword/punctuation runs entirely. No syntactic awareness -- a cruder
    fallback kept for later use, not the current default."""
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^\sA-Za-z]+", text)
    units: list[str] = []
    current: list[str] = []
    for tok in words:
        is_word = bool(re.match(r"^[A-Za-z]", tok))
        if is_word and tok.lower() not in _STOPWORDS:
            current.append(tok)
        else:
            if current:
                units.append(" ".join(current))
                current = []
    if current:
        units.append(" ".join(current))
    return units


@functools.lru_cache(maxsize=2)
def _spacy_chunk_pipeline(model_name: str) -> object:
    """Load a spaCy pipeline with the parser+tagger enabled (needed for
    noun_chunks/POS), unlike langembed.preprocess's lemmatization-only
    pipeline which excludes the parser for speed."""
    import spacy

    return spacy.load(model_name, exclude=["ner", "lemmatizer"])


@functools.lru_cache(maxsize=2)
def _idiom_matcher(model_name: str) -> object:
    from spacy.matcher import PhraseMatcher

    nlp = _spacy_chunk_pipeline(model_name)
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")  # type: ignore[attr-defined]
    matcher.add("IDIOM", [nlp.make_doc(idiom) for idiom in IDIOMS])  # type: ignore[attr-defined]
    return matcher


def spacy_units(text: str, model_name: str = "en_core_web_sm") -> list[str]:
    """Chunk `text` into idiom / noun-phrase / verb-phrase / single-content-word
    units, in that priority order, covering every meaningful token exactly
    once. Pure function words/punctuation not absorbed into a phrase are
    dropped rather than becoming their own unit."""
    import spacy.util

    nlp = _spacy_chunk_pipeline(model_name)
    matcher = _idiom_matcher(model_name)
    doc = nlp(text)  # type: ignore[operator]

    covered = [False] * len(doc)
    spans: list[tuple[int, int, str]] = []

    idiom_spans = spacy.util.filter_spans([doc[s:e] for _, s, e in matcher(doc)])  # type: ignore[operator]
    for span in idiom_spans:
        spans.append((span.start, span.end, span.text))
        for i in range(span.start, span.end):
            covered[i] = True

    noun_spans = spacy.util.filter_spans(
        [nc for nc in doc.noun_chunks if not any(covered[i] for i in range(nc.start, nc.end))]
    )
    for span in noun_spans:
        spans.append((span.start, span.end, span.text))
        for i in range(span.start, span.end):
            covered[i] = True

    # Verb-phrase heuristic: a verb/aux plus any immediately attached
    # particle/aux/negation tokens (e.g. "figure out", "did not show").
    i = 0
    while i < len(doc):
        if covered[i]:
            i += 1
            continue
        tok = doc[i]
        if tok.pos_ in ("VERB", "AUX"):
            j = i + 1
            while j < len(doc) and not covered[j] and doc[j].dep_ in ("prt", "aux", "neg"):
                j += 1
            spans.append((i, j, doc[i:j].text))
            for k in range(i, j):
                covered[k] = True
            i = j
            continue
        i += 1

    # Leftover single content tokens; pure function words/punctuation dropped.
    for i, tok in enumerate(doc):
        if covered[i] or tok.is_space or tok.is_punct:
            continue
        if tok.is_stop and tok.pos_ not in _CONTENT_POS:
            continue
        spans.append((i, i + 1, tok.text))

    spans.sort(key=lambda s: s[0])
    return [text for (_, _, text) in spans if text.strip()]


def extract_semantic_units(
    text: str, method: str = "spacy", model_name: str = "en_core_web_sm"
) -> list[str]:
    if method == "spacy":
        return spacy_units(text, model_name=model_name)
    if method == "regex":
        return regex_units(text)
    raise ValueError(f"unknown method: {method!r}, expected 'spacy' or 'regex'")
