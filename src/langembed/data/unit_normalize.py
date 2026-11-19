"""Normalized phrase-level units for the per-article semantic graph (v2, 2026-10-11).

Colleague feedback on the v1 graphs (one node per raw unit occurrence) was that clusters
looked dirty: the same word repeated as many nodes ("the sentence" / "a sentence", every
"XTREME" mention), stop words, copulas and tiny abbreviations as nodes, researcher surnames
from citations forming their own groups, and a large cluster of generic academic vocabulary.
This module fixes that before embedding:

- the reference list / acknowledgements are cut off and inline citations stripped
  (`clean_article_text`, `strip_citations`);
- units are chunked as before (idioms, noun phrases, verb groups, leftover content words),
  then trimmed of leading/trailing determiners, pronouns, prepositions, numbers and stop
  words, and person names (spaCy NER) are removed;
- units made only of stop words, copulas/light verbs, generic academic words or short
  abbreviations are dropped;
- each surviving unit gets a lemma key (lower-cased, singular, hyphen-insensitive), and all
  occurrences with the same key are merged into one node carrying a frequency count, its
  surface variants and a few context sentences (`merge_units`).
"""

from __future__ import annotations

import functools
import re
from collections import Counter, defaultdict

from langembed.data.semantic_units import IDIOMS

# Back-matter headings. A reference list is cut wherever it starts (in conference papers it
# sits mid-file, before the appendices); the weaker headings only in the last 40% of the text.
_REFERENCES_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?(references|bibliography|works cited|literature cited)\s*:?\s*$",
    re.IGNORECASE,
)
_LATE_BACK_MATTER_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?(acknowledg(?:e)?ments?|notes|literature|"
    r"(?:suggestions for )?further reading)\s*:?\s*$",
    re.IGNORECASE,
)
_FRONT_END_RE = re.compile(r"^\s*(?:\d+\.?\s*)?(abstract|introduction)\s*:?\s*$", re.IGNORECASE)
# Appendix after the references: content again ("Appendix", "Appendix A", "A.1 Setup").
_APPENDIX_RE = re.compile(
    r"^\s*(appendix|appendices)\b.{0,60}$|^\s*[A-H]\.\d+\s+[A-Z]", re.IGNORECASE
)
# "(Smith 1990)", "(Smith & Jones, 2003a; Brown et al. 2010: 12)", "(see Dowty 1979)"
_PAREN_CITATION_RE = re.compile(r"\((?=[^()]*\b(?:1[6-9]|20)\d{2}[a-z]?\b)[^()]{0,200}\)")
# "[12]", "[3, 4]", "[5-7]"
_BRACKET_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*[,;–\-]\s*\d+)*\s*\]")

# Light / copular verbs: a verb unit headed by one of these says nothing on its own.
LIGHT_VERBS = frozenset(
    """
    be have do get make take give go come seem become remain say tell use show see find let
    consider note call need want know think mean include follow provide involve appear
    refer argue claim assume suggest propose point discuss illustrate observe compare
    describe present mention try put keep turn begin start look set allow lead help
    """.split()
)

# Generic academic / discourse words: dropped when a unit consists only of them
# (kept inside longer phrases, e.g. "lifetime effect" stays whole).
GENERIC_WORDS = frozenset(
    """
    example case way fact point kind type sort form number part set term sense view account
    approach result paper section chapter work study question issue reason respect thing
    other following above below first second third last same different certain various
    possible general specific particular such many much more most several further also
    however thus hence therefore well way time one two three use usage instance notion
    idea aspect matter rest lot level role basis context situation need sake order end
    article author analysis discussion literature table figure appendix footnote note
    page line item version datum data hypothesis claim argument evidence problem
    respectively etc i.e. e.g. cf. ibid. viz. vs. al. et par avg vol pp ch
    """.split()
)

# Interlinear-gloss abbreviations (Leipzig conventions) that example glosses leave behind.
GLOSS_ABBREVIATIONS = frozenset(
    """
    erg abs acc nom dat gen loc abl ins instr com voc rel dem ptcpl ptcp att sg pl du pst prs
    fut neg def indef comp ind dir obj subj aux cop det poss refl recp caus pass irr real perf
    impf ipfv pfv prog hab top foc evid q emph nmlz nmlzr cl clf lnk mid ant obl
    """.split()
)

_PERSON_NUMBER_RE = re.compile(r"^(?:cl)?\d*(?:sg|pl|du|subj|obj|past)?\d*$")

_EDGE_DROP_POS = {
    "DET",
    "PRON",
    "ADP",
    "CCONJ",
    "SCONJ",
    "PART",
    "PUNCT",
    "SYM",
    "NUM",
    "SPACE",
    "INTJ",
    "X",
}
_CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


_YEAR = r"(?:1[6-9]|20)\d{2}[a-z]?"
# "McCloskey (2002)", "Smith et al. (2020)", "(Dowty 1979; Smith and Jones, 2003)"
_NARRATIVE_CITE_RE = re.compile(r"\b([A-Z][A-Za-zÀ-ſ'\-]+)(?:\s+et al\.?)?\s*\(\s*" + _YEAR)
_PAREN_CITE_NAME_RE = re.compile(r"\b([A-Z][A-Za-zÀ-ſ'\-]+)(?:\s+et al\.?)?,?\s+" + _YEAR + r"\b")
# reference-list author lists: "Lepikhin, Melvin Johnson," / "Smith, J." / "Asai, Kazuma"
_REF_AUTHOR_RE = re.compile(r"\b([A-Z][A-Za-zÀ-ſ'\-]+),\s+(?=[A-Z])")


def cited_names(lines: list[str]) -> set[str]:
    """Surnames (and given names in reference author lists) of cited researchers, used to drop
    units that are nothing but a cited name -- spaCy's small NER misses many of them."""
    names: set[str] = set()
    text = "\n".join(lines)
    names.update(_NARRATIVE_CITE_RE.findall(text))
    names.update(_PAREN_CITE_NAME_RE.findall(text))
    start = next((i for i, line in enumerate(lines) if _REFERENCES_RE.match(line)), None)
    if start is not None:
        end = next(
            (j for j in range(start + 1, len(lines)) if _APPENDIX_RE.match(lines[j])), len(lines)
        )
        for line in lines[start + 1 : end]:
            names.update(_REF_AUTHOR_RE.findall(line))
            m = re.match(r"^([A-Z][A-Za-zÀ-ſ'\-]+),", line)  # line-initial "Surname,"
            if m:
                names.add(m.group(1))
    return {n for n in names if len(n) > 2 and n.lower() not in GENERIC_WORDS}


def person_names(sentences: list[str], model_name: str = "en_core_web_sm") -> set[str]:
    """Tokens of every PERSON entity found anywhere in the article: the small model tags a name
    in some sentences and misses it in others, so the article-wide set is applied everywhere."""
    nlp = _nlp(model_name)
    names: set[str] = set()
    for doc in nlp.pipe(sentences):  # type: ignore[attr-defined]
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                names.update(t.text for t in ent if t.text[:1].isupper() and len(t.text) > 1)
    return names


def is_foreign_line(
    line: str,
    lexicon: frozenset[str] | set[str],
    min_words: int = 4,
    min_english_ratio: float = 0.5,
) -> bool:
    """A line of a cited example in another language (or its morpheme gloss): enough
    lower-case words and most of them not English."""
    words = [w.lower() for w in re.findall(r"[^\W\d_]{3,}", line) if not w[:1].isupper()]
    if not lexicon or len(words) < min_words:
        return False
    return sum(1 for w in words if w in lexicon) / len(words) < min_english_ratio


def clean_article_text(lines: list[str]) -> list[str]:
    """Remove front matter (title, authors, affiliations, e-mails: everything before an
    Abstract/Introduction heading in the first 10% of the lines) and back matter: from the
    first back-matter heading (references anywhere;
    acknowledgements / notes / further reading only in the last 40% of the text) up to an
    appendix heading that follows it, or to the end. Author lists in a reference section
    otherwise turn into clusters of unrelated surnames."""
    # front matter (title, authors, affiliations, e-mails) up to an early Abstract/Introduction
    front = next(
        (
            i
            for i, line in enumerate(lines[: max(len(lines) // 10, 1)])
            if _FRONT_END_RE.match(line)
        ),
        None,
    )
    if front is not None:
        lines = lines[front + 1 :]
    n = len(lines)
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if _REFERENCES_RE.match(line) or (i >= n * 0.6 and _LATE_BACK_MATTER_RE.match(line))
        ),
        None,
    )
    if start is None:
        return lines
    end = next((j for j in range(start + 1, n) if _APPENDIX_RE.match(lines[j])), n)
    return lines[:start] + lines[end:]


def strip_citations(sentence: str) -> str:
    s = _PAREN_CITATION_RE.sub("", sentence)
    s = _BRACKET_CITATION_RE.sub("", s)
    return re.sub(r"\s{2,}", " ", s).replace(" ,", ",").replace(" .", ".").strip()


@functools.lru_cache(maxsize=2)
def _nlp(model_name: str) -> object:
    import spacy

    return spacy.load(model_name)  # full pipeline: parser (chunks), NER (persons), lemmatizer


@functools.lru_cache(maxsize=2)
def _idiom_matcher(model_name: str) -> object:
    from spacy.matcher import PhraseMatcher

    nlp = _nlp(model_name)
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")  # type: ignore[attr-defined]
    matcher.add("IDIOM", [nlp.make_doc(idiom) for idiom in IDIOMS])  # type: ignore[attr-defined]
    return matcher


def _raw_spans(doc, matcher) -> list[tuple[int, int, str]]:
    """Same chunk priority as semantic_units.spacy_units: idioms, noun chunks, verb groups,
    leftover content tokens. Returns (start, end, kind) token spans."""
    import spacy.util

    covered = [False] * len(doc)
    spans: list[tuple[int, int, str]] = []

    def take(start: int, end: int, kind: str) -> None:
        spans.append((start, end, kind))
        for t in range(start, end):
            covered[t] = True

    for span in spacy.util.filter_spans([doc[s:e] for _, s, e in matcher(doc)]):
        take(span.start, span.end, "idiom")
    for span in spacy.util.filter_spans(
        [nc for nc in doc.noun_chunks if not any(covered[t] for t in range(nc.start, nc.end))]
    ):
        take(span.start, span.end, "np")
    i = 0
    while i < len(doc):
        if not covered[i] and doc[i].pos_ in ("VERB", "AUX"):
            j = i + 1
            while j < len(doc) and not covered[j] and doc[j].dep_ in ("prt", "aux", "neg"):
                j += 1
            take(i, j, "vp")
            i = j
            continue
        i += 1
    for t, tok in enumerate(doc):
        if not covered[t] and tok.pos_ in _CONTENT_POS and not tok.is_stop:
            take(t, t + 1, "word")
    spans.sort()
    return spans


def _is_short_abbrev(tok) -> bool:
    text = tok.text.strip(".")
    if "." in text and len(text.replace(".", "")) <= 4:  # "p.c", "i.e"
        return True
    if tok.text.endswith(".") and len(text) <= 4:  # "e.g.", "cf.", "pp.", "ex."
        return True
    if len(text) <= 2 and not (text.isupper() and len(text) == 2):  # keep NP, VP, MT, ...
        return True
    return False


_SECTION_REF_RE = re.compile(
    r"^[A-Z]\d+[a-z]?$|^(?:[A-Z]\.)?\d+[a-z]?(?:\.\d+)*\.?$|^(?=[ivx])x{0,3}(?:ix|iv|v?i{0,3})\.?$|^[A-Z]\.\d+"
)


def _token_ok(tok) -> bool:
    if tok.is_punct or tok.is_space or tok.like_num or tok.like_url or tok.like_email:
        return False
    if _SECTION_REF_RE.match(tok.text):  # "A.3", "17b", "iii" (sections, examples, numbering)
        return False
    if tok.lower_ in GLOSS_ABBREVIATIONS or tok.lower_ in _stop_words():
        return False
    if re.search(r"[-()/\d]", tok.text):  # only compound tokens: a plain "past" stays
        parts = [p for p in re.split(r"[-.()/]", tok.lower_) if p]
        if parts and all(p in GLOSS_ABBREVIATIONS or _PERSON_NUMBER_RE.match(p) for p in parts):
            return False  # morpheme glosses: "fut-1sg", "rel-2sg", "cl1(subj)-past"
    if re.search(
        r"[^\x00-\x7f\u2019\u2013\u2014]", tok.text
    ):  # transcribed example words with IPA / diacritics
        return False
    if tok.ent_type_ == "PERSON":
        return False
    if not re.search(r"[A-Za-z]", tok.text):
        return False
    return not _is_short_abbrev(tok)


@functools.lru_cache(maxsize=1)
def _stop_words() -> frozenset[str]:
    from spacy.lang.en.stop_words import STOP_WORDS

    return frozenset(STOP_WORDS)  # checked on the lower-cased text: "The", "There" too


def _norm_lemma(tok) -> str:
    lemma = tok.lemma_ if tok.lemma_ and tok.lemma_ != "-PRON-" else tok.text
    return lemma.lower().replace("-", " ").strip()


def english_ratio(doc, lexicon: frozenset[str] | set[str]) -> float:
    words = [t for t in doc if t.is_alpha and len(t.text) > 2 and not t.text[:1].isupper()]
    if not words:
        return 1.0
    return sum(1 for t in words if t.lower_ in lexicon or t.lemma_.lower() in lexicon) / len(words)


def normalized_units(
    sentence: str,
    model_name: str = "en_core_web_sm",
    drop_names: frozenset[str] | set[str] = frozenset(),
    lexicon: frozenset[str] | set[str] = frozenset(),
    min_english_ratio: float = 0.6,
) -> list[tuple[str, str]]:
    """(key, surface) pairs for the meaningful units of one sentence. `drop_names`: cited
    researchers' names (see cited_names); a unit made only of them is dropped. `lexicon`: an
    English word list; in a mostly non-English sentence (a glossed example from another
    language) only units made of English words are kept, so example-language words don't
    become nodes, while the article's own jargon in English prose is unaffected."""
    nlp = _nlp(model_name)
    doc = nlp(sentence)  # type: ignore[operator]
    foreign = bool(lexicon) and english_ratio(doc, lexicon) < min_english_ratio
    out: list[tuple[str, str]] = []
    for start, end, kind in _raw_spans(doc, _idiom_matcher(model_name)):
        if kind == "idiom":
            continue  # idioms ("in order to", "as well as") are function phrases, not topics
        toks = [t for t in doc[start:end] if _token_ok(t)]
        # trim function material off both edges ("the", "his", "of", "also", ...)
        while toks and (toks[0].pos_ in _EDGE_DROP_POS or toks[0].is_stop or toks[0].pos_ == "AUX"):
            toks.pop(0)
        while toks and (
            toks[-1].pos_ in _EDGE_DROP_POS or toks[-1].is_stop or toks[-1].pos_ == "AUX"
        ):
            toks.pop()
        if not toks:
            continue
        content = [t for t in toks if t.pos_ in _CONTENT_POS and not t.is_stop]
        if not content:
            continue
        if all(t.text in drop_names for t in content):
            continue
        # a capitalised ordinary word opening the sentence ("Suddenly", "Clearly") is
        # mis-tagged as a proper noun; its mid-sentence uses still count
        if (
            len(toks) == 1
            and (toks[0].i == 0 or toks[0].is_sent_start)
            and toks[0].text[:1].isupper()
            and not toks[0].text.isupper()
            and toks[0].lower_ in lexicon
        ):
            continue
        if foreign and not all(t.lower_ in lexicon or t.lemma_.lower() in lexicon for t in content):
            continue
        # a lone verb / participle / adverb ("resembles", "according", "rather") is a relation
        # or a discourse marker, not a topic; verbs stay inside longer phrases
        if len(content) == 1 and len(toks) == 1 and content[0].pos_ in ("VERB", "ADV", "AUX"):
            continue
        lemmas = [_norm_lemma(t) for t in toks]
        content_lemmas = [_norm_lemma(t) for t in content]
        if kind == "vp" or all(t.pos_ in ("VERB", "AUX") for t in content):
            if all(lem in LIGHT_VERBS for lem in content_lemmas):
                continue
        if all(lem in GENERIC_WORDS or lem in LIGHT_VERBS for lem in content_lemmas):
            continue
        key = re.sub(r"\s+", " ", " ".join(lemmas)).strip()
        if len(key.replace(" ", "")) < 3:
            continue
        surface = doc[toks[0].i : toks[-1].i + 1].text
        if not all(t.pos_ == "PROPN" or (t.text.isupper() and len(t.text) > 1) for t in toks):
            surface = " ".join(
                t.text
                if (t.pos_ == "PROPN" or (t.text.isupper() and len(t.text) > 1))
                else t.text.lower()
                for t in doc[toks[0].i : toks[-1].i + 1]
            )
            surface = re.sub(r"\s+([,.;:)])", r"\1", surface).replace("( ", "(").replace(" - ", "-")
        out.append((key, surface))
    return out


def merge_units(records: list[dict], max_contexts: int = 3, max_variants: int = 6) -> list[dict]:
    """records: {article_id, sent_idx, key, surface, parent_sentence} occurrences.
    Returns one record per (article_id, key) with count, variants, contexts and the most
    frequent surface form as `text` (the string that is embedded and displayed)."""
    # keys sharing a surface form are the same unit split by the lemmatizer
    # ("living" as noun vs. verb lemma "live"): union them before grouping
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(x: tuple[str, str]) -> tuple[str, str]:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_surface: dict[tuple[str, str], tuple[str, str]] = {}
    for r in records:
        k = (r["article_id"], r["key"])
        s = (r["article_id"], r["surface"].lower())
        if s in by_surface:
            a, b = find(k), find(by_surface[s])
            if a != b:
                parent[max(a, b)] = min(a, b)
        else:
            by_surface[s] = k
            find(k)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[find((r["article_id"], r["key"]))].append(r)
    merged = []
    for (article_id, key), occ in groups.items():
        surfaces = Counter(o["surface"] for o in occ)
        contexts, seen = [], set()
        for o in occ:
            if o["sent_idx"] not in seen and len(contexts) < max_contexts:
                seen.add(o["sent_idx"])
                contexts.append({"sent_idx": o["sent_idx"], "text": o["parent_sentence"]})
        merged.append(
            {
                "article_id": article_id,
                "key": key,
                "text": surfaces.most_common(1)[0][0],
                "count": len(occ),
                "n_sentences": len({o["sent_idx"] for o in occ}),
                "sent_ids": sorted({o["sent_idx"] for o in occ}),
                "first_sent_idx": min(o["sent_idx"] for o in occ),
                "variants": [s for s, _ in surfaces.most_common(max_variants)],
                "contexts": contexts,
            }
        )
    merged.sort(key=lambda m: (m["article_id"], m["first_sent_idx"], m["key"]))
    return merged
