# Per-article semantic graph v2

v2 of the single-article semantic graph (2026-10-11). It reworks the v1 graphs (`scripts/build_article_graphs.py`, one node per raw phrase occurrence) after feedback from the colleagues who read them. v1 is unchanged and still used by the corpus pipeline.

## Feedback and what changed

| Feedback | v2 |
|---|---|
| No way to find a word | Search box. It matches the term, its spellings and its lemma, lists the groups that contain it, and zooms to the hits. A `#q=word` link opens the graph with that search applied. |
| Duplicate nodes ("the sentence" / "a sentence", every "XTREME") | Units are trimmed of determiners, pronouns, prepositions and numbers, then lemmatized (lower case, singular, hyphen-insensitive). Every occurrence with the same key, or the same surface form, becomes one node with a frequency count, its spellings and context sentences. |
| Stop words, copulas, small abbreviations, low-value words | These are dropped: stop words, copulas and light verbs (*be, have, show, use…*), lone verbs and adverbs, short abbreviations (*e.g., cf., p.c*), numbering (*A.3, 17b, iii*), interlinear-gloss tags (*ERG, ACC, fut-1sg*), and units made only of generic academic words (*example, case, result, approach…*). |
| Groups of unrelated surnames (XTREME + researcher names) | The cause was the reference list read as text. It is now cut off, along with front matter, acknowledgements and notes, while appendices are kept. Inline citations are stripped. Cited names and spaCy PERSON names found anywhere in the article are dropped. |
| Example-language words as nodes | Lines that are mostly non-English (glossed examples) are removed before sentence splitting. A rare lower-case word missing from the English word list is not kept as a node. |
| Too many, meaningless groups; graph should be smaller | Only units occurring at least twice become nodes, at most 400. Nodes are sized by frequency. Louvain clustering uses a fixed seed (reproducible), with at most 12 groups. |
| Loose groups | Embeddings are mean-centred per article before cosine similarity, because the project's SimCSE vectors are anisotropic (any two phrases ≈ 0.69). The colleague graphs use `sentence-transformers/all-mpnet-base-v2`, which groups short phrases much better. The model is a parameter of `run_article_graphs_v2.sh` (`MODEL=`). |
| Labels like "Bootstrapping / Single" | Each group gets a 2–5 word name from the local LLM (`label_clusters_llm.py`), based on its most frequent terms. The legend lists the top terms under it. The group's most representative phrase is kept as `phrase_label` and used as the fallback. |

## Pipeline

```
scripts/run_article_graphs_v2.sh <run-dir>     # <run-dir>/pdfs/*.pdf, optional titles.json, models/
  pdftotext (host)                       -> text/<stem>.txt
  prepare_article_units_v2.py (ml image) -> article_units_v2.jsonl   merged, normalized units
  embed_article_units_cpu.py (ml image)  -> article_unit_embeddings_v2.jsonl
  build_article_graphs_v2.py (ml image)  -> out/<stem>.json, out/article_graphs_v2.json
  label_clusters_llm.py (host, llm-router :11434)
  render_article_graph_viewer.py (host)  -> html/semantic_graph_<stem>.{html,json}
```

Code: `src/langembed/data/unit_normalize.py` (cleaning, normalization, merging) and `src/langembed/data/article_graph_v2.py` (selection, edges, clustering, labels). The viewer is `templates/article_graph_viewer.html`.

## Result on the colleague articles

| Article | v1 nodes / groups | v2 nodes / groups |
|---|---|---|
| Andrews, *Relative clauses* | 6204 / 106 | 239 / 9 |
| Mittwoch 2008, *Tenses for the living and the dead* | 4915 / 90 | 218 / 9 |
| mReFinED 2023 | 3215 / 65 | 127 / 8 |
