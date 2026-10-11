"""Stage 3.5 of the v2 per-article semantic graph (host, stdlib only): give each topic group a
short generalizing name with the local LLM (OpenAI-compatible endpoint of the llm-router).

The builder's label is the group's most representative phrase, which for a group like
"dog, clothes, kangaroo, knife" is just "Dog". Colleagues found such labels too narrow, so the
LLM is shown the article title and the group's most frequent phrases and asked for a 2-5 word
name. The representative phrase is kept as `phrase_label` and stays the label if the LLM call
fails. Rewrites every <article_id>.json and article_graphs_v2.json in --out-dir in place.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

PROMPT = """You name topic groups in a semantic graph of one scientific article.
Article: "{title}"
The group contains these terms (most frequent first): {phrases}
Other groups in this article are already named: {taken}

Give this group a short, specific name of 2-5 English words that \
generalizes what its terms have in common
(for example "Example sentence nouns", "Grammatical functions", "Evaluation metrics").
Base the name mainly on the first, most frequent terms; a few terms at the end of the list may be
loosely attached and should not decide the name.
Answer with the name only, no quotes, no explanation."""


def sentence_case(name: str, proper: set[str] = frozenset()) -> str:
    """'Performance Evaluation Metrics' -> 'Performance evaluation metrics'; acronyms,
    mixed-case names (NER, MReFinED) and words capitalized in the group's own terms
    (`proper`, e.g. Wikipedia) are kept."""
    words = name.split()
    out = [
        w
        if (i == 0 or w.isupper() or any(ch.isupper() for ch in w[1:]) or w in proper)
        else w.lower()
        for i, w in enumerate(words)
    ]
    return " ".join(out)


def ask(url: str, model: str, prompt: str, timeout: int) -> str:
    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "max_tokens": 24,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = json.load(r)["choices"][0]["message"]["content"] or ""
    text = text.strip().splitlines()[0] if text.strip() else ""
    text = re.sub(r'^[\s"\'*`]+|[\s"\'*`.]+$', "", text)
    return text if 2 <= len(text) <= 60 else ""


def label_article(article: dict, url: str, model: str, timeout: int) -> None:
    by_cluster: dict[int, list[dict]] = {}
    for n in article["nodes"]:
        by_cluster.setdefault(n["cluster"], []).append(n)
    taken: list[str] = []
    for c in article["clusters"]:
        c.setdefault("phrase_label", c["label"])
        if c["id"] < 0:
            continue
        top = sorted(by_cluster.get(c["id"], []), key=lambda n: -n["count"])[:18]
        prompt = PROMPT.format(
            title=article["title"],
            phrases=", ".join(n["text"] for n in top),
            taken=", ".join(taken) or "(none yet)",
        )
        try:
            name = ask(url, model, prompt, timeout)
        except Exception as e:  # keep the phrase label
            print(f"  LLM error for group {c['id']}: {e}")
            name = ""
        if name and name.lower() not in (t.lower() for t in taken):
            proper = {
                w for n in by_cluster.get(c["id"], []) for w in n["text"].split() if w[:1].isupper()
            }
            c["label"] = sentence_case(name[:1].upper() + name[1:], proper)
            c["label_source"] = f"llm:{model}"
        else:
            c["label"] = c["phrase_label"]
            c["label_source"] = "representative phrase"
        taken.append(c["label"])
        print(f"  [{c['size']:3d}] {c['label']}   (phrase: {c['phrase_label']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--titles", default="", help='JSON {"<article_id>": "<display title>"}')
    ap.add_argument("--url", default="http://127.0.0.1:11434/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.8:27b-mtp-128k")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    titles = json.loads(Path(args.titles).read_text(encoding="utf-8")) if args.titles else {}
    combined_path = out_dir / "article_graphs_v2.json"
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    for article in combined["articles"]:
        article["title"] = titles.get(article["article_id"], article["title"])
        print(article["title"])
        label_article(article, args.url, args.model, args.timeout)
        (out_dir / f"{article['article_id']}.json").write_text(
            json.dumps({"articles": [article]}, ensure_ascii=False), encoding="utf-8"
        )
    combined_path.write_text(json.dumps(combined, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
