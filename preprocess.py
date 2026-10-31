"""
Preprocess IN-Abs / IN-Ext judgments: light cleaning + legal-aware sentence splitting,
with retention statistics so you can prove little data was lost.

Usage (adjust paths to your folder layout):
  python preprocess.py --judgements dataset/IN-Abs/train-data/judgement \
                       --summaries  dataset/IN-Abs/train-data/summary \
                       --out processed/train
Summaries are optional (omit --summaries for raw-judgement-only runs).
"""
import argparse, json, re, unicodedata
from pathlib import Path

MIN_WORDS = 4  # sentences shorter than this are treated as noise (page numbers, stray headings)
RULING_RE = re.compile(r"\b(allowed|dismissed|rejected|affirmed|quashed|set aside|disposed|remanded)\b", re.I)


def is_noise(s: str) -> bool:
    if len(re.findall(r"[A-Za-z]{3,}", s)) == 0:      # no real words: ".", ". \"", "(2)", "1950."
        return True
    if len(s.split()) < MIN_WORDS and not RULING_RE.search(s):   # keep short rulings like "Appeal allowed."
        return True
    return False

ABBREVS = [
    "v", "vs", "no", "nos", "mr", "mrs", "ms", "dr", "sri", "smt", "shri", "m/s", "ltd", "pvt", "co",
    "corp", "inc", "sec", "s", "ss", "art", "arts", "cl", "sub", "para", "paras", "rs", "re", "ors",
    "anr", "etc", "viz", "ie", "eg", "cf", "ex", "dt", "vol", "ed", "p", "pp", "j", "cj", "jj", "hon",
    "st", "mt", "dept", "govt", "addl", "regd", "approx", "u/s", "w.p", "crl", "civ", "app", "misc",
    # law-report / court abbreviations
    "all", "cal", "bom", "mad", "pat", "lah", "ker", "del", "guj", "raj", "ori", "nag", "pun", "cri",
    "cr", "lj", "scr", "scc", "air", "mlj", "ilr", "itr", "tax", "cas", "rep", "supp", "sup", "ct", "act",
]
ABBR_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in ABBREVS) + r")\.", re.IGNORECASE)
INITIALS_RE = re.compile(r"\b([A-Z])\.")  # single-capital initials: "A. K. Sharma", "S.C.C."
DOT = "\u2024"  # placeholder for protected dots


def clean_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\u00ad", "")                      # soft hyphens
    t = re.sub(r"-\n(?=[a-z])", "", t)               # hyphenated line breaks
    t = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", t)        # lines that are only a page/para number
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n\n", t)
    # join hard-wrapped lines inside a paragraph, keep paragraph breaks
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def split_sentences(text: str):
    text = ABBR_RE.sub(lambda m: m.group(1) + DOT, text)
    text = INITIALS_RE.sub(lambda m: m.group(1) + DOT, text)
    text = re.sub(r"(\d)\.(\d)", r"\1" + DOT + r"\2", text)          # 12.5, 2.1
    parts = re.split(r"(?<=[.?!])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])", text)
    return [p.replace(DOT, ".").strip() for p in parts if p.strip()]


def process(text: str):
    raw_sents = split_sentences(clean_text(text))
    kept, prev = [], None
    for s in raw_sents:
        if is_noise(s):
            continue
        if s == prev:                                   # only consecutive exact duplicates
            continue
        kept.append(s)
        prev = s
    return raw_sents, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgements", required=True)
    ap.add_argument("--summaries")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jdir = Path(a.judgements)
    sdir = Path(a.summaries) if a.summaries else None
    files = sorted(p for p in jdir.iterdir() if p.is_file())
 
    stats = dict(docs_in=0, docs_out=0, sents_before=0, sents_after=0,
                 words_before=0, words_after=0, long_sents_over_150w=0, dropped_docs=[])
    with open(out / "data.jsonl", "w", encoding="utf-8") as f:
        for p in files:
            stats["docs_in"] += 1
            raw, kept = process(p.read_text(encoding="utf-8", errors="ignore"))
            stats["sents_before"] += len(raw)
            stats["words_before"] += sum(len(s.split()) for s in raw)
            rec = {"id": p.stem, "sentences": kept}
            if sdir:
                sp = sdir / p.name
                if not sp.exists():
                    stats["dropped_docs"].append((p.name, "no summary file")); continue
                _, ksum = process(sp.read_text(encoding="utf-8", errors="ignore"))
                rec["summary_sentences"] = ksum
                if not ksum:
                    stats["dropped_docs"].append((p.name, "empty summary")); continue
            if not kept:
                stats["dropped_docs"].append((p.name, "empty after cleaning")); continue
            stats["docs_out"] += 1
            stats["sents_after"] += len(kept)
            stats["words_after"] += sum(len(s.split()) for s in kept)
            stats["long_sents_over_150w"] += sum(len(s.split()) > 150 for s in kept)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pct = lambda x, y: round(100 * x / y, 2) if y else 0
    stats["doc_retention_%"] = pct(stats["docs_out"], stats["docs_in"])
    stats["sentence_retention_%"] = pct(stats["sents_after"], stats["sents_before"])
    stats["word_retention_%"] = pct(stats["words_after"], stats["words_before"])
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()