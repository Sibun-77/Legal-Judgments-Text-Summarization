"""
Create 1/0 extractive labels: for every headnote (summary) sentence, find the most similar
judgment sentence (TF-IDF cosine) and mark it 1. Everything else is 0.

Input : data.jsonl from preprocess.py  (fields: id, sentences, summary_sentences)
Output: labeled.jsonl                   (adds: labels = list of 0/1, same length as sentences)

Usage: python label.py --inp processed/train/data.jsonl --out processed/train/labeled.jsonl
"""
import argparse, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

MIN_SIM = 0.15   # a summary sentence with no judgment sentence this similar is ignored


def label_doc(sents, summ):
    labels = [0] * len(sents)
    if not sents or not summ:
        return labels, 0
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    try:
        m = vec.fit_transform(sents + summ)
    except ValueError:                      # empty vocabulary
        return labels, 0
    doc_m, sum_m = m[: len(sents)], m[len(sents):]
    sim = (sum_m @ doc_m.T).toarray()       # rows: summary sentences, cols: judgment sentences
    matched = 0
    for row in sim:
        j = int(row.argmax())
        if row[j] >= MIN_SIM:
            labels[j] = 1
            matched += 1
    return labels, matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    n_docs = n_sents = n_pos = n_summ = n_matched = 0
    with open(a.inp, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as g:
        for line in f:
            r = json.loads(line)
            labels, matched = label_doc(r["sentences"], r["summary_sentences"])
            r["labels"] = labels
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_docs += 1; n_sents += len(labels); n_pos += sum(labels)
            n_summ += len(r["summary_sentences"]); n_matched += matched
    print(f"docs: {n_docs}")
    print(f"judgment sentences: {n_sents}, labelled 1: {n_pos} ({100*n_pos/max(n_sents,1):.1f}%)")
    print(f"summary sentences matched: {n_matched}/{n_summ} ({100*n_matched/max(n_summ,1):.1f}%)")


if __name__ == "__main__":
    main()