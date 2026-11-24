"""TF-IDF baseline: rank CVs by cosine similarity to the JD text.

Character n-grams are used because Thai has no spaces between words.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_scores(jd_text: str, cv_texts: list[str]) -> list[float]:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)
    m = vec.fit_transform([jd_text] + [t or " " for t in cv_texts])
    return [float(x) * 100 for x in cosine_similarity(m[0], m[1:]).ravel()]
