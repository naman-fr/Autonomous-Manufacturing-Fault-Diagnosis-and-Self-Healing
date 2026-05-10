from __future__ import annotations

import math
import re
from collections import Counter

from amfd.core.models import FeatureVector, RetrievedEvidence

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


class HybridMaintenanceRetriever:
    """BM25-style lexical retrieval with a lightweight late-interaction reranker."""

    def __init__(self, documents: dict[str, str] | None = None) -> None:
        self.documents = documents or {
            "cwru_bearing_notes": (
                "CWRU bearing experiments include normal, inner race, outer race, and ball "
                "faults under varied load and speed. Bearing defects create impulsive vibration "
                "and frequency components above the running-speed band."
            ),
            "pu_bearing_notes": (
                "Paderborn bearing data includes vibration, motor current, speed, torque, radial "
                "load, and temperature for healthy and damaged bearings."
            ),
            "maintenance_sop": (
                "For high vibration, reduce load, inspect bearing lubrication and race damage, "
                "check coupling alignment, and schedule controlled shutdown when critical."
            ),
            "control_loop_sop": (
                "RPM drift with vibration can indicate drive instability, load fluctuation, or "
                "control-loop calibration issues."
            ),
        }
        self._tokenized = {key: self._tokens(text) for key, text in self.documents.items()}
        self._avg_len = sum(len(tokens) for tokens in self._tokenized.values()) / len(
            self._tokenized
        )

    def retrieve(self, features: FeatureVector, limit: int = 3) -> list[RetrievedEvidence]:
        query = self._query_from_features(features)
        query_tokens = self._tokens(query)
        candidates: list[RetrievedEvidence] = []
        for source, text in self.documents.items():
            bm25 = self._bm25(query_tokens, self._tokenized[source])
            rerank = self._late_interaction_score(query_tokens, self._tokenized[source], text)
            candidates.append(
                RetrievedEvidence(
                    source=source,
                    text=text,
                    bm25_score=round(bm25, 4),
                    rerank_score=round(rerank, 4),
                )
            )
        ranked = sorted(
            candidates,
            key=lambda item: (item.rerank_score, item.bm25_score),
            reverse=True,
        )
        return ranked[:limit]

    def _bm25(self, query: list[str], document: list[str]) -> float:
        counts = Counter(document)
        score = 0.0
        k1 = 1.5
        b = 0.75
        for token in query:
            df = sum(1 for tokens in self._tokenized.values() if token in tokens)
            idf = math.log(1 + (len(self._tokenized) - df + 0.5) / (df + 0.5))
            tf = counts[token]
            denom = tf + k1 * (1 - b + b * len(document) / self._avg_len)
            score += idf * (tf * (k1 + 1) / denom) if denom else 0.0
        return score

    @staticmethod
    def _late_interaction_score(query: list[str], document: list[str], text: str) -> float:
        overlap = len(set(query) & set(document)) / max(1, len(set(query)))
        phrase_boost = 0.15 if "bearing" in text.lower() and "vibration" in text.lower() else 0.0
        return min(1.0, overlap + phrase_boost)

    @staticmethod
    def _query_from_features(features: FeatureVector) -> str:
        terms = ["vibration", "fault", "maintenance"]
        if features.crest_factor >= 3:
            terms.extend(["bearing", "impulsive", "race"])
        if features.rpm_mean and features.rpm_mean < 1750:
            terms.extend(["rpm", "drive", "control"])
        if features.dominant_frequency_hz < 120:
            terms.extend(["imbalance", "running", "speed"])
        return " ".join(terms)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text)]
