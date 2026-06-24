from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from amfd.core.config import DiagnosisConfig
from amfd.core.models import FeatureVector, RetrievedEvidence

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

DEFAULT_KNOWLEDGE_BASE = {
    "cwru_bearing_notes": (
        "CWRU bearing experiments include normal, inner race, outer race, and ball faults "
        "under varied load and speed. Bearing defects create impulsive vibration and spectral "
        "energy in the fault-band neighborhood."
    ),
    "pu_bearing_notes": (
        "Paderborn bearing data includes vibration, motor current, speed, torque, radial load, "
        "and temperature for healthy and damaged bearings."
    ),
    "maintenance_sop": (
        "For high vibration, reduce load, inspect bearing lubrication and race damage, check "
        "coupling alignment, and schedule controlled shutdown when critical."
    ),
    "control_loop_sop": (
        "RPM drift with vibration can indicate drive instability, load fluctuation, or "
        "control-loop calibration issues."
    ),
}


class HybridMaintenanceRetriever:
    """Hybrid lexical + rerank retriever over the project knowledge base."""

    def __init__(self, documents: dict[str, str] | None = None) -> None:
        self.documents = documents or DEFAULT_KNOWLEDGE_BASE
        self._tokenized = {key: self._tokens(text) for key, text in self.documents.items()}
        total_tokens = sum(len(tokens) for tokens in self._tokenized.values())
        self._avg_len = total_tokens / max(1, len(self._tokenized))

    @classmethod
    def load(cls, path: str | Path | None = None) -> HybridMaintenanceRetriever:
        if path is None:
            return cls()
        kb_path = Path(path)
        if not kb_path.exists():
            return cls()
        try:
            raw = json.loads(kb_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return cls()
        documents = {
            str(key): str(value)
            for key, value in raw.items()
            if isinstance(key, str) and isinstance(value, str) and value.strip()
        }
        return cls(documents or None)

    @classmethod
    def from_config(cls, config: DiagnosisConfig) -> HybridMaintenanceRetriever:
        return cls.load(config.knowledge_base_path)

    def retrieve(
        self,
        features: FeatureVector,
        limit: int = 3,
        focus: str | None = None,
    ) -> list[RetrievedEvidence]:
        query = self._query_from_features(features, focus=focus)
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
    def _query_from_features(features: FeatureVector, focus: str | None = None) -> str:
        terms = ["vibration", "fault", "maintenance"]
        if focus:
            terms.extend(focus.replace("_", " ").split())
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
