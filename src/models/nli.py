"""
Natural Language Inference wrapper, used by the Contradiction Agent to
compare pairs of evidence statements and by the Verification Agent to check
whether a claim is actually entailed by its cited evidence.
"""

from typing import Dict

from src.config import models


class NLIModel:
    def __init__(self, model_name: str = models.nli_model):
        self.model_name = model_name
        self._pipeline = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import pipeline

        self._pipeline = pipeline("text-classification", model=self.model_name, top_k=None)

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Return a dict like {"ENTAILMENT": 0.1, "NEUTRAL": 0.2, "CONTRADICTION": 0.7}.
        Cross-encoder NLI models expect "premise [SEP] hypothesis"-style
        input; the sentence-transformers-trained checkpoints used here
        accept a single string with the pair joined, which the pipeline
        text-classification head handles when given `text_pair`.
        """
        if self._pipeline is None:
            self.load()

        raw = self._pipeline({"text": premise, "text_pair": hypothesis})
        # `top_k=None` on a text-classification pipeline returns a list of
        # {"label": ..., "score": ...} dicts for every class.
        scores = {item["label"].upper(): float(item["score"]) for item in raw}
        return scores

    def label_for(self, premise: str, hypothesis: str) -> str:
        scores = self.predict(premise, hypothesis)
        return max(scores, key=scores.get) if scores else "NEUTRAL"


_shared_instance = None


def get_nli_model() -> NLIModel:
    global _shared_instance
    if _shared_instance is None:
        _shared_instance = NLIModel()
    return _shared_instance
