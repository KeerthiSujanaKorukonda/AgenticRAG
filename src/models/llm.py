"""
Text generation wrapper (google/flan-t5-small by default, CPU-friendly,
Hugging Face only — no paid APIs).

Callers are responsible for building strictly evidence-grounded prompts;
this module does not add any of its own "creative" framing. If no evidence
is passed in, callers should not call generate() at all — see
src/agents/reasoning_agent.py and final_generator in the graph, which return
an explicit "Insufficient evidence" message instead of ever prompting the
model without grounding.
"""

from typing import Optional

from src.config import models


class GenerationModel:
    def __init__(self, model_name: str = models.generation_model):
        self.model_name = model_name
        self._pipeline = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import pipeline

        self._pipeline = pipeline("text2text-generation", model=self.model_name)

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = models.generation_max_new_tokens,
        temperature: float = models.generation_temperature,
    ) -> str:
        if self._pipeline is None:
            self.load()

        do_sample = temperature > 0
        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-4) if do_sample else None,
        )
        return outputs[0]["generated_text"].strip()


_shared_instance = None


def get_generation_model() -> GenerationModel:
    global _shared_instance
    if _shared_instance is None:
        _shared_instance = GenerationModel()
    return _shared_instance


def build_grounded_prompt(instruction: str, evidence_blocks: list, question: str) -> str:
    """
    Construct a prompt that forces the model to answer only from the
    supplied evidence blocks (each a short string with its source already
    inlined by the caller), explicitly instructing it to say so if the
    evidence doesn't support an answer.
    """
    evidence_text = "\n\n".join(f"[{i+1}] {block}" for i, block in enumerate(evidence_blocks))
    return (
        f"{instruction}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Question: {question}\n\n"
        "Answer using ONLY the evidence above. If the evidence does not "
        "support a confident answer, say so explicitly instead of guessing."
    )
