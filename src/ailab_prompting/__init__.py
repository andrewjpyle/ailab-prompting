"""ailab_prompting: a small, eval-gated, local-only prompt-engineering lab.

Measures the three prompting strategies (zero-shot, few-shot, chain-of-thought) on a
structured-extraction task, plus a prompt-injection defense, against a recorded real model.
"""

from ailab_prompting.data import Example, InjectionExample, load_extractions, load_injections
from ailab_prompting.extract import Extractor, ExtractResult
from ailab_prompting.metrics import field_accuracy, record_exact, schema_macro_f1
from ailab_prompting.providers import (
    CassetteMissError,
    LLMProvider,
    OllamaProvider,
    ReplayProvider,
    StubProvider,
)
from ailab_prompting.strategies import STRATEGIES, PromptStrategy, build_prompt

__all__ = [
    "STRATEGIES",
    "CassetteMissError",
    "Example",
    "ExtractResult",
    "Extractor",
    "InjectionExample",
    "LLMProvider",
    "OllamaProvider",
    "PromptStrategy",
    "ReplayProvider",
    "StubProvider",
    "build_prompt",
    "field_accuracy",
    "load_extractions",
    "load_injections",
    "record_exact",
    "schema_macro_f1",
]

__version__ = "0.1.0"
