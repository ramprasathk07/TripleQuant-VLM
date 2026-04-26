"""
Quantization module for TripleQuant-VLM.

Importing this package automatically registers all quantizers via the
``@register`` decorator. Entry points need only call:

    from src.quantization.registry import get_quantizer_class
    cls = get_quantizer_class("awq")
"""
# Import quantizers to trigger @register decoration
from .awq import AWQQuantizer
from .gptq import GPTQQuantizer
from .base import BaseQuantizer
from .registry import get_quantizer_class, list_methods

__all__ = [
    "AWQQuantizer",
    "GPTQQuantizer",
    "BaseQuantizer",
    "get_quantizer_class",
    "list_methods",
]
