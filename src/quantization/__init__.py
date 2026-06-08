"""Quantization backends and factory interface for TripleQuant-VLM.

Exports the base quantizer class and the factory function to retrieve
quantizer instances dynamically.
"""
from __future__ import annotations

from .base import BaseQuantizer
from .factory import get_quantizer
