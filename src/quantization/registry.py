"""
Decorator-based registry that maps quantization method names → quantizer classes.

Usage (inside a quantizer module):
    from .registry import register

    @register("awq")
    class AWQQuantizer(BaseQuantizer):
        ...

Usage (from entry point):
    from src.quantization.registry import get_quantizer_class
    cls = get_quantizer_class("awq")   # → AWQQuantizer
"""
from __future__ import annotations

from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseQuantizer

_REGISTRY: dict[str, Type["BaseQuantizer"]] = {}


def register(method_name: str):
    """Class decorator that registers a quantizer under *method_name*."""

    def _decorator(cls):
        name = method_name.lower()
        if name in _REGISTRY:
            raise ValueError(
                f"Quantizer '{name}' is already registered by {_REGISTRY[name].__name__}. "
                "Use a unique method name."
            )
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_quantizer_class(method: str) -> Type["BaseQuantizer"]:
    """Return the quantizer class registered under *method*.

    Raises:
        KeyError: if no quantizer is registered for *method*.
    """
    key = method.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"No quantizer registered for method '{key}'. "
            f"Available: [{available}]"
        )
    return _REGISTRY[key]


def list_methods() -> list[str]:
    """Return a sorted list of all registered method names."""
    return sorted(_REGISTRY)
