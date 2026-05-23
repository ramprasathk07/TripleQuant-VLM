"""Registry decorator mapping quantization method names to quantizer classes.

This module provides a registration system for quantization methods. Classes that
implement quantization strategies can register themselves using the @register()
decorator, enabling dynamic lookup by method name via get_quantizer_class().

Example:
    from .registry import register

    @register("awq")
    class AWQQuantizer(BaseQuantizer):
        pass

    from .registry import get_quantizer_class
    cls = get_quantizer_class("awq")  # Returns AWQQuantizer
"""
from __future__ import annotations

from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseQuantizer

_REGISTRY: dict[str, Type["BaseQuantizer"]] = {}  # Maps method names to registered quantizer classes.


def register(method_name: str):
    """Register a quantizer class under a method name.

    Args:
        method_name: Unique identifier for the quantization method (e.g., "awq", "gptq").

    Returns:
        Decorator function that registers the decorated class and returns it.

    Raises:
        ValueError: If method_name is already registered.
    """
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
    """Return the quantizer class registered for a given method name.

    Args:
        method: Quantization method name (case-insensitive).

    Returns:
        Registered quantizer class for the method.

    Raises:
        KeyError: If no quantizer is registered for the method.
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
    """List all registered quantization method names in sorted order.

    Returns:
        Sorted list of registered method identifiers.
    """
    return sorted(_REGISTRY)
