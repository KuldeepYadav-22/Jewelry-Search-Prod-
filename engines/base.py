"""
engines/base.py
Abstract base class for all inference engines.
"""
from __future__ import annotations

import abc
import torch


class BaseEngine(abc.ABC):
    """
    Resolves compute device at construction. All engines inherit from this.
    """

    def __init__(self, config):
        self.config = config
        self.device = self._resolve_device(getattr(config, "device", "auto"))

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device(device_str)

    @abc.abstractmethod
    def run(self, image):
        """Execute the engine on a single PIL image."""
        ...
