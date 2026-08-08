"""Minimal ModelOutput container.

The research code imported this from `transformers`. Vendoring a small
dataclass keeps `patchex.model` importable for inference without pulling in
the Trainer stack (transformers is still required for the ESM-2 tokenizer in
`patchex.embed`, but not for the model definition itself).
"""
from dataclasses import dataclass, field, fields
from typing import Any, Optional


@dataclass
class ModelOutput:
    loss: Optional[Any] = None
    pred: Optional[Any] = None
    w_A: Optional[Any] = None
    w_B: Optional[Any] = None
    patch_preds: Optional[Any] = None
    patch_mask: Optional[Any] = None
    y_mix: Optional[Any] = None
    y_direct: Optional[Any] = None

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return tuple(self.to_tuple())[key]

    def keys(self):
        return [f.name for f in fields(self) if getattr(self, f.name) is not None]

    def to_tuple(self):
        return tuple(getattr(self, k) for k in self.keys())
