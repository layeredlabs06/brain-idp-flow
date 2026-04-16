"""Frozen ESM-2 per-residue embedding wrapper."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class ESM2Embedder(nn.Module):
    """Wraps a frozen ESM-2 model to produce per-residue embeddings.

    Default: esm2_t12_35M_UR50D (480-dim output, ~35M params).
    """

    def __init__(
        self,
        model_name: str = "esm2_t12_35M_UR50D",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self._device = device
        self._model = None
        self._alphabet = None
        self._batch_converter = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import esm

        model, alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
        model = model.eval()
        if self._device is not None:
            model = model.to(self._device)

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad_(False)

        self._model = model
        self._alphabet = alphabet
        self._batch_converter = alphabet.get_batch_converter()

    @property
    def embed_dim(self) -> int:
        dims = {
            "esm2_t12_35M_UR50D": 480,
            "esm2_t30_150M_UR50D": 640,
            "esm2_t33_650M_UR50D": 1280,
            "esm2_t36_3B_UR50D": 2560,
        }
        return dims.get(self.model_name, 480)

    @property
    def num_layers(self) -> int:
        """Number of transformer layers in the ESM-2 model."""
        self._lazy_load()
        return self._model.num_layers

    @torch.no_grad()
    def forward(self, sequences: list[str]) -> Tensor:
        """Embed a batch of protein sequences.

        Args:
            sequences: list of amino acid strings, length B.

        Returns:
            (B, L_max, D) tensor of per-residue embeddings (padded).
        """
        self._lazy_load()

        data = [("prot", seq) for seq in sequences]
        _labels, _strs, tokens = self._batch_converter(data)

        if self._device is not None:
            tokens = tokens.to(self._device)

        results = self._model(tokens, repr_layers=[self._model.num_layers])
        embeddings = results["representations"][self._model.num_layers]

        # Strip BOS/EOS tokens: [BOS, seq..., EOS] -> [seq...]
        return embeddings[:, 1:-1, :]

    def embed_single(self, sequence: str) -> Tensor:
        """Embed a single sequence. Returns (L, D)."""
        return self.forward([sequence]).squeeze(0)

    @torch.no_grad()
    def embed_all_layers(self, sequences: list[str]) -> dict[int, Tensor]:
        """Extract representations from all transformer layers.

        Used for CKA/CCA analysis to understand how representations
        evolve across layers and differ between model scales.

        Args:
            sequences: list of amino acid strings, length B.

        Returns:
            Dict mapping layer index to (B, L_max, D) tensor.
            Layer 0 = input embedding, layers 1..N = transformer outputs.
        """
        self._lazy_load()

        data = [("prot", seq) for seq in sequences]
        _labels, _strs, tokens = self._batch_converter(data)

        if self._device is not None:
            tokens = tokens.to(self._device)

        all_layers = list(range(self._model.num_layers + 1))
        results = self._model(tokens, repr_layers=all_layers)

        return {
            layer: results["representations"][layer][:, 1:-1, :]
            for layer in all_layers
        }

    def embed_single_all_layers(self, sequence: str) -> dict[int, Tensor]:
        """Extract all-layer representations for a single sequence.

        Returns:
            Dict mapping layer index to (L, D) tensor.
        """
        layer_dict = self.embed_all_layers([sequence])
        return {layer: emb.squeeze(0) for layer, emb in layer_dict.items()}
