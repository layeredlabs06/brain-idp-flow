"""PyTorch Dataset for protein structure ensembles with mutation info."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class ProteinEnsembleDataset(Dataset):
    """Dataset of (sequence_ids, Cα_coords, mutation_info) triples.

    Loaded from NPZ with arrays:
      - seq_ids: (N,) int — index into a sequence table
      - coords:  (N, L, 3) float32 — Cα coordinates
      - mut_pos: (N,) int — mutation position (0 = wild type)
      - mut_aa:  (N,) int — mutant amino acid index (0 = wild type)
    """

    AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
    AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_VOCAB)}  # 0 = padding/WT

    def __init__(
        self,
        npz_path: str | Path,
        max_len: int = 160,
        augment_rotation: bool = False,
    ):
        data = np.load(str(npz_path), allow_pickle=True)
        self.coords = torch.from_numpy(data["coords"].astype(np.float32))
        self.seq_ids = torch.from_numpy(data["seq_ids"].astype(np.int64))

        self.mut_pos = torch.from_numpy(
            data["mut_pos"].astype(np.int64) if "mut_pos" in data else
            np.zeros(len(self.coords), dtype=np.int64)
        )
        self.mut_aa = torch.from_numpy(
            data["mut_aa"].astype(np.int64) if "mut_aa" in data else
            np.zeros(len(self.coords), dtype=np.int64)
        )

        # Filter by max_len
        L = self.coords.shape[1]
        if L > max_len:
            self.coords = self.coords[:, :max_len, :]

        self.max_len = max_len
        self.augment_rotation = augment_rotation

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> dict:
        coords = self.coords[idx].clone()

        # Center
        coords = coords - coords.mean(dim=0, keepdim=True)

        # Optional rotation augmentation
        if self.augment_rotation and self.training_mode:
            from brain_idp_flow.geometry.se3 import apply_random_rotation
            coords = apply_random_rotation(coords.unsqueeze(0)).squeeze(0)

        return {
            "coords": coords,
            "seq_id": self.seq_ids[idx],
            "mut_pos": self.mut_pos[idx],
            "mut_aa": self.mut_aa[idx],
        }

    @property
    def training_mode(self) -> bool:
        return getattr(self, "_training", False)

    def train(self) -> "ProteinEnsembleDataset":
        self._training = True
        return self

    def eval(self) -> "ProteinEnsembleDataset":
        self._training = False
        return self
