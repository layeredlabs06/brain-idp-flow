"""Download and parse Protein Ensemble Database (PED) entries."""

from __future__ import annotations

from pathlib import Path
import urllib.request
import gzip
import io

import numpy as np


PED_URL = "https://proteinensemble.org/api/v1/entries/{ped_id}/ensemble"


def download_ped_ensemble(
    ped_id: str,
    cache_dir: str | Path = "data/ped",
) -> Path:
    """Download PED ensemble file, return cached path."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{ped_id}.cif"

    if out_path.exists():
        return out_path

    url = PED_URL.format(ped_id=ped_id)
    try:
        response = urllib.request.urlopen(url)
        data = response.read()
        # Try to decompress if gzipped
        try:
            data = gzip.decompress(data)
        except gzip.BadGzipFile:
            pass
        out_path.write_bytes(data)
    except Exception as e:
        raise RuntimeError(f"Failed to download PED entry {ped_id}: {e}") from e

    return out_path


def load_ped_ensemble(
    ped_id: str,
    cache_dir: str | Path = "data/ped",
) -> np.ndarray:
    """Load PED ensemble as Cα coordinates.

    Returns: (N_frames, L, 3) float32 array.
    """
    from brain_idp_flow.data.pdb_loader import extract_all_models_ca

    cif_path = download_ped_ensemble(ped_id, cache_dir)
    return extract_all_models_ca(cif_path)


def load_ped_or_fallback(
    ped_id: str,
    sequence_length: int,
    cache_dir: str | Path = "data/ped",
    n_fallback: int = 50,
) -> np.ndarray:
    """Try loading PED ensemble; if unavailable, generate random coil fallback."""
    try:
        return load_ped_ensemble(ped_id, cache_dir)
    except Exception:
        # Random coil fallback: chain of 3.8Å Cα-Cα bonds with random angles
        rng = np.random.default_rng(42)
        frames = []
        for _ in range(n_fallback):
            coords = np.zeros((sequence_length, 3), dtype=np.float32)
            for i in range(1, sequence_length):
                direction = rng.standard_normal(3).astype(np.float32)
                direction /= np.linalg.norm(direction) + 1e-8
                coords[i] = coords[i - 1] + 3.8 * direction
            coords -= coords.mean(axis=0)
            frames.append(coords)
        return np.stack(frames)
