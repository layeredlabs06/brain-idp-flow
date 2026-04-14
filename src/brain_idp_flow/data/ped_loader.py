"""Download and parse Protein Ensemble Database (PED) entries."""

from __future__ import annotations

from pathlib import Path
import urllib.request
import gzip
import io

import numpy as np


PED_URL = "https://deposition.proteinensemble.org/api/v1/entries/{ped_id}/download-ensembles/"


def download_ped_ensemble(
    ped_id: str,
    cache_dir: str | Path = "data/ped",
) -> Path:
    """Download PED ensemble tar.gz, extract PDB files, return directory."""
    import tarfile

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = cache_dir / ped_id

    if extract_dir.exists() and any(extract_dir.glob("*.pdb")):
        return extract_dir

    url = PED_URL.format(ped_id=ped_id)
    tar_path = cache_dir / f"{ped_id}.tar.gz"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "brain-idp-flow/0.1"})
        response = urllib.request.urlopen(req, timeout=60)
        tar_path.write_bytes(response.read())
    except Exception as e:
        raise RuntimeError(f"Failed to download PED entry {ped_id}: {e}") from e

    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(tar_path), "r:gz") as tar:
        tar.extractall(str(extract_dir))

    tar_path.unlink()
    return extract_dir


def load_ped_ensemble(
    ped_id: str,
    cache_dir: str | Path = "data/ped",
) -> np.ndarray:
    """Load PED ensemble as Cα coordinates.

    Returns: (N_frames, L, 3) float32 array.
    """
    from Bio.PDB import PDBParser

    extract_dir = download_ped_ensemble(ped_id, cache_dir)

    # Find all PDB files recursively
    pdb_files = sorted(Path(extract_dir).rglob("*.pdb"))
    if not pdb_files:
        # Try mmCIF
        cif_files = sorted(Path(extract_dir).rglob("*.cif"))
        if cif_files:
            from brain_idp_flow.data.pdb_loader import extract_all_models_ca
            return extract_all_models_ca(cif_files[0])
        raise FileNotFoundError(f"No PDB/CIF files in {extract_dir}")

    parser = PDBParser(QUIET=True)
    frames = []
    for pdb_file in pdb_files:
        try:
            structure = parser.get_structure("prot", str(pdb_file))
            for model in structure.get_models():
                chain = list(model.get_chains())[0]
                ca = []
                for residue in chain.get_residues():
                    if residue.id[0] != " ":
                        continue
                    if "CA" in residue:
                        ca.append(residue["CA"].get_vector().get_array())
                if ca:
                    frames.append(ca)
        except Exception:
            continue

    if not frames:
        raise RuntimeError(f"No valid Cα coordinates extracted from {extract_dir}")

    # Ensure consistent length
    min_len = min(len(f) for f in frames)
    return np.array([f[:min_len] for f in frames], dtype=np.float32)


def load_ped_or_fallback(
    ped_id: str,
    sequence_length: int,
    cache_dir: str | Path = "data/ped",
    n_fallback: int = 500,
) -> np.ndarray:
    """Try loading PED ensemble; if unavailable, generate random coil fallback.

    The fallback generates diverse random coil conformations with realistic
    bond lengths (3.8 Å Cα-Cα) and a persistence length of ~5 residues.
    """
    try:
        ensemble = load_ped_ensemble(ped_id, cache_dir)
        # If PED has very few frames, augment with random perturbations
        if len(ensemble) < 100:
            ensemble = _augment_ensemble(ensemble, target_n=max(500, len(ensemble)))
        return ensemble
    except Exception:
        return _generate_random_coil_ensemble(sequence_length, n_fallback)


def _augment_ensemble(
    ensemble: np.ndarray,
    target_n: int = 500,
) -> np.ndarray:
    """Augment a small PED ensemble with Gaussian perturbations."""
    n_existing = len(ensemble)
    if n_existing >= target_n:
        return ensemble

    rng = np.random.default_rng(42)
    augmented = [ensemble]

    while sum(len(a) for a in augmented) < target_n:
        # Small random perturbation of existing frames
        idx = rng.choice(n_existing, size=min(n_existing, target_n - sum(len(a) for a in augmented)))
        noise_scale = 0.5  # Angstroms — small perturbation
        perturbed = ensemble[idx] + rng.normal(0, noise_scale, ensemble[idx].shape).astype(np.float32)
        # Re-center
        perturbed -= perturbed.mean(axis=1, keepdims=True)
        augmented.append(perturbed)

    return np.concatenate(augmented, axis=0)[:target_n]


def _generate_random_coil_ensemble(
    sequence_length: int,
    n_frames: int = 500,
) -> np.ndarray:
    """Generate random coil ensemble with persistence length.

    Uses a worm-like chain model with persistence length ~5 residues
    for more realistic IDP conformations than pure random walk.
    """
    rng = np.random.default_rng(42)
    bond_length = 3.8  # Cα-Cα distance in Angstroms
    persistence_length = 5.0  # in residue units
    # Correlation factor between consecutive bond vectors
    kappa = np.exp(-1.0 / persistence_length)

    frames = []
    for _ in range(n_frames):
        coords = np.zeros((sequence_length, 3), dtype=np.float32)
        # Initial direction
        direction = rng.standard_normal(3).astype(np.float32)
        direction /= np.linalg.norm(direction) + 1e-8

        for i in range(1, sequence_length):
            # Correlated random walk (worm-like chain)
            noise = rng.standard_normal(3).astype(np.float32)
            noise /= np.linalg.norm(noise) + 1e-8
            direction = kappa * direction + (1 - kappa) * noise
            direction /= np.linalg.norm(direction) + 1e-8
            coords[i] = coords[i - 1] + bond_length * direction

        coords -= coords.mean(axis=0)
        frames.append(coords)

    return np.stack(frames)
