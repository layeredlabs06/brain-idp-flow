"""Download and parse PDB/mmCIF files, extract Cα coordinates."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import urllib.request
import gzip
import io

import numpy as np


PDB_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"


def download_mmcif(pdb_id: str, cache_dir: str | Path = "data/pdb") -> Path:
    """Download mmCIF file from RCSB, return cached path."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{pdb_id.lower()}.cif"

    if out_path.exists():
        return out_path

    url = PDB_URL.format(pdb_id=pdb_id.upper())
    response = urllib.request.urlopen(url)
    data = gzip.decompress(response.read())
    out_path.write_bytes(data)
    return out_path


def extract_ca_coords(
    cif_path: str | Path,
    chain_id: Optional[str] = None,
    model_num: int = 0,
) -> np.ndarray:
    """Extract Cα coordinates from mmCIF file.

    Returns: (L, 3) float32 array.
    """
    from Bio.PDB import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("prot", str(cif_path))
    model = list(structure.get_models())[model_num]

    chains = list(model.get_chains())
    if chain_id is not None:
        chain = next(c for c in chains if c.id == chain_id)
    else:
        chain = chains[0]

    ca_coords = []
    for residue in chain.get_residues():
        if residue.id[0] != " ":
            continue  # skip hetero atoms
        if "CA" in residue:
            ca_coords.append(residue["CA"].get_vector().get_array())

    return np.array(ca_coords, dtype=np.float32)


def extract_all_models_ca(cif_path: str | Path) -> np.ndarray:
    """Extract Cα from all models in a multi-model mmCIF (e.g. NMR ensemble).

    Returns: (N_models, L, 3) float32.
    """
    from Bio.PDB import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("prot", str(cif_path))

    all_models = []
    for model in structure.get_models():
        chain = list(model.get_chains())[0]
        ca = []
        for residue in chain.get_residues():
            if residue.id[0] != " ":
                continue
            if "CA" in residue:
                ca.append(residue["CA"].get_vector().get_array())
        all_models.append(ca)

    # Ensure consistent length
    min_len = min(len(m) for m in all_models)
    return np.array([m[:min_len] for m in all_models], dtype=np.float32)
