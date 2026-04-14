"""Deep Mutational Scanning (DMS) data loader.

Loads Aβ42 nucleation scores from Seuma et al. 2022 (Nature Communications).
"An atlas of amyloid aggregation" — DOI: 10.1038/s41467-022-34742-3

Data source: https://github.com/BEBlab/DIM-abeta
TSV format: ID = "{WT}-{pos}-{MT}", nscore_c = nucleation score, dataset = "single"
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

ABETA42_WT = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

SEUMA_TSV_URL = (
    "https://raw.githubusercontent.com/BEBlab/DIM-abeta/main/"
    "required%20data/MS_BL_BB_indels_processed_data.tsv"
)


def download_seuma_dms(
    cache_dir: str | Path = "data/dms",
) -> Path:
    """Download Seuma et al. 2022 DMS data from GitHub.

    Returns path to local TSV file.
    """
    import urllib.request

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / "seuma2022_abeta42.tsv"

    if local_path.exists():
        return local_path

    print(f"Downloading Seuma 2022 DMS data...")
    req = urllib.request.Request(SEUMA_TSV_URL, headers={"User-Agent": "brain-idp-flow/0.1"})
    response = urllib.request.urlopen(req, timeout=60)
    local_path.write_bytes(response.read())
    print(f"Saved to {local_path}")

    return local_path


def load_seuma_dms(
    filepath: str | Path | None = None,
    cache_dir: str | Path = "data/dms",
) -> list[dict]:
    """Load Seuma et al. 2022 DMS data.

    If filepath is None, downloads from GitHub automatically.
    Format: TSV with ID column like "D-1-K" and nscore_c column.

    Args:
        filepath: path to TSV file, or None to auto-download
        cache_dir: cache directory for auto-download

    Returns:
        list of dicts with keys: pos, wt, mt, mutation_id, nucleation_score, agg_rate
    """
    if filepath is None:
        filepath = download_seuma_dms(cache_dir)

    filepath = Path(filepath)

    if filepath.suffix == ".xlsx":
        return _load_excel(filepath)
    else:
        return _load_tsv_seuma(filepath)


def _load_tsv_seuma(filepath: Path) -> list[dict]:
    """Load Seuma 2022 TSV format.

    Columns: aa_seq, ID, dataset, mean_count, nscore_c, nscore1_c, ...
    ID format for singles: "D-1-K" (WT-pos-MT)
    """
    import csv

    results = []

    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            # Only single amino acid substitutions
            dataset = row.get("dataset", "").strip()
            if dataset != "single":
                continue

            # Parse ID: "D-1-K" -> (D, 1, K)
            mut_id = row.get("ID", "").strip()
            parts = mut_id.split("-")
            if len(parts) != 3:
                continue

            wt, pos_str, mt = parts[0], parts[1], parts[2]
            try:
                pos = int(pos_str)
            except ValueError:
                continue

            # Skip non-standard
            if wt not in STANDARD_AA or mt not in STANDARD_AA:
                continue
            if wt == mt:
                continue

            # Validate against sequence
            if pos < 1 or pos > len(ABETA42_WT):
                continue
            if ABETA42_WT[pos - 1] != wt:
                continue

            # Nucleation score
            try:
                nscore = float(row.get("nscore_c", ""))
            except (ValueError, TypeError):
                continue

            # Error estimate
            try:
                sigma = float(row.get("sigma", "0"))
            except (ValueError, TypeError):
                sigma = 0.0

            # fAD classification
            fad = row.get("fAD", "non-fAD").strip()

            # Convert nucleation score to relative aggregation rate
            # nscore > 0 = more aggregation-prone than WT
            # nscore < 0 = less aggregation-prone
            # Use exp() for monotonic mapping
            agg_rate = float(np.exp(np.clip(nscore, -5, 5)))

            results.append({
                "pos": pos,
                "wt": wt,
                "mt": mt,
                "mutation_id": f"{wt}{pos}{mt}",
                "nucleation_score": nscore,
                "sigma": sigma,
                "agg_rate": agg_rate,
                "is_fad": fad.startswith("fAD"),
                "target": "abeta42",
                "source": "Seuma2022",
            })

    print(f"Loaded {len(results)} single-point mutations from Seuma 2022 DMS")
    return results


def _load_excel(filepath: Path) -> list[dict]:
    """Load Excel format (fallback)."""
    try:
        import openpyxl
    except ImportError:
        raise ImportError("pip install openpyxl to read Excel files")

    wb = openpyxl.load_workbook(str(filepath), read_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(rows[0])]

    # Convert to TSV-like format and reuse parser
    import tempfile, csv
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        for row in rows[1:]:
            writer.writerow(dict(zip(headers, row)))
        tmp_path = Path(f.name)

    result = _load_tsv_seuma(tmp_path)
    tmp_path.unlink()
    return result


def generate_all_single_mutations(
    sequence: str = ABETA42_WT,
    target_id: str = "abeta42",
) -> list[dict]:
    """Generate all possible single-point mutations for a sequence.

    Useful when DMS data is not available — generates the mutation list
    for flow model / ESM-2 scoring.

    Returns:
        list of dicts with pos, wt, mt, mutation_id, target
    """
    mutations = []
    for pos_0, wt in enumerate(sequence):
        pos = pos_0 + 1
        for mt in STANDARD_AA:
            if mt == wt:
                continue
            mutations.append({
                "pos": pos,
                "wt": wt,
                "mt": mt,
                "mutation_id": f"{wt}{pos}{mt}",
                "target": target_id,
            })

    return mutations
