"""Brain IDP target metadata and mutation registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import yaml


@dataclass(frozen=True)
class Mutation:
    id: str
    pos: int
    wt: str
    mt: str
    agg_rate_relative: float = 1.0
    alias: str = ""


@dataclass(frozen=True)
class Target:
    id: str
    name: str
    uniprot: str
    region: tuple[int, int]
    length: int
    ped_id: str
    disease: str
    sequence: str
    mutations: tuple[Mutation, ...] = field(default_factory=tuple)

    def mutant_sequence(self, mutation: Mutation) -> str:
        seq = list(self.sequence)
        idx = mutation.pos - 1
        assert seq[idx] == mutation.wt, (
            f"Expected {mutation.wt} at pos {mutation.pos}, got {seq[idx]}"
        )
        seq[idx] = mutation.mt
        return "".join(seq)


def load_targets(config_path: str | Path) -> dict[str, Target]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    targets = {}
    for tid, info in cfg["targets"].items():
        seq = info["sequence"].replace("\n", "").replace(" ", "")
        mutations = tuple(
            Mutation(
                id=m["id"],
                pos=m["pos"],
                wt=m["wt"],
                mt=m["mt"],
                agg_rate_relative=m.get("agg_rate_relative", 1.0),
                alias=m.get("alias", ""),
            )
            for m in info.get("mutations", [])
        )
        targets[tid] = Target(
            id=tid,
            name=info["name"],
            uniprot=info["uniprot"],
            region=tuple(info["region"]),
            length=info["length"],
            ped_id=info["ped_id"],
            disease=info["disease"],
            sequence=seq,
            mutations=mutations,
        )
    return targets
