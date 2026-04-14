"""ESM-2 masked marginal log-likelihood ratio for mutation effect scoring."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


class ESM2MutationScorer:
    """Zero-shot mutation effect scoring via ESM-2 masked marginal probabilities.

    For a mutation X→Y at position i:
      LLR = log P(Y | context) - log P(X | context)
    where context = all other residues (position i is masked).
    """

    def __init__(
        self,
        model_name: str = "esm2_t12_35M_UR50D",
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._alphabet = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import esm
        self._model, self._alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
        self._model = self._model.eval().to(self.device)
        for p in self._model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def score_mutation(
        self,
        sequence: str,
        pos: int,
        wt_aa: str,
        mt_aa: str,
    ) -> dict:
        """Score a single mutation.

        Args:
            sequence: WT amino acid sequence
            pos: 1-indexed mutation position
            wt_aa: wild-type amino acid (single letter)
            mt_aa: mutant amino acid (single letter)

        Returns:
            dict with llr_site, log_p_wt, log_p_mt, delta_ppl
        """
        self._load()

        idx_0 = pos - 1  # 0-indexed in sequence
        assert sequence[idx_0] == wt_aa, f"Expected {wt_aa} at pos {pos}, got {sequence[idx_0]}"

        batch_converter = self._alphabet.get_batch_converter()

        # Tokenize and mask the mutation position
        _, _, tokens = batch_converter([("prot", sequence)])
        tokens = tokens.to(self.device)

        # In ESM tokens: position in tokens = pos (because of BOS token at index 0)
        mask_idx = pos  # BOS shifts everything by 1
        tokens_masked = tokens.clone()
        tokens_masked[0, mask_idx] = self._alphabet.mask_idx

        # Forward pass with masked token
        logits = self._model(tokens_masked)["logits"]  # (1, L+2, vocab)
        log_probs = F.log_softmax(logits[0, mask_idx], dim=-1)

        wt_token = self._alphabet.get_idx(wt_aa)
        mt_token = self._alphabet.get_idx(mt_aa)

        log_p_wt = log_probs[wt_token].item()
        log_p_mt = log_probs[mt_token].item()
        llr = log_p_mt - log_p_wt

        # Full-sequence pseudo-perplexity (optional, more expensive)
        # For speed, compute pseudo-log-likelihood for WT and mutant sequences
        ppl_wt = self._pseudo_ppl(sequence)
        mut_seq = list(sequence)
        mut_seq[idx_0] = mt_aa
        ppl_mt = self._pseudo_ppl("".join(mut_seq))

        return {
            "llr_site": llr,
            "log_p_wt": log_p_wt,
            "log_p_mt": log_p_mt,
            "ppl_wt": ppl_wt,
            "ppl_mt": ppl_mt,
            "delta_ppl": ppl_mt - ppl_wt,
        }

    @torch.no_grad()
    def _pseudo_ppl(self, sequence: str) -> float:
        """Compute pseudo-perplexity by masking each position sequentially.

        For speed on long sequences, sample 20 random positions instead of all.
        """
        import random

        batch_converter = self._alphabet.get_batch_converter()
        _, _, tokens = batch_converter([("prot", sequence)])
        tokens = tokens.to(self.device)

        L = len(sequence)
        positions = list(range(1, L + 1))  # token positions (BOS offset)
        if L > 20:
            positions = random.sample(positions, 20)

        total_nll = 0.0
        for mask_pos in positions:
            masked = tokens.clone()
            masked[0, mask_pos] = self._alphabet.mask_idx

            logits = self._model(masked)["logits"]
            log_probs = F.log_softmax(logits[0, mask_pos], dim=-1)
            total_nll -= log_probs[tokens[0, mask_pos]].item()

        return total_nll / len(positions)


def score_all_mutations(targets: dict, device: Optional[torch.device] = None) -> list[dict]:
    """Score all mutations across all targets with ESM-2 LLR.

    Returns list of dicts with target, mutation_id, llr_site, etc.
    """
    scorer = ESM2MutationScorer(device=device)
    results = []

    for tid, target in targets.items():
        print(f"Scoring {target.name}...")
        for mut in target.mutations:
            scores = scorer.score_mutation(
                target.sequence, mut.pos, mut.wt, mut.mt
            )
            results.append({
                "target": tid,
                "mutation": mut.id,
                "pos": mut.pos,
                "wt": mut.wt,
                "mt": mut.mt,
                "agg_rate": mut.agg_rate_relative,
                **scores,
            })
            print(f"  {mut.id}: LLR={scores['llr_site']:.3f}, "
                  f"ΔPPL={scores['delta_ppl']:.3f}")

    return results
