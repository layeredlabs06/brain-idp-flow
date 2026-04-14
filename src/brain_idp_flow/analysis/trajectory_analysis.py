"""Trajectory-based feature extraction from ODE sampling dynamics.

Extracts novel features from the flow matching ODE trajectory:
- Velocity magnitude profiles (folding velocity fingerprints)
- Velocity convergence times
- Contact switching rates (contact kinetics)
- Contact formation ordering
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Direction A: Folding Velocity Fingerprints
# ---------------------------------------------------------------------------


def velocity_magnitude_profile(velocities: Tensor) -> Tensor:
    """Per-residue velocity magnitude at each ODE step.

    Args:
        velocities: (n_steps, B, L, 3)

    Returns:
        (n_steps, B, L) velocity magnitudes
    """
    return velocities.norm(dim=-1)


def late_stage_velocity(
    velocities: Tensor,
    times: Tensor,
    threshold: float = 0.8,
) -> Tensor:
    """Mean velocity magnitude for t > threshold.

    Args:
        velocities: (n_steps, B, L, 3)
        times: (n_steps,)
        threshold: time cutoff (default 0.8)

    Returns:
        (B, L) mean late-stage velocity magnitude
    """
    mask = times > threshold
    if not mask.any():
        mask = torch.ones_like(times, dtype=torch.bool)
        mask[: len(times) // 2] = False  # fallback: last half

    late_vel = velocity_magnitude_profile(velocities[mask])  # (K, B, L)
    return late_vel.mean(dim=0)


def velocity_convergence_time(
    velocities: Tensor,
    times: Tensor,
    epsilon: float = 0.1,
) -> Tensor:
    """Per-residue convergence time: earliest t where |v_i| < epsilon * max|v_i|.

    Args:
        velocities: (n_steps, B, L, 3)
        times: (n_steps,)
        epsilon: fraction of max velocity

    Returns:
        (B, L) convergence time per residue (1.0 if never converged)
    """
    mag = velocity_magnitude_profile(velocities)  # (n_steps, B, L)
    max_mag = mag.max(dim=0).values  # (B, L)
    threshold = epsilon * max_mag  # (B, L)

    # Find first step where magnitude drops below threshold
    B, L = max_mag.shape
    conv_time = torch.ones(B, L, device=velocities.device)

    for step_idx in range(len(times)):
        below = mag[step_idx] < threshold  # (B, L)
        # Only update if not already converged
        not_yet = conv_time >= 1.0
        update = below & not_yet
        conv_time[update] = times[step_idx].item()

    return conv_time


def velocity_fingerprint_features(
    velocities: Tensor,
    times: Tensor,
    mutation_pos: int,
    window: int = 5,
) -> dict:
    """Extract velocity-based features for a single mutation.

    Args:
        velocities: (n_steps, B, L, 3) from trajectory sampling
        times: (n_steps,)
        mutation_pos: 0-indexed mutation position
        window: residues on each side for neighbor comparison

    Returns:
        dict of scalar features (averaged over B samples)
    """
    _, B, L, _ = velocities.shape
    pos = min(mutation_pos, L - 1)

    # Late-stage velocity
    late_vel = late_stage_velocity(velocities, times)  # (B, L)
    late_vel_site = late_vel[:, pos].mean().item()
    late_vel_global = late_vel.mean().item()

    # Convergence time
    conv = velocity_convergence_time(velocities, times)  # (B, L)
    conv_site = conv[:, pos].mean().item()

    # Convergence delay vs neighbors
    start = max(0, pos - window)
    end = min(L, pos + window + 1)
    neighbor_mask = torch.ones(L, dtype=torch.bool)
    neighbor_mask[pos] = False
    neighbor_mask[:start] = False
    neighbor_mask[end:] = False
    if neighbor_mask.any():
        conv_neighbors = conv[:, neighbor_mask].mean().item()
    else:
        conv_neighbors = conv_site
    conv_delay = conv_site - conv_neighbors

    # Velocity variance across samples at late stage (model uncertainty)
    vel_var = late_vel[:, pos].var().item()

    # Velocity magnitude profile at site (for visualization)
    mag = velocity_magnitude_profile(velocities)  # (steps, B, L)
    site_profile = mag[:, :, pos].mean(dim=1)  # (steps,)

    return {
        "late_velocity_site": late_vel_site,
        "late_velocity_global": late_vel_global,
        "convergence_time_site": conv_site,
        "convergence_delay_vs_neighbors": conv_delay,
        "velocity_variance_late": vel_var,
        "_velocity_profile": site_profile.cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Direction B: Contact Kinetics
# ---------------------------------------------------------------------------


def contact_switching_rate(
    coords: Tensor,
    threshold: float = 8.0,
) -> Tensor:
    """Count contact↔non-contact transitions per residue pair across ODE steps.

    Uses streaming computation to avoid storing full (steps, B, L, L) tensor.

    Args:
        coords: (n_steps, B, L, 3)
        threshold: contact distance threshold in Angstroms

    Returns:
        (B, L, L) switching rate (transitions / (n_steps - 1))
    """
    n_steps, B, L, _ = coords.shape
    switches = torch.zeros(B, L, L, device=coords.device)

    # Initial contact state
    prev_dists = torch.cdist(coords[0], coords[0])  # (B, L, L)
    prev_contacts = prev_dists < threshold

    for step in range(1, n_steps):
        curr_dists = torch.cdist(coords[step], coords[step])
        curr_contacts = curr_dists < threshold
        switches += (curr_contacts != prev_contacts).float()
        prev_contacts = curr_contacts

    return switches / max(n_steps - 1, 1)


def contact_formation_order(
    coords: Tensor,
    times: Tensor,
    threshold: float = 8.0,
    persistence: int = 3,
) -> Tensor:
    """For contacts present at t=1: find when they became stably formed.

    A contact is "stably formed" at step s if it is present at steps
    s, s+1, ..., s+persistence-1 (i.e., persists for `persistence` consecutive steps).
    This avoids counting transient random contacts from early noise.

    Args:
        coords: (n_steps, B, L, 3)
        times: (n_steps,)
        threshold: contact distance in Angstroms
        persistence: minimum consecutive steps to count as stable

    Returns:
        (B, L, L) stable formation time (NaN for contacts not present at t=1)
    """
    n_steps, B, L, _ = coords.shape

    # Final contacts
    final_dists = torch.cdist(coords[-1], coords[-1])
    final_contacts = final_dists < threshold

    # Pre-compute all contact maps
    all_contacts = []
    for step in range(n_steps):
        dists = torch.cdist(coords[step], coords[step])
        all_contacts.append(dists < threshold)

    # Track stable formation time
    formation_time = torch.full(
        (B, L, L), float("nan"), device=coords.device
    )

    for step in range(n_steps - persistence + 1):
        # Check if contact is present for `persistence` consecutive steps
        persistent = all_contacts[step]
        for k in range(1, persistence):
            persistent = persistent & all_contacts[step + k]

        # Update: only for final contacts not yet assigned
        not_assigned = formation_time.isnan()
        update = final_contacts & not_assigned & persistent
        formation_time[update] = times[step].item()

    return formation_time


def contact_kinetics_features(
    coords: Tensor,
    times: Tensor,
    mutation_pos: int,
    threshold: float = 8.0,
    window: int = 5,
) -> dict:
    """Extract contact dynamics features for a single mutation.

    Args:
        coords: (n_steps, B, L, 3)
        times: (n_steps,)
        mutation_pos: 0-indexed mutation position
        threshold: contact distance
        window: neighbor window size

    Returns:
        dict of scalar features (averaged over B samples)
    """
    _, B, L, _ = coords.shape
    pos = min(mutation_pos, L - 1)

    # Switching rate
    switch_rate = contact_switching_rate(coords, threshold)  # (B, L, L)
    switch_site = switch_rate[:, pos, :].mean().item()
    switch_global = switch_rate.mean().item()

    # Long-range switching (|i-j| > 10)
    lr_mask = torch.abs(torch.arange(L, device=coords.device) - pos) > 10
    if lr_mask.any():
        switch_lr = switch_rate[:, pos, :][:, lr_mask].mean().item()
    else:
        switch_lr = 0.0

    # Formation order (stable contacts, persistence=3)
    form_order = contact_formation_order(coords, times, threshold)  # (B, L, L)
    site_form = form_order[:, pos, :]
    valid = ~site_form.isnan()
    if valid.any():
        contact_order_site = site_form[valid].mean().item()
    else:
        contact_order_site = 0.5  # neutral default

    # Early contact fraction: stable contacts at site formed before t=0.5
    early_mask = site_form < 0.5
    early_and_valid = early_mask & valid
    if valid.any():
        early_frac = early_and_valid.float().sum().item() / valid.float().sum().item()
    else:
        early_frac = 0.5  # neutral default

    # Contact formation delay vs global: how late does this site form contacts
    # compared to all other residue pairs?
    all_valid = ~form_order.isnan()
    if all_valid.any() and valid.any():
        global_mean_formation = form_order[all_valid].mean().item()
        site_mean_formation = site_form[valid].mean().item()
        formation_delay = site_mean_formation - global_mean_formation
    else:
        formation_delay = 0.0

    return {
        "switching_rate_site": switch_site,
        "switching_rate_global": switch_global,
        "switching_rate_long_range": switch_lr,
        "contact_order_site": contact_order_site,
        "early_contact_fraction_site": early_frac,
        "contact_formation_delay": formation_delay,
    }


# ---------------------------------------------------------------------------
# Combined extraction
# ---------------------------------------------------------------------------


def extract_trajectory_features(
    trajectory: dict,
    mutation_pos: int,
    threshold: float = 8.0,
    window: int = 5,
) -> dict:
    """Extract all trajectory-based features from a single mutation's ODE run.

    Args:
        trajectory: dict from sample_ensemble_with_trajectory containing
            "coords": (n_steps, B, L, 3)
            "velocities": (n_steps, B, L, 3)
            "times": (n_steps,)
        mutation_pos: 0-indexed mutation position
        threshold: contact distance threshold
        window: neighbor window for relative features

    Returns:
        dict of all trajectory features (velocity + contact kinetics)
    """
    coords = trajectory["coords"]
    velocities = trajectory["velocities"]
    times = trajectory["times"]

    vel_feats = velocity_fingerprint_features(
        velocities, times, mutation_pos, window=window,
    )
    contact_feats = contact_kinetics_features(
        coords, times, mutation_pos, threshold=threshold, window=window,
    )

    # Merge (velocity profile is prefixed with _ for non-scalar)
    features = {}
    features.update(vel_feats)
    features.update(contact_feats)
    return features
