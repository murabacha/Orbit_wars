"""
main.py — Orbit Wars: Producer Hybrid v7

Base: Producer Hybrid v6 (faster tempo, floor-sized fleets, comet expiry
guard + evacuation, quadratic frontline pressure).

v7 adds five upgrades, each behind its own config switch so they can be
ablated independently:

  [v7-1] Reactive-reinforcement margin  (enable_reinforce_margin)
         capture_floor() has always accepted an optional per-target,
         per-arrival-turn `reinforcement` margin — v6 never used it.
         We estimate how many enemy ships can reach each target before our
         fleet arrives and pad the capture floor accordingly. Long-ETA
         captures stop dying to a 3-ship reactive top-up (the engine's
         tie-annihilation rule makes a 1-ship shortfall a total loss).

  [v7-2] Coordinated pair strikes  (enable_pair_strikes)
         LaunchSet / _greedy_select / the exact flow scorer all support
         multi-launch candidates (L > 1) — v6 only ever used L = 1.
         For every shortlisted target we add ONE pair candidate: the two
         strongest viable sources splitting a floor-sized capture between
         them. Pairs are only generated where no single source can clear
         the floor alone, so they unlock otherwise-impossible captures
         without duplicating the single-source options. The exact
         recurrence scores staggered arrivals truthfully (first fleet
         softens, second captures), so the pair does not need turn-perfect
         synchronisation to be evaluated correctly.

  [v7-3] Comet capture  (enable_comet_capture)
         planner_core.attack_target_mask() categorically excludes comets,
         so v6 could never *take* one — it only guarded against attacking
         expiring ones. Comets produce ships and live ~40 turns; an
         early-path comet is free production. We append up to
         `max_comet_targets` enemy/neutral comets with enough remaining
         lifetime to the shortlist, and gate candidates with a stricter
         hold requirement (eta + comet_capture_min_hold <= remaining) so
         every capture has time to pay back. v6's evacuation logic
         already handles getting the garrison off before expiry.

  [v7-4] Production-biased targeting  (prod_target_bonus)
         The stock shortlist ranks attack targets by proximity alone.
         We rank by  -distance + prod_target_bonus * production  instead.
         constants.py's early-termination calibration weighs production
         5x over ships in 2P dominance scoring — the engine itself says
         production is what wins.

  [v7-5] Pressure-aware safe drain  (enable_adaptive_drain)
         v6 protects every source over the full horizon H, so rear
         planets under zero threat still hold back H turns of garrison.
         We scale the protection window per source with normalised enemy
         pressure: frontline planets keep the full H, quiet rear planets
         drop to `drain_min_horizon`, freeing ships exactly where it is
         safe to do so.

Everything else is v6 behaviour, untouched.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from dataclasses import dataclass

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch import Tensor

from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import (
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache, min_distance_to_targets
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _plan_regroup,
    _stable_topk_indices,
    attack_target_mask,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    friendly_flip_targets,
    is_comet_planet,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs — v6 values plus the v7 feature block at the end."""

    # ── planning window ───────────────────────────────────────────────────
    horizon: int = 15

    # ── shortlists ────────────────────────────────────────────────────────
    max_sources_per_lane: int = 10
    max_offensive_targets: int = 14
    max_defensive_targets: int = 5

    # ── scoring / greedy ─────────────────────────────────────────────────
    max_waves_per_turn: int = 8
    roi_threshold: float = 1.35
    min_ships_to_launch: float = 3.0

    # ── regroup ───────────────────────────────────────────────────────────
    enable_regroup: bool = True
    max_regroup_time: float = 5.0
    regroup_pressure_delta_min: float = 0.20
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 9
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 5e-4

    # ── FFA-only leader/prod bonuses (unused in 2P) ───────────────────────
    ffa_leader_attack_bonus: float = 0.0
    ffa_target_prod_bonus: float = 0.0

    # ── floor-sized fleets ────────────────────────────────────────────────
    enable_floor_sized_fleets: bool = True
    floor_pad_ships: float = 3.0
    floor_pad_frac: float = 0.12

    # ── comet expiry guard / evacuation (v6) ─────────────────────────────
    comet_min_hold: float = 4.0
    comet_evac_steps: int = 6

    # ══ v7 features ═══════════════════════════════════════════════════════

    # [v7-1] reactive-reinforcement margin in the capture floor
    enable_reinforce_margin: bool = True
    reinforce_margin_frac: float = 0.35    # fraction of reachable enemy mass counted
    reinforce_eta_free: float = 3.0        # turns the enemy can't react within
    reinforce_eta_scale: float = 5.0       # ramp length to full reaction likelihood

    # [v7-2] coordinated two-source strikes
    enable_pair_strikes: bool = True
    pair_pad_ships: float = 4.0            # extra pad over the single-wave pad
                                           # (staggered-arrival safety)

    # [v7-3] comet capture
    enable_comet_capture: bool = True
    comet_capture_min_hold: float = 10.0   # required (remaining - eta) to attack a comet
    max_comet_targets: int = 2

    # [v7-4] production-biased target shortlist
    prod_target_bonus: float = 3.0         # distance-units credited per production point

    # [v7-5] pressure-aware safe drain
    enable_adaptive_drain: bool = True
    drain_min_horizon: int = 6             # protection window for zero-pressure sources


def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


# ---------------------------------------------------------------------------
# Pressure proxy (v6 quadratic decay)
# ---------------------------------------------------------------------------

def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    """Quadratic-decay enemy-mass proxy per planet — ``[P]``."""
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye
    linear = (1.0 - d0 / reach_dist).clamp(min=0.0)
    decay = linear * linear
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)


# ---------------------------------------------------------------------------
# [v7-1] Reactive-reinforcement margin
# ---------------------------------------------------------------------------

def reactive_reinforcement_margin(
    obs, cache, *, target_idx: Tensor, K: int, player_id: int,
    frac: float, eta_free: float, eta_scale: float,
) -> Tensor:
    """Estimated enemy ships that can reach each target by arrival turn k.

    Returns ``[T, K]`` — fed into ``capture_floor(reinforcement=...)`` which
    adds it to the defender count on capture cells (never on reinforcement
    cells of planets we already own at k).

    Model: enemy planet ``e`` can deliver its garrison to target ``t`` by turn
    ``k`` if ``ceil(dist(e,t) / speed(garrison_e)) + 1 <= k`` (the +1 is one
    turn of reaction delay). Counted mass is scaled by ``frac`` (the enemy
    won't strip a planet bare just to reinforce) and by the
    ``reinforcement_timing_factor`` ramp ρ(k) ∈ [0,1]: shots that land within
    ``eta_free`` turns face no reaction at all, then likelihood ramps to 1
    over the next ``eta_scale`` turns. Pure arithmetic → CPU/CUDA agree.
    """
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    T = int(target_idx.shape[0])
    if T == 0 or K <= 0:
        return torch.zeros(T, max(K, 0), dtype=dtype, device=device)

    pid = int(player_id)
    ships = obs.ships.to(dtype)
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != pid)

    tgt = target_idx.clamp(0, P - 1)
    d_et = cache.cross_dist[0].to(dtype)[:, tgt]                       # [P, T]
    speed_e = fleet_speed(ships.clamp(min=1.0)).clamp(min=1e-6)       # [P]
    steps = torch.ceil(d_et / speed_e.view(P, 1)) + 1.0               # [P, T]

    k_grid = torch.arange(1, K + 1, device=device, dtype=dtype)       # [K]
    arrives = steps.view(P, T, 1) <= k_grid.view(1, 1, K)             # [P, T, K]

    # the target's own garrison is the defender, not a reinforcer
    is_self = (torch.arange(P, device=device).view(P, 1) == tgt.view(1, T))
    counted = enemy.view(P, 1) & ~is_self                              # [P, T]

    contrib = torch.where(
        arrives & counted.unsqueeze(-1),
        ships.view(P, 1, 1).expand(P, T, K),
        torch.zeros(1, dtype=dtype, device=device),
    )
    total = contrib.sum(dim=0)                                          # [T, K]
    rho = reinforcement_timing_factor(
        k_grid, eta_free=float(eta_free), eta_scale=float(eta_scale)
    ).view(1, K)
    return float(frac) * rho * total


# ---------------------------------------------------------------------------
# [v7-3] + [v7-4] Target shortlist: prod-biased ranking + comet appendix
# ---------------------------------------------------------------------------

def build_target_shortlist_v7(
    obs, obs_tensors, garrison_status, cache, *,
    config, K_eta, H, prod, source_mask, comet_remaining: Tensor | None,
):
    """Offensive (prod-biased) ∪ defensive ∪ capturable-comet shortlist.

    Re-implements planner_core.build_target_shortlist with two changes:
    attack ranking is ``-distance + prod_target_bonus * prod`` instead of
    pure proximity [v7-4], and up to ``max_comet_targets`` enemy/neutral
    comets with enough remaining lifetime are appended [v7-3] (the stock
    ``attack_target_mask`` excludes comets categorically).
    """
    P = obs.P
    device = obs.device
    n_attack = max(1, min(int(config.max_offensive_targets), P))
    R = max(0, min(int(config.max_defensive_targets), P))

    attack_mask = attack_target_mask(obs, obs_tensors)        # enemy ∪ neutral, non-comet
    proximity = min_distance_to_targets(cache, source_mask, attack_mask, max_k=K_eta)
    pref = -proximity + float(config.prod_target_bonus) * prod.to(proximity.dtype)
    pref = torch.where(attack_mask, pref, torch.full_like(pref, float("-inf")))
    atk_idx, atk_exists = _candidate_indices(pref, attack_mask, n_attack)

    chunks_idx, chunks_exists = [atk_idx], [atk_exists]

    if R > 0:
        flip_mask, urgency = friendly_flip_targets(obs, garrison_status, H=H, prod=prod)
        def_idx, def_exists = _candidate_indices(urgency, flip_mask, R)
        chunks_idx.append(def_idx)
        chunks_exists.append(def_exists)

    if bool(config.enable_comet_capture) and comet_remaining is not None:
        comet = is_comet_planet(obs_tensors, P, device)
        if comet is not None:
            need = float(config.comet_capture_min_hold) + 1.0   # at least eta=1 + hold
            cmask = (
                comet & obs.alive & (obs.is_enemy | obs.is_neutral)
                & (comet_remaining >= need)
            )
            if bool(cmask.any()):
                cprox = min_distance_to_targets(cache, source_mask, cmask, max_k=K_eta)
                cpref = torch.where(cmask, -cprox, torch.full_like(cprox, float("-inf")))
                c_idx, c_exists = _candidate_indices(
                    cpref, cmask, max(1, int(config.max_comet_targets))
                )
                chunks_idx.append(c_idx)
                chunks_exists.append(c_exists)

    return torch.cat(chunks_idx, dim=0), torch.cat(chunks_exists, dim=0)


# ---------------------------------------------------------------------------
# Wave planner
# ---------------------------------------------------------------------------

def plan_lite_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    comet_remaining: Tensor | None = None,
    pressure: Tensor | None = None,
):
    """Two-size + pair-strike single/double-source attack planner + regroup."""
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)

    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))

    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)

    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist_v7(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
        comet_remaining=comet_remaining,
    )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)

    # [v7-5] pressure-aware protection window: frontline sources protect the
    # full H, quiet rear sources only drain_min_horizon. safe_drain's
    # `turn_grid <= H_eff` comparison broadcasts a [S, 1] H_eff cleanly.
    if bool(config.enable_adaptive_drain) and pressure is not None:
        pmax = pressure.max().clamp(min=1e-6)
        pnorm = (pressure / pmax).clamp(0.0, 1.0)[source_idx.clamp(0, P - 1)]    # [S]
        h_min = float(min(int(config.drain_min_horizon), H))
        H_eff = (h_min + (float(H) - h_min) * pnorm).to(dtype).view(S, 1)
    else:
        H_eff = torch.full((), float(H), dtype=dtype, device=device)

    drain = safe_drain(
        garrison_status, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )

    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)

    # [v7-1] reactive-reinforcement margin folded into the capture floor.
    margin = None
    if bool(config.enable_reinforce_margin):
        margin = reactive_reinforcement_margin(
            obs, cache, target_idx=target_idx, K=K_eta, player_id=pid,
            frac=float(config.reinforce_margin_frac),
            eta_free=float(config.reinforce_eta_free),
            eta_scale=float(config.reinforce_eta_scale),
        )
    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid, reinforcement=margin,
    )
    K = int(floor.shape[-1])

    def _aim_for(sizes_st: Tensor):
        active = reachable_mask(
            movement, source_idx=source_idx, target_idx=target_idx,
            fleet_sizes=sizes_st.unsqueeze(-1), eta_cap=eta_cap,
        ).squeeze(-1)
        aim = intercept_angle(
            movement,
            source_idx.unsqueeze(1),
            target_idx.unsqueeze(0),
            sizes_st,
            active=active,
        )
        eta = aim["eta"]
        viable = aim["viable"] & (eta <= eta_cap.view(1, T))
        return aim["angle"], eta, viable

    def _floor_at(eta: Tensor) -> Tensor:
        if K > 0:
            k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
            return floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
        return torch.ones(S, T, dtype=dtype, device=device)

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    base_ok = src_neq_tgt & source_exists.view(S, 1) & target_exists.view(1, T)
    drain_int = drain.view(S, 1).expand(S, T).floor()

    # Option A: full safe-drain wave
    angle_a, eta_a, viable_a = _aim_for(drain_int)
    floor_a = _floor_at(eta_a)
    valid_a = viable_a & (drain_int >= floor_a) & (drain_int >= 1.0) & base_ok
    options = [(drain_int, angle_a, eta_a, valid_a)]

    # Option B: floor-matched wave (right-sized capture)
    if bool(config.enable_floor_sized_fleets):
        pad = float(config.floor_pad_ships)
        frac = 1.0 + float(config.floor_pad_frac)
        size_b = torch.minimum(drain_int, (floor_a * frac + pad).ceil())
        _, eta_b0, _ = _aim_for(size_b)
        floor_b0 = _floor_at(eta_b0)
        size_b = torch.minimum(
            drain_int, torch.maximum(size_b, (floor_b0 * frac + pad).ceil())
        ).floor()
        angle_b, eta_b, viable_b = _aim_for(size_b)
        floor_b = _floor_at(eta_b)
        valid_b = (
            viable_b & (size_b >= floor_b) & (size_b >= 1.0) & base_ok
            & (size_b < drain_int)
            & ~target_is_mine.view(1, T)
        )
        options.append((size_b, angle_b, eta_b, valid_b))

    # ── pack candidates on a unified contributor axis L = 2 ─────────────────
    # Single-source options occupy slot 0 with slot 1 inactive; [v7-2] pair
    # candidates use both slots. _greedy_select, make_launch_set and the
    # exact flow scorer are all already L-aware.
    L = 2
    short_range = torch.arange(T, device=device)
    p_src, p_send, p_ang, p_eta, p_val, p_short = [], [], [], [], [], []
    _act: list[Tensor] = []

    def _push_single(sizes_o, angle_o, eta_o, valid_o):
        Ci = S * T
        src1 = source_idx.view(S, 1).expand(S, T).reshape(Ci, 1)
        send1 = torch.where(valid_o, sizes_o, torch.zeros_like(sizes_o)).reshape(Ci, 1)
        ang1 = angle_o.reshape(Ci, 1)
        eta1 = torch.where(valid_o, eta_o, torch.ones_like(eta_o)).reshape(Ci, 1)
        act1 = valid_o.reshape(Ci, 1)
        zsend = torch.zeros(Ci, 1, dtype=dtype, device=device)
        p_src.append(torch.cat([src1, src1], dim=1))
        p_send.append(torch.cat([send1, zsend], dim=1))
        p_ang.append(torch.cat([ang1, zsend], dim=1))
        p_eta.append(torch.cat([eta1, torch.ones(Ci, 1, dtype=dtype, device=device)], dim=1))
        p_val.append(valid_o.reshape(Ci))
        _act.append(torch.cat([act1, torch.zeros(Ci, 1, dtype=torch.bool, device=device)], dim=1))
        p_short.append(short_range.view(1, T).expand(S, T).reshape(Ci))

    for sizes_o, angle_o, eta_o, valid_o in options:
        _push_single(sizes_o, angle_o, eta_o, valid_o)

    # [v7-2] one pair candidate per target: the two strongest viable sources
    # split a floor-sized capture. Only generated where the floor exceeds
    # what any single source can deliver — pairs add capability, they do not
    # duplicate the single-source candidates.
    if bool(config.enable_pair_strikes) and S >= 2:
        rank = torch.where(
            viable_a & base_ok & ~target_is_mine.view(1, T),
            drain_int, torch.full_like(drain_int, float("-inf")),
        )                                                                  # [S, T]
        order = _stable_topk_indices(rank.transpose(0, 1), 2)             # [T, 2] into S
        top_val = rank.transpose(0, 1).gather(1, order)                    # [T, 2]
        pair_exists = (
            torch.isfinite(top_val).all(dim=1)
            & (order[:, 0] != order[:, 1])
            & target_exists & ~target_is_mine
        )                                                                  # [T]
        s_slot = source_idx[order.clamp(0, S - 1)]                         # [T, 2] planet slots
        d12 = drain.floor()[order.clamp(0, S - 1)]                         # [T, 2]

        # floor estimate at the later of the two single-wave ETAs
        eta_aT = eta_a.transpose(0, 1)                                     # [T, S]
        e12 = eta_aT.gather(1, order.clamp(0, S - 1))                      # [T, 2]
        k_est = e12.max(dim=1).values.clamp(min=1.0, max=float(max(K, 1)))
        if K > 0:
            k_idx = (k_est.ceil().long() - 1).clamp(0, K - 1)
            floor_est = floor.gather(1, k_idx.view(T, 1)).squeeze(1)       # [T]
        else:
            floor_est = torch.ones(T, dtype=dtype, device=device)

        pad = 1.0 + float(config.floor_pad_frac)
        need = (floor_est * pad + float(config.pair_pad_ships)).ceil()     # [T]
        pair_needed = floor_est > d12.max(dim=1).values                    # singles can't
        c1 = torch.minimum(d12[:, 0], (need * 0.5).ceil() + 1.0).clamp(min=0.0)
        c2 = torch.minimum(d12[:, 1], (need - c1).clamp(min=1.0)).clamp(min=0.0)
        sizes_p = torch.stack([c1, c2], dim=1).floor()                     # [T, 2]

        pre = (
            pair_exists & pair_needed & (sizes_p >= 1.0).all(dim=1)
        ).view(T, 1).expand(T, 2)
        aim_p = intercept_angle(
            movement, s_slot, target_idx.view(T, 1), sizes_p, active=pre,
        )
        eta_p = aim_p["eta"]                                               # [T, 2]
        viable_p = aim_p["viable"] & (eta_p <= eta_cap.view(T, 1))
        k_pair = eta_p.max(dim=1).values.clamp(min=1.0, max=float(max(K, 1)))
        if K > 0:
            kp_idx = (torch.nan_to_num(k_pair, nan=1.0, posinf=float(K))
                      .ceil().long() - 1).clamp(0, K - 1)
            floor_pair = floor.gather(1, kp_idx.view(T, 1)).squeeze(1)     # [T]
        else:
            floor_pair = torch.ones(T, dtype=dtype, device=device)
        valid_p = (
            pair_exists & pair_needed
            & viable_p.all(dim=1)
            & (sizes_p >= 1.0).all(dim=1)
            & (sizes_p.sum(dim=1) >= floor_pair)
        )                                                                  # [T]
        act_p = valid_p.view(T, 1).expand(T, 2)
        p_src.append(s_slot)
        p_send.append(torch.where(act_p, sizes_p, torch.zeros_like(sizes_p)))
        p_ang.append(torch.where(act_p, aim_p["angle"],
                                 torch.zeros_like(aim_p["angle"])))
        p_eta.append(torch.where(act_p & torch.isfinite(eta_p), eta_p,
                                 torch.ones_like(eta_p)))
        p_val.append(valid_p)
        _act.append(act_p)
        p_short.append(short_range)

    cand_src = torch.cat(p_src, dim=0)
    cand_send = torch.cat(p_send, dim=0)
    cand_angle = torch.cat(p_ang, dim=0)
    cand_eta = torch.cat(p_eta, dim=0)
    cand_valid = torch.cat(p_val, dim=0)
    cand_active = torch.cat(_act, dim=0)
    cand_tgt_short = torch.cat(p_short, dim=0)
    cand_tgt_slot = target_idx[cand_tgt_short]
    C = int(cand_valid.shape[0])
    cand_is_def = target_is_mine[cand_tgt_short]

    # comet gating: never invest in a body that leaves before eta + hold.
    # [v7-3] capturable comets use the stricter comet_capture_min_hold so the
    # capture has time to pay back; v6's comet_min_hold still guards the rest.
    if comet_remaining is not None and bool(torch.isfinite(comet_remaining).any()):
        rem_c = comet_remaining[cand_tgt_slot.clamp(0, P - 1)]              # [C]
        is_comet_t = torch.isfinite(rem_c)
        eta_eff = torch.where(cand_active, cand_eta,
                              torch.full_like(cand_eta, float("-inf"))).max(dim=-1).values
        eta_eff = torch.nan_to_num(eta_eff, nan=1.0, neginf=1.0)
        hold = torch.where(
            cand_is_def,
            torch.full_like(rem_c, float(config.comet_min_hold)),
            torch.full_like(rem_c, float(
                max(config.comet_min_hold, config.comet_capture_min_hold)
                if bool(config.enable_comet_capture) else config.comet_min_hold
            )),
        )
        too_late = (eta_eff + hold) > rem_c
        cand_valid = cand_valid & ~(is_comet_t & too_late)

    cand_active = cand_active & cand_valid.unsqueeze(-1)

    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.view(C, 1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active,
        player_id=pid,
    )
    score = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launches, player_id=pid,
    )
    if int(player_count) >= 4 and (
        float(config.ffa_leader_attack_bonus) > 0.0
        or float(config.ffa_target_prod_bonus) > 0.0
    ):
        owner = obs.owner_abs.to(torch.long)
        owner_valid = (owner >= 0) & (owner < int(player_count)) & obs.alive
        owner_idx = owner.clamp(min=0, max=max(int(player_count) - 1, 0))
        prod_by_owner = torch.zeros(int(player_count), dtype=dtype, device=device)
        ships_by_owner = torch.zeros(int(player_count), dtype=dtype, device=device)
        prod_by_owner.scatter_add_(0, owner_idx, torch.where(owner_valid, prod.to(dtype), torch.zeros_like(prod.to(dtype))))
        ships_by_owner.scatter_add_(0, owner_idx, torch.where(owner_valid, obs.ships.to(dtype), torch.zeros_like(obs.ships.to(dtype))))
        strength = prod_by_owner + 0.025 * ships_by_owner
        my_strength = strength[pid].detach()

        target_owner = owner[target_idx.clamp(0, P - 1)].clamp(min=0, max=max(int(player_count) - 1, 0))
        target_owned_enemy = (
            target_exists
            & obs.is_enemy[target_idx.clamp(0, P - 1)]
            & (obs.owner_abs[target_idx.clamp(0, P - 1)] >= 0)
        )
        owner_strength = strength[target_owner]
        leader_delta = (owner_strength - my_strength).clamp(min=0.0)
        target_bonus_short = torch.where(
            target_owned_enemy,
            float(config.ffa_leader_attack_bonus) * leader_delta
            + float(config.ffa_target_prod_bonus) * prod[target_idx.clamp(0, P - 1)].to(dtype),
            torch.zeros_like(owner_strength),
        )
        score = score + target_bonus_short[cand_tgt_short]
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
    )

    if not bool(config.enable_regroup):
        return wave_entries
    enemy_mass = pressure if pressure is not None else cheap_enemy_pressure(
        obs, cache, horizon=float(K_eta), player_id=pid
    )
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


# ---------------------------------------------------------------------------
# Per-turn pipeline
# ---------------------------------------------------------------------------

def run_turn(
    obs_tensors: dict,
    *,
    config: ProducerLiteConfig,
    player_count: int,
    memory,
    comet_info: dict | None = None,
) -> dict:
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)

    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=int(player_count)),
        cached_movement=getattr(memory, "movement", None),
    )
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]

    comet_remaining = None
    if comet_info:
        try:
            ids = obs_tensors["planets"][..., 0].reshape(-1).long()
            if int(ids.numel()) == P:
                rem = torch.full((P,), float("inf"), dtype=obs.ships.dtype, device=device)
                for cid, r in comet_info.items():
                    hit = (ids == int(cid)).nonzero(as_tuple=True)[0]
                    if int(hit.numel()) > 0:
                        rem[int(hit[0])] = float(r)
                comet_remaining = rem
        except Exception:
            comet_remaining = None

    K_eta = max(1, min(int(config.horizon), H))
    pressure = cheap_enemy_pressure(
        obs, cache, horizon=float(K_eta), player_id=int(obs.player_id)
    )

    entries = plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=config, player_count=int(player_count),
        comet_remaining=comet_remaining, pressure=pressure,
    )
    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement, launches=launches, owner_id=int(obs.player_id),
        obs_tensors=obs_tensors,
    )
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)


# ── 4P FFA preset — only knobs that differ from 2P ─────────────────────────
CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=12,
    max_sources_per_lane=7,
    max_offensive_targets=9,
    max_defensive_targets=3,
    roi_threshold=1.45,
    min_ships_to_launch=4.0,
    max_regroup_time=5.0,
    max_regroup_targets_per_source=9,
    ffa_leader_attack_bonus=0.08,
    ffa_target_prod_bonus=0.15,
    # v7 in 4P: keep the margin slightly leaner (three rivals can't all
    # afford to reinforce one planet), comet capture a touch stricter.
    reinforce_margin_frac=0.25,
    comet_capture_min_hold=12.0,
)


def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict, comet_info: dict | None = None):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
            comet_info=comet_info,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


# ── Comet utilities (v6, unchanged) ─────────────────────────────────────────

_SUN_X, _SUN_Y, _SUN_R = 50.0, 50.0, 10.0


def _oget(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _parse_comet_remaining(obs) -> dict:
    out: dict = {}
    try:
        groups = _oget(obs, "comets", None) or []
        for g in groups:
            if isinstance(g, dict):
                pids = g.get("planet_ids") or []
                paths = g.get("paths") or []
                idx = int(g.get("path_index", 0) or 0)
            else:
                pids = getattr(g, "planet_ids", None) or []
                paths = getattr(g, "paths", None) or []
                idx = int(getattr(g, "path_index", 0) or 0)
            for i, cid in enumerate(pids):
                path = paths[i] if i < len(paths) else (paths[0] if len(paths) else None)
                if path is None:
                    continue
                out[int(cid)] = max(0, int(len(path)) - 1 - idx)
    except Exception:
        return {}
    return out


def _segment_clears_sun(x0, y0, x1, y1, margin: float = 1.5) -> bool:
    dx, dy = x1 - x0, y1 - y0
    l2 = dx * dx + dy * dy
    if l2 <= 1e-9:
        return math.hypot(x0 - _SUN_X, y0 - _SUN_Y) > _SUN_R + margin
    t = max(0.0, min(1.0, ((_SUN_X - x0) * dx + (_SUN_Y - y0) * dy) / l2))
    cx, cy = x0 + t * dx, y0 + t * dy
    return math.hypot(cx - _SUN_X, cy - _SUN_Y) > _SUN_R + margin


def _is_static_planet(p) -> bool:
    return math.hypot(float(p[2]) - 50.0, float(p[3]) - 50.0) + float(p[4]) >= 49.999


def _comet_evac_moves(obs, player_id: int, moves, remaining: dict, evac_steps: int):
    """Evacuate ships from comets about to leave the board."""
    try:
        base = [list(m) for m in (moves or [])]
        if not remaining:
            return base
        planets = _oget(obs, "planets", None) or []
        comet_ids = set(int(c) for c in (_oget(obs, "comet_planet_ids", None) or []))
        by_id = {int(p[0]): p for p in planets}
        committed: dict = {}
        for m in base:
            committed[int(m[0])] = committed.get(int(m[0]), 0) + int(m[2])

        own_static = [p for p in planets
                      if int(p[1]) == int(player_id) and int(p[0]) not in comet_ids
                      and _is_static_planet(p)]
        own_orbit = [p for p in planets
                     if int(p[1]) == int(player_id) and int(p[0]) not in comet_ids
                     and not _is_static_planet(p)]
        others = [p for p in planets
                  if int(p[0]) not in comet_ids and int(p[1]) != int(player_id)]

        for cid, rem in remaining.items():
            if int(rem) > int(evac_steps):
                continue
            p = by_id.get(int(cid))
            if p is None or int(p[1]) != int(player_id):
                continue
            avail = int(p[5]) - committed.get(int(cid), 0)
            if avail < 1:
                continue
            px, py = float(p[2]), float(p[3])
            best = None
            pools = (
                sorted(own_static, key=lambda q: (q[2] - px) ** 2 + (q[3] - py) ** 2),
                sorted(own_orbit, key=lambda q: (q[2] - px) ** 2 + (q[3] - py) ** 2),
                sorted(others, key=lambda q: (float(q[5]) >= avail,
                                              (q[2] - px) ** 2 + (q[3] - py) ** 2)),
            )
            for pool in pools:
                for q in pool:
                    if int(q[0]) == int(cid):
                        continue
                    if _segment_clears_sun(px, py, float(q[2]), float(q[3])):
                        best = q
                        break
                if best is not None:
                    break
            if best is None:
                continue
            ang = math.atan2(float(best[3]) - py, float(best[2]) - px)
            base.append([int(cid), float(ang), int(avail)])
            committed[int(cid)] = committed.get(int(cid), 0) + int(avail)
        return base
    except Exception:
        return moves


# ── Entry point ─────────────────────────────────────────────────────────────

def agent(obs):
    """Single-observation entry point for local play and Kaggle."""
    player = _oget(obs, "player", 0)
    player_id = int(player if player is not None else 0)
    comet_info = _parse_comet_remaining(obs)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors, comet_info=comet_info)
    moves = sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
    cfg = _config_for(_RUNTIME.memory.cached_player_count or 2)
    return _comet_evac_moves(obs, player_id, moves, comet_info, cfg.comet_evac_steps)
