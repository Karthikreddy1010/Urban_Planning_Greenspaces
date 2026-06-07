"""
env.py - Gym-compatible RL environment for urban green-space placement.

State   = feature tensor (H, W, C) - updated as placements occur
Action  = index into the flat list of *feasible* grid cells
Episode = sequence of N placements
"""

import numpy as np
from typing import Tuple, Dict, Optional

from config import GRID_CFG, ENV_CFG, REWARD_CFG


class GreenSpaceEnv:
    """
    Reinforcement-learning environment for placing green spaces in Newark.

    The agent selects grid cells to convert into green space.  Each placement
    triggers physical effects (LST cooling, PM2.5 reduction) modelled by
    spatial Gaussian decay.  The reward balances cooling, pollution reduction,
    land-use cost, and spatial clustering.
    """

    # Channel indices (must match GRID_CFG.channels order)
    CH_POP = 0
    CH_LST = 1
    CH_PM25 = 2
    CH_IMP = 3
    CH_GREEN = 4
    CH_TRAFFIC = 5
    CH_FEAS = 6

    def __init__(self,
                 feature_tensor: np.ndarray,
                 valid_mask: np.ndarray,
                 norm_params: dict,
                 grid_meta: dict,
                 env_cfg=None,
                 reward_cfg=None):
        """
        Parameters
        ----------
        feature_tensor : ndarray (H, W, C)
            Normalised feature tensor (will be copied each reset).
        valid_mask : ndarray (H, W) bool
            True for cells inside Newark boundary.
        norm_params : dict
            Per-channel {min, max} from preprocessing (for un-normalising).
        grid_meta : dict
            Keys: 'resolution', 'rows', 'cols'.
        """
        self.cfg = env_cfg or ENV_CFG
        self.rcfg = reward_cfg or REWARD_CFG

        self._base_tensor = feature_tensor.copy()
        self._valid_mask = valid_mask.copy()
        self._norm_params = norm_params
        self._resolution = grid_meta["resolution"]

        self.H, self.W, self.C = feature_tensor.shape

        # Pre-compute distance kernels for Gaussian decay
        self._cooling_kernel = self._make_gaussian_kernel(
            self.cfg.cooling_sigma, self.cfg.cooling_magnitude
        )
        self._pm25_kernel = self._make_gaussian_kernel(
            self.cfg.pm25_sigma, self.cfg.pm25_magnitude
        )

        # State variables (set on reset)
        self.state: Optional[np.ndarray] = None
        self.feasible_mask: Optional[np.ndarray] = None
        self.step_count: int = 0
        self.placed_cells: list = []
        self.done: bool = True

        # Per-episode accumulators
        self._total_lst_reduction = 0.0
        self._total_pm25_reduction = 0.0

    # ── Public API ───────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset environment to initial state.  Returns initial state tensor."""
        self.state = self._base_tensor.copy()
        self.feasible_mask = (self.state[:, :, self.CH_FEAS] > 0.5).astype(np.float32)
        # Also require the cell to be inside the boundary
        self.feasible_mask *= self._valid_mask.astype(np.float32)
        self.step_count = 0
        self.placed_cells = []
        self.done = False
        self._total_lst_reduction = 0.0
        self._total_pm25_reduction = 0.0
        return self.state.copy()

    def get_feasible_actions(self) -> np.ndarray:
        """
        Return a 1-D boolean array of length (H * W) indicating which
        flat indices are feasible actions.
        """
        return self.feasible_mask.flatten().astype(bool)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute an action (flat grid index) and return
        (next_state, reward, done, info).

        Parameters
        ----------
        action : int
            Flat index into (H * W) grid.  Must be a feasible cell.
        """
        assert not self.done, "Episode is done - call reset()."

        r = action // self.W
        c = action % self.W

        # Validate feasibility
        if self.feasible_mask[r, c] < 0.5:
            # Invalid action - return negative reward, no state change
            return self.state.copy(), -1.0, False, {"invalid_action": True}

        # ── Apply placement effects ──────────────────────────────────────
        cooling = self._apply_cooling(r, c)
        pm25_red = self._apply_pm25_reduction(r, c)

        # Update green-space channel
        self.state[r, c, self.CH_GREEN] = 1.0
        # Update impervious channel (placed cell becomes non-impervious)
        self.state[r, c, self.CH_IMP] = 0.0
        # Update feasibility (remove placed cell)
        self.state[r, c, self.CH_FEAS] = 0.0
        self.feasible_mask[r, c] = 0.0

        self.step_count += 1
        self.placed_cells.append((r, c))
        self._total_lst_reduction += cooling
        self._total_pm25_reduction += pm25_red

        # ── Compute reward ───────────────────────────────────────────────
        reward, reward_info = self._compute_reward(r, c, cooling, pm25_red)

        # ── Check termination ────────────────────────────────────────────
        self.done = (
            self.step_count >= self.cfg.max_placements or
            self.feasible_mask.sum() == 0
        )

        info = {
            "step": self.step_count,
            "placed_cell": (r, c),
            "cooling": cooling,
            "pm25_reduction": pm25_red,
            "total_lst_reduction": self._total_lst_reduction,
            "total_pm25_reduction": self._total_pm25_reduction,
            "n_feasible_remaining": int(self.feasible_mask.sum()),
            "invalid_action": False,
            **reward_info,
        }

        return self.state.copy(), reward, self.done, info

    @property
    def action_space_size(self) -> int:
        return self.H * self.W

    # ── Private: physical effects ────────────────────────────────────────

    def _apply_cooling(self, r: int, c: int) -> float:
        """
        Apply Gaussian LST cooling centred at (r, c).
        Returns the total cooling (sum of ΔT across affected cells).
        """
        kernel = self._cooling_kernel
        kh, kw = kernel.shape
        kr, kc = kh // 2, kw // 2

        total_cooling = 0.0
        for dr in range(-kr, kr + 1):
            for dc in range(-kc, kc + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W:
                    if self._valid_mask[nr, nc]:
                        delta = kernel[dr + kr, dc + kc]
                        # Normalise the delta relative to LST range
                        lst_range = (self._norm_params["lst"]["max"] -
                                     self._norm_params["lst"]["min"])
                        if lst_range > 0:
                            norm_delta = delta / lst_range
                        else:
                            norm_delta = 0.0
                        old_val = self.state[nr, nc, self.CH_LST]
                        new_val = max(0.0, old_val - norm_delta)
                        self.state[nr, nc, self.CH_LST] = new_val
                        total_cooling += delta

        return total_cooling

    def _apply_pm25_reduction(self, r: int, c: int) -> float:
        """
        Apply Gaussian PM2.5 reduction centred at (r, c).
        Returns the total PM2.5 reduction.
        """
        kernel = self._pm25_kernel
        kh, kw = kernel.shape
        kr, kc = kh // 2, kw // 2

        total_reduction = 0.0
        for dr in range(-kr, kr + 1):
            for dc in range(-kc, kc + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W:
                    if self._valid_mask[nr, nc]:
                        delta = kernel[dr + kr, dc + kc]
                        pm_range = (self._norm_params["pm25"]["max"] -
                                    self._norm_params["pm25"]["min"])
                        if pm_range > 0:
                            norm_delta = delta / pm_range
                        else:
                            norm_delta = 0.0
                        old_val = self.state[nr, nc, self.CH_PM25]
                        new_val = max(0.0, old_val - norm_delta)
                        self.state[nr, nc, self.CH_PM25] = new_val
                        total_reduction += delta

        return total_reduction

    # ── Private: reward ──────────────────────────────────────────────────

    def _compute_reward(self, r: int, c: int,
                        cooling: float, pm25_red: float) -> Tuple[float, dict]:
        """
        R = α·Cooling + β·PollutionReduction − γ·Cost − δ·SpatialScatter

        All components are normalised to roughly [0, 1] before weighting.
        """
        # Normalise cooling (divide by max possible single-step cooling)
        max_cooling = self.cfg.cooling_magnitude * 9  # rough 3×3 core
        norm_cooling = min(cooling / max(max_cooling, 1e-8), 1.0)

        # Normalise PM2.5 reduction
        max_pm25 = self.cfg.pm25_magnitude * 5
        norm_pm25 = min(pm25_red / max(max_pm25, 1e-8), 1.0)

        # Cost: based on impervious surface at placement (more impervious = more costly)
        # Use the ORIGINAL impervious value (before we zeroed it)
        cost = self._base_tensor[r, c, self.CH_IMP]  # already in [0, 1]

        # Spatial scatter penalty: fraction of placed neighbours within radius
        scatter = self._compute_scatter_penalty(r, c)

        # Weighted sum
        reward = (
            self.rcfg.alpha * norm_cooling +
            self.rcfg.beta * norm_pm25 -
            self.rcfg.gamma * cost -
            self.rcfg.delta * scatter
        )

        info = {
            "reward_cooling": self.rcfg.alpha * norm_cooling,
            "reward_pm25": self.rcfg.beta * norm_pm25,
            "reward_cost": -self.rcfg.gamma * cost,
            "reward_scatter": -self.rcfg.delta * scatter,
        }
        return reward, info

    def _compute_scatter_penalty(self, r: int, c: int) -> float:
        """
        Spatial scatter penalty: 1.0 if the new placement has NO nearby
        previously placed cells, 0.0 if it's adjacent to existing placements.
        Encourages cluster formation.
        """
        if len(self.placed_cells) <= 1:
            return 0.0  # No penalty for first placement

        radius_cells = int(self.cfg.scatter_radius / self._resolution)
        min_dist = float("inf")

        for pr, pc in self.placed_cells[:-1]:  # exclude current
            dist = np.sqrt((r - pr) ** 2 + (c - pc) ** 2)
            min_dist = min(min_dist, dist)

        if min_dist <= 1.5:  # adjacent
            return 0.0
        elif min_dist >= radius_cells:
            return 1.0
        else:
            return min_dist / radius_cells

    # ── Private: kernel construction ─────────────────────────────────────

    def _make_gaussian_kernel(self, sigma_m: float,
                              magnitude: float) -> np.ndarray:
        """
        Create a 2-D Gaussian kernel with physical-unit sigma.
        Kernel size is 3σ in each direction.
        """
        sigma_cells = sigma_m / self._resolution
        radius = int(np.ceil(3 * sigma_cells))
        size = 2 * radius + 1

        kernel = np.zeros((size, size), dtype=np.float32)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                d2 = (dr ** 2 + dc ** 2) * (self._resolution ** 2)
                kernel[dr + radius, dc + radius] = magnitude * np.exp(
                    -d2 / (2 * sigma_m ** 2)
                )
        return kernel

    # ── Utility ──────────────────────────────────────────────────────────

    def get_population_at(self, r: int, c: int) -> float:
        """Return normalised population density at (r, c)."""
        return self.state[r, c, self.CH_POP]

    def get_equity_score(self) -> float:
        """
        Mean normalised population density across all placements.
        Higher = more equitable (placements favour dense areas).
        """
        if len(self.placed_cells) == 0:
            return 0.0
        return np.mean([
            self._base_tensor[r, c, self.CH_POP]
            for r, c in self.placed_cells
        ])
