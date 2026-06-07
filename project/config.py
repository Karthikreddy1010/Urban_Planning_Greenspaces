"""
config.py - Central configuration for the Urban Green Space RL system.

All file paths, hyperparameters, reward weights, and training parameters
are defined here so every other module imports from a single source.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

# ── Base paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "Urban_planning_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Data file paths ──────────────────────────────────────────────────────────
@dataclass
class DataPaths:
    """Paths to every raw dataset, relative to DATA_DIR."""
    newark_boundary: str = os.path.join(DATA_DIR, "newark_boundary.geojson")
    road_network: str = os.path.join(DATA_DIR, "road_network_newark.geojson")
    pollution_points: str = os.path.join(DATA_DIR, "essex_pollution_points_AQS.geojson")
    aqs_annual_csv: str = os.path.join(DATA_DIR, "essex_annual_AQS_combined.csv")
    demographics_csv: str = os.path.join(DATA_DIR, "newark_demographics.csv")
    weather_csv: str = os.path.join(DATA_DIR, "newark_weather_2000_2025.csv")
    newark_parcels_geojson: str = os.path.join(DATA_DIR, "Newark_Parcels_3631818904235536078.geojson")
    parcels_shp: str = os.path.join(DATA_DIR, "parcels_shp_dbf_Essex", "EssexCountyParcels.shp")
    open_space_shp: str = os.path.join(
        DATA_DIR,
        "Open_Space_-4571155014818248037",
        "Open_Space.shp",
    )
    preserved_shp: str = os.path.join(
        DATA_DIR,
        "preserved",
        "Open_Space_Hidden_Gems_(Selected_Green_Acres_Program_Funded_Parkland)_in_New_Jersey.shp",
    )
    aadt_traffic_shp: str = os.path.join(DATA_DIR, "AADT_traffic", "Annual_Average_Daily_Traffic.shp")
    landsat_dir: str = os.path.join(DATA_DIR, "Landsat")
    nlcd_landcover_dir: str = os.path.join(DATA_DIR, "NLCD_Landcover")
    nlcd_impervious_dir: str = os.path.join(DATA_DIR, "NLCD_Imprevioussurface")
    soil_spatial_dir: str = os.path.join(DATA_DIR, "soil", "NJ013", "spatial")

    # Specific Landsat scene (most recent summer, low cloud)
    landsat_scene_id: str = "LC08_L2SP_013032_20240827_20240831_02_T1"
    # NLCD year to use
    nlcd_year: int = 2024


# ── CRS ───────────────────────────────────────────────────────────────────────
TARGET_CRS = "EPSG:32618"  # UTM Zone 18N - native CRS for Landsat data


# ── Grid parameters ───────────────────────────────────────────────────────────
@dataclass
class GridConfig:
    resolution: float = 100.0  # metres per cell
    # Feature channels in the tensor (order matters - index used throughout)
    channels: Tuple[str, ...] = (
        "population_density",
        "lst",
        "pm25",
        "impervious",
        "green_space",
        "traffic",
        "feasibility",
    )

    @property
    def n_channels(self) -> int:
        return len(self.channels)


# ── Landsat conversion parameters ────────────────────────────────────────────
@dataclass
class LandsatConfig:
    """
    Landsat 8 Collection 2 Level-2 Surface Temperature product parameters.
    ST_B10 DN values are scaled integers.  Physical temperature in Kelvin:
        T_K = DN * scale + offset
    """
    st_scale: float = 0.00341802
    st_offset: float = 149.0
    kelvin_to_celsius: float = -273.15
    # Surface reflectance scale for SR bands (B4, B5)
    sr_scale: float = 2.75e-05
    sr_offset: float = -0.2
    # Fill / nodata value in L2SP products
    fill_value: int = 0


# ── Reward function weights ──────────────────────────────────────────────────
@dataclass
class RewardConfig:
    alpha: float = 0.4    # cooling (LST reduction)
    beta: float = 0.3     # pollution reduction (PM2.5)
    gamma: float = 0.15   # cost penalty
    delta: float = 0.15   # spatial scatter penalty


# ── Environment parameters ───────────────────────────────────────────────────
@dataclass
class EnvConfig:
    max_placements: int = 20       # actions per episode (budget)
    cooling_sigma: float = 300.0   # Gaussian decay σ for LST cooling (metres)
    cooling_magnitude: float = 2.0 # peak LST reduction in degC at placement cell
    pm25_sigma: float = 200.0      # Gaussian decay σ for PM2.5 reduction (metres)
    pm25_magnitude: float = 1.5    # peak PM2.5 reduction (µg/m³) at placement
    scatter_radius: float = 500.0  # radius (m) for spatial scatter penalty


# ── PPO hyperparameters ──────────────────────────────────────────────────────
@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99          # discount factor
    gae_lambda: float = 0.95     # GAE λ
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4       # PPO epochs per rollout
    minibatch_size: int = 64
    # Lagrangian constraint parameters
    lagrangian_lr: float = 0.05
    equity_threshold: float = 0.5  # min normalised population density at placements


# ── Model architecture ───────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    cnn_channels: Tuple[int, ...] = (32, 64, 128)
    gnn_channels: Tuple[int, ...] = (128, 64, 32)
    value_hidden: int = 64


# ── Training parameters ─────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    num_episodes: int = 500
    log_interval: int = 10
    save_interval: int = 50
    seed: int = 42


# ── Convenience: default instances ──────────────────────────────────────────
DATA_PATHS = DataPaths()
GRID_CFG = GridConfig()
LANDSAT_CFG = LandsatConfig()
REWARD_CFG = RewardConfig()
ENV_CFG = EnvConfig()
PPO_CFG = PPOConfig()
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()
