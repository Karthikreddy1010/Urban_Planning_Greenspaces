# Urban Green Space Placement via Constrained Reinforcement Learning

This project implements a complete, production-ready spatial optimization system based on a simplified FORGE (Feedback-driven Optimization for Green Infrastructure placement) methodology. It uses deep reinforcement learning (Constrained PPO) combined with Graph Neural Networks (GraphSAGE) to place urban green spaces (parks) in Newark, New Jersey, to optimize multi-objective environmental and social criteria.

---

## Objective

Train an RL agent to place a budget of **N = 20 green spaces** (represented as 100m × 100m grid cells) to:
1. **Reduce Land Surface Temperature (LST)** via local Gaussian-decay cooling.
2. **Reduce PM2.5 Pollution** via local Gaussian-decay deposition.
3. **Respect Land Feasibility** (avoid placing on water bodies, existing green spaces, or protected/preserved land).
4. **Prioritize Population Density (Equity)** via a Constrained Lagrangian Multiplier formulation that penalizes placements in low-population density areas.

---

## Project Structure

```
project/
├── config.py           # Hyperparameters, file paths, reward weights, and constants
├── data_loader.py      # Load raw GIS data (GeoTIFFs, Shapefiles, GeoJSONs, CSVs)
├── preprocess.py       # Grid generation (100m), rasterization of features, and normalization
├── graph_builder.py    # Construction of GNN graph data (8-direction spatial + road network edges)
├── env.py              # Gym-compatible RL environment simulating cooling and pollution reduction
├── model.py            # Neural network: CNN Encoder + GraphSAGE + Masked Policy & Value heads
├── ppo.py              # Constrained PPO algorithm with Lagrangian equity updates
├── train.py            # Training loop with metric history tracking and visualization
├── evaluate.py         # Evaluation of trained agent against Random and Greedy baselines
├── requirements.txt    # Python library dependencies
└── README.md           # Instructions and project documentation
```

---

## Installation & Setup

1. **Python Environment**: Ensure you have Python 3.8+ installed.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: On Windows, shapefile libraries (Fiona, Rasterio) and PyTorch Geometric may require matching versions. You can install pre-compiled wheels if standard `pip` fails.*

---

## Data Layout

The raw datasets should be located in a folder named `Urban_planning_data` adjacent to the `project/` directory:
```
Urban_planning_data/
├── Landsat/                                   # Landsat 8/9 ST_B10 and SR bands
├── NLCD_Landcover/                            # Land cover GeoTIFFs
├── NLCD_Imprevioussurface/                    # Impervious surface GeoTIFFs
├── AADT_traffic/                              # Traffic shapefile
├── preserved/                                 # Green acres program shapefile
├── Open_Space_-4571155014818248037/           # Open Space shapefile
├── newark_boundary.geojson                    # Boundary polygon of Newark
├── road_network_newark.geojson                # Newark road street network
├── essex_pollution_points_AQS.geojson         # AQS PM2.5 monitoring points
└── newark_demographics.csv                    # Demographics (population) CSV
```

---

## How to Run

### 1. Training the RL Agent
To train the agent with default configurations (500 episodes, 20 placements budget):
```bash
python train.py
```
**Custom Options:**
- `--episodes 200`: Train for 200 episodes.
- `--budget 15`: Place 15 green spaces.
- `--equity-threshold 0.6`: Require higher population density weight.
- `--no-cache`: Force rebuilding the spatial grid from raw GIS files instead of loading the cache.
- `--data-dir "/path/to/data"`: Specify a custom datasets folder.

During training, metrics are saved to `results/` along with training plots (`results/training_curves.png`) and periodic model checkpoints (`model_ep_*.pt`).

### 2. Evaluation and Baselines Comparison
To compare the trained agent against **Random** and **Greedy** baselines:
```bash
python evaluate.py --model-path results/model_final.pt
```

This will run evaluations and output:
- **Comparison Table**: printed in the terminal showing total reward, cooling, PM2.5, and equity.
- **`results/evaluation_metrics.csv`**: tabular results of the runs.
- **`results/eval_placement_comparison.png`**: map comparing placed locations.
- **`results/eval_lst_comparison.png`**: heatmap comparing LST cooling spread.
- **`results/eval_pm25_comparison.png`**: map showing PM2.5 reduction fields.
- **`results/eval_metrics_comparison.png`**: bar charts comparing performance.

---

## Verification Tests

To verify that the implementation works correctly on your system, you can run the following test commands:

```bash
# Verify data loading (no errors, correct shapes)
python -c "from data_loader import load_all; load_all(); print('✓ Data loaders OK')"

# Verify preprocessing pipeline and cache generation
python -c "from preprocess import preprocess_all; t = preprocess_all(use_cache=False); print(f'✓ Feature tensor OK. Shape: {t[1].shape}')"

# Run a quick training check (5 episodes)
python train.py --episodes 5 --budget 5

# Run a quick evaluation check
python evaluate.py --model-path results/model_final.pt --budget 5
```
