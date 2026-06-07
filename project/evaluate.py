"""
evaluate.py - Evaluation and comparison of RL agent against baselines.

This module:
  1. Implements Random and Greedy baselines.
  2. Loads the trained RL agent and runs it deterministically.
  3. Evaluates all methods on total reward, LST cooling, PM2.5 reduction, and equity.
  4. Generates comparison plots: placement maps, LST heatmaps, PM2.5 heatmaps, and metrics bars.
  5. Saves comparison results to CSV and outputs a formatted table.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box

from config import (
    DATA_DIR, RESULTS_DIR, GRID_CFG, ENV_CFG, REWARD_CFG, TARGET_CRS
)
from preprocess import preprocess_all
from graph_builder import build_graph
from env import GreenSpaceEnv
from model import GreenSpacePolicyNet


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Green Space RL Agent")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Path to Newark GIS datasets folder")
    parser.add_argument("--model-path", type=str,
                        default=os.path.join(RESULTS_DIR, "model_final.pt"),
                        help="Path to the trained policy network model weights")
    parser.add_argument("--budget", type=int, default=ENV_CFG.max_placements,
                        help="Number of green spaces to place per episode")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force preprocessing from scratch (no cache)")
    parser.add_argument("--seed", type=int, default=100,
                        help="Random seed for evaluation runs")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Baselines ─────────────────────────────────────────────────────────────────

def run_random_baseline(env: GreenSpaceEnv, budget: int) -> dict:
    """Select budget cells uniformly at random from feasible cells."""
    env.reset()
    done = False
    episode_reward = 0.0
    
    while not done:
        feasible = env.get_feasible_actions()
        feasible_idx = np.where(feasible)[0]
        if len(feasible_idx) == 0:
            break
        action = np.random.choice(feasible_idx)
        state, reward, done, info = env.step(action)
        episode_reward += reward
        
    return {
        "reward": episode_reward,
        "cooling": env._total_lst_reduction,
        "pm25_red": env._total_pm25_reduction,
        "equity": env.get_equity_score(),
        "placed_cells": list(env.placed_cells),
        "final_state": env.state.copy()
    }


def run_greedy_baseline(env: GreenSpaceEnv, budget: int) -> dict:
    """
    Select cells greedily. At each step, evaluates a sub-sample of
    up to 1,000 feasible cells to find the one maximizing immediate reward.
    """
    env.reset()
    done = False
    episode_reward = 0.0
    
    while not done:
        feasible = env.get_feasible_actions()
        feasible_idx = np.where(feasible)[0]
        if len(feasible_idx) == 0:
            break
            
        # Sub-sample feasible cells if there are too many (speeds up evaluation)
        max_samples = 1000
        if len(feasible_idx) > max_samples:
            sampled_idx = np.random.choice(feasible_idx, max_samples, replace=False)
        else:
            sampled_idx = feasible_idx
            
        best_action = None
        best_reward = -float("inf")
        
        # Save env state
        saved_state = env.state.copy()
        saved_feasible_mask = env.feasible_mask.copy()
        saved_step_count = env.step_count
        saved_placed_cells = list(env.placed_cells)
        saved_done = env.done
        saved_total_lst = env._total_lst_reduction
        saved_total_pm25 = env._total_pm25_reduction
        
        for action in sampled_idx:
            _, reward, _, _ = env.step(action)
            if reward > best_reward:
                best_reward = reward
                best_action = action
                
            # Restore state
            env.state = saved_state.copy()
            env.feasible_mask = saved_feasible_mask.copy()
            env.step_count = saved_step_count
            env.placed_cells = list(saved_placed_cells)
            env.done = saved_done
            env._total_lst_reduction = saved_total_lst
            env._total_pm25_reduction = saved_total_pm25
            
        state, reward, done, info = env.step(best_action)
        episode_reward += reward
        
    return {
        "reward": episode_reward,
        "cooling": env._total_lst_reduction,
        "pm25_red": env._total_pm25_reduction,
        "equity": env.get_equity_score(),
        "placed_cells": list(env.placed_cells),
        "final_state": env.state.copy()
    }


def run_rl_agent(env: GreenSpaceEnv, model: GreenSpacePolicyNet, graph_data, device: str) -> dict:
    """Select cells using the trained RL agent policy (deterministically)."""
    env.reset()
    done = False
    episode_reward = 0.0
    
    node_rows = graph_data.node_rows.cpu().numpy()
    node_cols = graph_data.node_cols.cpu().numpy()
    
    model.eval()
    while not done:
        node_feasible = env.feasible_mask[node_rows, node_cols] > 0.5
        node_feasible_t = torch.tensor(node_feasible, dtype=torch.bool, device=device)
        state_t = torch.tensor(env.state, dtype=torch.float32, device=device)
        
        # Act deterministically
        action_node_id, log_prob, value = model.act(
            state_tensor=state_t,
            graph_data=graph_data,
            feasible_mask=node_feasible_t,
            deterministic=True
        )
        
        r = int(node_rows[action_node_id])
        c = int(node_cols[action_node_id])
        flat_action = r * env.W + c
        
        state, reward, done, info = env.step(flat_action)
        episode_reward += reward
        
    return {
        "reward": episode_reward,
        "cooling": env._total_lst_reduction,
        "pm25_red": env._total_pm25_reduction,
        "equity": env.get_equity_score(),
        "placed_cells": list(env.placed_cells),
        "final_state": env.state.copy()
    }


# ── Plotting Utilities ────────────────────────────────────────────────────────

def plot_spatial_placements(results: dict, grid_gdf: gpd.GeoDataFrame,
                             boundary_gdf: gpd.GeoDataFrame, filepath: str):
    """Plot maps of Newark with the 20 placed green spaces highlighted."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    boundary_union = boundary_gdf.geometry.union_all()
    res = grid_gdf.attrs["resolution"]
    
    methods = ["Random", "Greedy", "RL Agent"]
    keys = ["random", "greedy", "rl"]
    colors = ["royalblue", "darkorange", "forestgreen"]
    
    for idx, (method, key, col) in enumerate(zip(methods, keys, colors)):
        ax = axs[idx]
        boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.5, zorder=2)
        
        if key in results:
            placed = results[key]["placed_cells"]
            # Reconstruct polygons for placed cells
            placed_polys = []
            for r, c in placed:
                x0 = grid_gdf.attrs["origin_x"] + c * res
                y0 = grid_gdf.attrs["origin_y"] - (r + 1) * res
                placed_polys.append(box(x0, y0, x0 + res, y0 + res))
                
            placed_gdf = gpd.GeoDataFrame(geometry=placed_polys, crs=TARGET_CRS)
            placed_gdf.plot(ax=ax, color=col, edgecolor="darkgreen", label="New Parks", zorder=3)
            
        ax.set_title(f"{method} Placements")
        ax.axis("off")
        
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"[evaluate] Saved placement comparison map to {filepath}")


def plot_diff_heatmap(results: dict, base_state: np.ndarray, channel_idx: int,
                      title: str, cmap: str, filepath: str):
    """Plot the spatial reduction fields (before - after) for a specific channel."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    methods = ["Random", "Greedy", "RL Agent"]
    keys = ["random", "greedy", "rl"]
    
    # Calculate global min/max for shared colorbar
    diffs = {}
    vmax = -float("inf")
    for key in keys:
        if key in results:
            diff = base_state[:, :, channel_idx] - results[key]["final_state"][:, :, channel_idx]
            # Zero-out cells that are nan in original base state
            diff[np.isnan(base_state[:, :, channel_idx])] = np.nan
            diffs[key] = diff
            vmax = max(vmax, np.nanmax(diff))
            
    vmax = max(vmax, 1e-5)
    
    for idx, (method, key) in enumerate(zip(methods, keys)):
        ax = axs[idx]
        if key in diffs:
            im = ax.imshow(diffs[key], cmap=cmap, vmin=0, vmax=vmax)
            ax.set_title(f"{method}: {title}")
        ax.axis("off")
        
    # Shared colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=f"Reduction")
    
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved heatmap comparison to {filepath}")


def plot_metrics_bar(df: pd.DataFrame, filepath: str):
    """Plot comparative metrics bar chart."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ["reward", "cooling", "pm25_red", "equity"]
    titles = ["Total Reward", "LST Cooling (degC)", "PM2.5 Reduction (ug/m3)", "Equity Score"]
    colors = ["forestgreen", "crimson", "royalblue", "darkorange"]
    
    for idx, (metric, title, col) in enumerate(zip(metrics, titles, colors)):
        ax = axs[idx // 2, idx % 2]
        ax.bar(df["Method"], df[metric], color=col, width=0.4, edgecolor="black")
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--")
        
        # Annotate bars
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() * 1.01),
                        ha="center", va="bottom", fontsize=10)
            
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"[evaluate] Saved metrics bar chart to {filepath}")


def evaluate():
    args = parse_args()
    set_seed(args.seed)
    
    # ── 1. Data preprocessing ──────────────────────────────────────────────
    grid_gdf, feature_tensor, norm_params, valid_mask = preprocess_all(
        data_dir=args.data_dir,
        use_cache=not args.no_cache
    )
    
    # Save a copy of the base state tensor
    base_state = feature_tensor.copy()
    
    # Load boundary geometry for plotting
    boundary_gdf = gpd.read_file(os.path.join(args.data_dir, "newark_boundary.geojson"))
    if boundary_gdf.crs is None:
        boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
    boundary_gdf = boundary_gdf.to_crs(TARGET_CRS)
    
    grid_meta = {
        "resolution": grid_gdf.attrs["resolution"],
        "rows": grid_gdf.attrs["rows"],
        "cols": grid_gdf.attrs["cols"]
    }
    
    # Setup environments for each run
    env_random = GreenSpaceEnv(feature_tensor, valid_mask, norm_params, grid_meta)
    env_greedy = GreenSpaceEnv(feature_tensor, valid_mask, norm_params, grid_meta)
    env_rl = GreenSpaceEnv(feature_tensor, valid_mask, norm_params, grid_meta)
    
    results = {}
    
    # ── 2. Run Baselines ───────────────────────────────────────────────────
    print("[evaluate] Running Random Baseline (average of 10 runs)...")
    random_runs = []
    for r in range(10):
        random_runs.append(run_random_baseline(env_random, args.budget))
        
    results["random"] = {
        "reward": np.mean([x["reward"] for x in random_runs]),
        "cooling": np.mean([x["cooling"] for x in random_runs]),
        "pm25_red": np.mean([x["pm25_red"] for x in random_runs]),
        "equity": np.mean([x["equity"] for x in random_runs]),
        "placed_cells": random_runs[0]["placed_cells"], # Representative run
        "final_state": random_runs[0]["final_state"]
    }
    
    print("[evaluate] Running Greedy Baseline...")
    results["greedy"] = run_greedy_baseline(env_greedy, args.budget)
    
    # ── 3. Run Trained Agent ───────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if os.path.exists(args.model_path):
        print(f"[evaluate] Loading trained agent model from {args.model_path}...")
        graph_data = build_graph(grid_gdf, feature_tensor, valid_mask).to(device)
        
        model = GreenSpacePolicyNet(
            in_channels=GRID_CFG.n_channels,
            grid_rows=grid_meta["rows"],
            grid_cols=grid_meta["cols"]
        ).to(device)
        
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        print("[evaluate] Running RL Agent...")
        results["rl"] = run_rl_agent(env_rl, model, graph_data, device)
    else:
        print(f"[evaluate] [warn] Model checkpoint {args.model_path} not found. Skipping RL Agent evaluation.")
        
    # ── 4. Summarise Results ───────────────────────────────────────────────
    eval_rows = []
    for key in ["random", "greedy", "rl"]:
        if key in results:
            name = "Random Baseline" if key == "random" else "Greedy Baseline" if key == "greedy" else "RL Agent"
            eval_rows.append({
                "Method": name,
                "reward": results[key]["reward"],
                "cooling": results[key]["cooling"],
                "pm25_red": results[key]["pm25_red"],
                "equity": results[key]["equity"]
            })
            
    df_results = pd.DataFrame(eval_rows)
    
    # Print comparison table
    print("\n" + "=" * 65)
    print("EVALUATION PERFORMANCE COMPARISON")
    print("=" * 65)
    print(df_results.to_string(index=False, formatters={
        "reward": "{:.3f}".format,
        "cooling": "{:.2f}degC".format,
        "pm25_red": "{:.2f}ug/m3".format,
        "equity": "{:.3f}".format
    }))
    print("=" * 65)
    
    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "evaluation_metrics.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[evaluate] Saved metrics summary CSV to {csv_path}")
    
    # ── 5. Generate Plots ──────────────────────────────────────────────────
    print("\n[evaluate] Generating comparison plots...")
    
    # Placements map
    plot_spatial_placements(results, grid_gdf, boundary_gdf, os.path.join(RESULTS_DIR, "eval_placement_comparison.png"))
    
    # LST cooling heatmaps
    plot_diff_heatmap(results, base_state, env_random.CH_LST, "LST Cooling (degC)", "coolwarm", os.path.join(RESULTS_DIR, "eval_lst_comparison.png"))
    
    # PM2.5 reduction heatmaps
    plot_diff_heatmap(results, base_state, env_random.CH_PM25, "PM2.5 Reduction (ug/m3)", "Blues", os.path.join(RESULTS_DIR, "eval_pm25_comparison.png"))
    
    # Metrics bar charts
    plot_metrics_bar(df_results, os.path.join(RESULTS_DIR, "eval_metrics_comparison.png"))
    
    print("\n[OK] Evaluation complete. All results saved to results/ folder.")


if __name__ == "__main__":
    evaluate()
