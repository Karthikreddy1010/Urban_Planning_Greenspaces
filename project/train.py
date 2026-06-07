"""
train.py - Training orchestration for the Urban Green Space RL placement system.

This module handles:
  1. Parsing command-line arguments.
  2. Setting seeds for reproducibility.
  3. Preprocessing data and building the graph.
  4. Running the constrained PPO training loop.
  5. Logging training progress and metrics.
  6. Saving checkpoints and final model weights.
  7. Plotting and saving training metrics curves.
"""

import os
import time
import argparse
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import (
    DATA_DIR, RESULTS_DIR, TARGET_CRS, GRID_CFG,
    ENV_CFG, REWARD_CFG, PPO_CFG, MODEL_CFG, TRAIN_CFG
)
from preprocess import preprocess_all
from graph_builder import build_graph
from env import GreenSpaceEnv
from model import GreenSpacePolicyNet
from ppo import ConstrainedPPO


def parse_args():
    parser = argparse.ArgumentParser(description="Train Green Space RL Agent")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Path to Newark GIS datasets folder")
    parser.add_argument("--episodes", type=int, default=TRAIN_CFG.num_episodes,
                        help="Number of episodes to train")
    parser.add_argument("--budget", type=int, default=ENV_CFG.max_placements,
                        help="Number of green spaces to place per episode")
    parser.add_argument("--lr", type=float, default=PPO_CFG.lr,
                        help="Policy/value network learning rate")
    parser.add_argument("--lagrangian-lr", type=float, default=PPO_CFG.lagrangian_lr,
                        help="Lagrangian multiplier learning rate")
    parser.add_argument("--equity-threshold", type=float, default=PPO_CFG.equity_threshold,
                        help="Normalised population density threshold for equity")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force preprocessing from scratch (no cache)")
    parser.add_argument("--seed", type=int, default=TRAIN_CFG.seed,
                        help="Random seed for reproducibility")
    parser.add_argument("--log-interval", type=int, default=TRAIN_CFG.log_interval,
                        help="Interval of episodes to print logs")
    parser.add_argument("--save-interval", type=int, default=TRAIN_CFG.save_interval,
                        help="Interval of episodes to save checkpoints")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_metrics(history: dict, filepath: str):
    """Plot training curves and save as image."""
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    
    # 1. Total Reward
    axs[0, 0].plot(history["episode"], history["reward"], label="Reward", color="forestgreen")
    axs[0, 0].set_title("Episode Reward")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].grid(True)
    
    # 2. Cooling & PM2.5 Reductions
    axs[0, 1].plot(history["episode"], history["total_cooling"], label="LST Cooling (degC)", color="crimson")
    axs[0, 1].plot(history["episode"], history["total_pm25_red"], label="PM2.5 Reduction (µg/m³)", color="royalblue")
    axs[0, 1].set_title("Physical Reductions")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Cumulative Improvement")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # 3. Equity Score & Threshold
    axs[1, 0].plot(history["episode"], history["equity_score"], label="Equity Score", color="darkorange")
    axs[1, 0].axhline(y=PPO_CFG.equity_threshold, color="gray", linestyle="--", label="Threshold")
    axs[1, 0].set_title("Equity (Population Density at Placements)")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Normalised Density")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # 4. Lagrangian Multiplier (lambda_equity)
    axs[1, 1].plot(history["episode"], history["lambda_equity"], label="λ_equity", color="purple")
    axs[1, 1].set_title("Lagrangian Equity Multiplier")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("λ Value")
    axs[1, 1].grid(True)
    
    # 5. Policy & Value Losses
    axs[2, 0].plot(history["episode"], history["policy_loss"], label="Policy Loss", color="teal")
    axs[2, 0].plot(history["episode"], history["value_loss"], label="Value Loss", color="chocolate")
    axs[2, 0].set_title("Losses")
    axs[2, 0].set_xlabel("Episode")
    axs[2, 0].set_ylabel("Loss")
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    
    # 6. Policy Entropy
    axs[2, 1].plot(history["episode"], history["entropy"], label="Entropy", color="mediumpurple")
    axs[2, 1].set_title("Policy Entropy")
    axs[2, 1].set_xlabel("Episode")
    axs[2, 1].set_ylabel("Entropy")
    axs[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"[train] Saved training curves plot to {filepath}")


def train():
    args = parse_args()
    set_seed(args.seed)
    
    # Override configs with CLI args
    ENV_CFG.max_placements = args.budget
    PPO_CFG.lr = args.lr
    PPO_CFG.lagrangian_lr = args.lagrangian_lr
    PPO_CFG.equity_threshold = args.equity_threshold
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Using device: {device.upper()}")
    
    # ── 1. Data preprocessing ──────────────────────────────────────────────
    grid_gdf, feature_tensor, norm_params, valid_mask = preprocess_all(
        data_dir=args.data_dir,
        use_cache=not args.no_cache
    )
    
    # ── 2. Build graph ─────────────────────────────────────────────────────
    graph_data = build_graph(grid_gdf, feature_tensor, valid_mask)
    # Move graph to target device
    graph_data = graph_data.to(device)
    
    # ── 3. Initialize env and model ────────────────────────────────────────
    grid_meta = {
        "resolution": grid_gdf.attrs["resolution"],
        "rows": grid_gdf.attrs["rows"],
        "cols": grid_gdf.attrs["cols"]
    }
    
    env = GreenSpaceEnv(
        feature_tensor=feature_tensor,
        valid_mask=valid_mask,
        norm_params=norm_params,
        grid_meta=grid_meta
    )
    
    model = GreenSpacePolicyNet(
        in_channels=GRID_CFG.n_channels,
        grid_rows=grid_meta["rows"],
        grid_cols=grid_meta["cols"]
    ).to(device)
    
    agent = ConstrainedPPO(model=model, device=device)
    
    print(f"[train] Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    print(f"[train] Action space size: {env.action_space_size}")
    print(f"[train] Equity threshold target: {PPO_CFG.equity_threshold:.2f}")
    print(f"[train] Starting training for {args.episodes} episodes...")
    print("=" * 85)
    print(f"{'Ep':<6} | {'Reward':<8} | {'Cooling':<8} | {'PM2.5':<8} | {'Equity':<7} | {'l_eq':<7} | {'P-Loss':<8} | {'V-Loss':<8} | {'Ent':<6}")
    print("-" * 85)
    
    history = {
        "episode": [],
        "reward": [],
        "total_cooling": [],
        "total_pm25_red": [],
        "equity_score": [],
        "lambda_equity": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": []
    }
    
    best_reward = -float("inf")
    start_time = time.time()
    
    for ep in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            # Get node-level feasibility mask
            # node_feasible is a bool mask mapping nodes in GNN to whether they are feasible
            node_rows = graph_data.node_rows.cpu().numpy()
            node_cols = graph_data.node_cols.cpu().numpy()
            node_feasible = env.feasible_mask[node_rows, node_cols] > 0.5
            
            node_feasible_t = torch.tensor(node_feasible, dtype=torch.bool, device=device)
            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            
            # Select action
            action_node_id, log_prob, value = model.act(
                state_tensor=state_t,
                graph_data=graph_data,
                feasible_mask=node_feasible_t,
                deterministic=False
            )
            
            # Map selected node index back to flat grid index
            r = int(node_rows[action_node_id])
            c = int(node_cols[action_node_id])
            flat_action = r * env.W + c
            
            # Step in environment
            next_state, reward, done, info = env.step(flat_action)
            
            # Store in rollout buffer
            agent.store_transition(
                state=state,
                action=action_node_id,
                reward=reward,
                log_prob=log_prob,
                value=value,
                done=done,
                feasible_mask=node_feasible
            )
            
            state = next_state
            episode_reward += reward
            
        # End of episode: Compute equity score & trigger update
        equity_score = env.get_equity_score()
        metrics = agent.update(
            graph_data=graph_data,
            last_value=0.0,
            equity_score=equity_score
        )
        
        # Save metrics in history
        history["episode"].append(int(ep))
        history["reward"].append(float(episode_reward))
        history["total_cooling"].append(float(env._total_lst_reduction))
        history["total_pm25_red"].append(float(env._total_pm25_reduction))
        history["equity_score"].append(float(equity_score))
        history["lambda_equity"].append(float(metrics.get("lambda_equity", 0.0)))
        history["policy_loss"].append(float(metrics.get("policy_loss", 0.0)))
        history["value_loss"].append(float(metrics.get("value_loss", 0.0)))
        history["entropy"].append(float(metrics.get("entropy", 0.0)))
        
        # Log to stdout
        if ep % args.log_interval == 0 or ep == 1:
            print(f"{ep:<6d} | {episode_reward:<8.3f} | {env._total_lst_reduction:<8.2f} | "
                  f"{env._total_pm25_reduction:<8.2f} | {equity_score:<7.3f} | "
                  f"{metrics.get('lambda_equity', 0.0):<7.3f} | {metrics.get('policy_loss', 0.0):<8.4f} | "
                  f"{metrics.get('value_loss', 0.0):<8.4f} | {metrics.get('entropy', 0.0):<6.2f}")
            
        # Save best model checkpoint
        if episode_reward > best_reward and ep > 10:
            best_reward = episode_reward
            best_path = os.path.join(RESULTS_DIR, "model_best.pt")
            torch.save(model.state_dict(), best_path)
            
        # Regular save interval
        if ep % args.save_interval == 0:
            checkpoint_path = os.path.join(RESULTS_DIR, f"model_ep_{ep}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[train] Saved checkpoint: {checkpoint_path}")
            
    # Save final model
    final_path = os.path.join(RESULTS_DIR, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n[train] Training completed in {(time.time() - start_time) / 60:.2f} minutes.")
    print(f"[train] Saved final model to {final_path}")
    
    # Save training history to JSON
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] Saved training history to {history_path}")
    
    # Plot training curves
    curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plot_metrics(history, curves_path)


if __name__ == "__main__":
    train()
