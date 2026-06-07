"""
model.py - Neural-network architecture for the green-space placement agent.

Architecture:
  CNN Encoder  → spatial feature extraction from (H, W, C) tensor
  GraphSAGE    → 3-layer message-passing on the cell graph
  Policy Head  → action probabilities over all cells (masked softmax)
  Value Head   → scalar state-value estimate V(s)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch

from config import MODEL_CFG, GRID_CFG


class CNNEncoder(nn.Module):
    """
    3-layer CNN that extracts per-cell spatial features from the grid tensor.
    Input  : (B, C_in, H, W)
    Output : (B, C_out, H, W)
    """

    def __init__(self, in_channels: int, channel_list: tuple = None):
        super().__init__()
        channel_list = channel_list or MODEL_CFG.cnn_channels
        layers = []
        ch_in = in_channels
        for ch_out in channel_list:
            layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ])
            ch_in = ch_out
        self.net = nn.Sequential(*layers)
        self.out_channels = channel_list[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphSAGEEncoder(nn.Module):
    """
    3-layer GraphSAGE that refines node features using neighbourhood
    aggregation on the spatial + road graph.
    """

    def __init__(self, in_channels: int, channel_list: tuple = None):
        super().__init__()
        channel_list = channel_list or MODEL_CFG.gnn_channels
        self.convs = nn.ModuleList()
        ch_in = in_channels
        for ch_out in channel_list:
            self.convs.append(SAGEConv(ch_in, ch_out))
            ch_in = ch_out
        self.out_channels = channel_list[-1]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


class PolicyHead(nn.Module):
    """Linear projection → per-node logit for action selection."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (N,)."""
        return self.linear(x).squeeze(-1)


class ValueHead(nn.Module):
    """Global mean-pool → MLP → scalar V(s)."""

    def __init__(self, in_channels: int, hidden: int = None):
        super().__init__()
        hidden = hidden or MODEL_CFG.value_hidden
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, D) node embeddings
        Returns scalar value V(s).
        """
        pooled = x.mean(dim=0, keepdim=True)  # (1, D)
        return self.mlp(pooled).squeeze()


class GreenSpacePolicyNet(nn.Module):
    """
    Full actor-critic model:

    CNN Encoder  - (B, C, H, W) → (B, C', H, W)
    ↓ flatten valid cells
    GraphSAGE    - (N, C') × edge_index → (N, D)
    ↓ dual heads
    Policy Head  - (N,) logits  →  π(a|s)
    Value Head   - scalar       →  V(s)
    """

    def __init__(self,
                 in_channels: int = None,
                 grid_rows: int = None,
                 grid_cols: int = None):
        super().__init__()
        in_channels = in_channels or GRID_CFG.n_channels

        self.cnn = CNNEncoder(in_channels)
        self.gnn = GraphSAGEEncoder(self.cnn.out_channels)
        self.policy = PolicyHead(self.gnn.out_channels)
        self.value = ValueHead(self.gnn.out_channels)

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    def forward(self,
                state_tensor: torch.Tensor,
                graph_data: Data,
                feasible_mask: torch.Tensor = None):
        """
        Parameters
        ----------
        state_tensor : (H, W, C) or (1, C, H, W)
            Current environment state.
        graph_data : torch_geometric.data.Data
            Must have .edge_index, .node_rows, .node_cols.
        feasible_mask : (N,) bool tensor, optional
            True for nodes that can be selected.

        Returns
        -------
        action_probs : (N,) probability distribution over nodes
        value : scalar V(s)
        action_logits : (N,) raw logits (for PPO)
        """
        # ── CNN encoding ─────────────────────────────────────────────────
        if state_tensor.dim() == 3:
            # (H, W, C) → (1, C, H, W)
            x_cnn = state_tensor.permute(2, 0, 1).unsqueeze(0)
        elif state_tensor.dim() == 4:
            x_cnn = state_tensor
        else:
            raise ValueError(f"Unexpected state_tensor shape: {state_tensor.shape}")

        cnn_out = self.cnn(x_cnn)  # (1, C', H, W)

        # ── Extract node features from CNN output ────────────────────────
        cnn_out = cnn_out.squeeze(0)  # (C', H, W)
        node_rows = graph_data.node_rows
        node_cols = graph_data.node_cols
        # Index into the spatial feature map at each node's (row, col)
        node_features = cnn_out[:, node_rows, node_cols].T  # (N, C')

        # ── GNN encoding ─────────────────────────────────────────────────
        node_embeddings = self.gnn(node_features, graph_data.edge_index)  # (N, D)

        # ── Policy head ──────────────────────────────────────────────────
        logits = self.policy(node_embeddings)  # (N,)

        # Mask infeasible actions with -inf
        if feasible_mask is not None:
            logits = logits.masked_fill(~feasible_mask, float("-inf"))

        action_probs = F.softmax(logits, dim=0)

        # ── Value head ───────────────────────────────────────────────────
        value = self.value(node_embeddings)

        return action_probs, value, logits

    def act(self,
            state_tensor: torch.Tensor,
            graph_data: Data,
            feasible_mask: torch.Tensor = None,
            deterministic: bool = False):
        """
        Select an action and return (action_node_id, log_prob, value).

        The action is a *node index* in the graph (not a flat grid index).
        Convert to grid (r, c) using graph_data.node_rows / node_cols.
        """
        with torch.no_grad():
            probs, value, logits = self.forward(state_tensor, graph_data, feasible_mask)

        if deterministic:
            action = probs.argmax().item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()

        log_prob = torch.log(probs[action] + 1e-10)
        return action, log_prob, value

    def evaluate_actions(self,
                         state_tensor: torch.Tensor,
                         graph_data: Data,
                         actions: torch.Tensor,
                         feasible_mask: torch.Tensor = None):
        """
        Evaluate previously taken actions for PPO update.

        Parameters
        ----------
        state_tensor : (H, W, C)
        actions : (K,) long tensor of node indices
        feasible_mask : (N,) bool

        Returns
        -------
        log_probs, values, entropy  (all for the given actions)
        """
        probs, value, logits = self.forward(state_tensor, graph_data, feasible_mask)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, value, entropy


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick architecture check with dummy data
    H, W, C = 65, 127, 7
    N = 500  # dummy node count

    model = GreenSpacePolicyNet(in_channels=C, grid_rows=H, grid_cols=W)
    print(f"Model parameters: {count_parameters(model):,}")

    # Dummy inputs
    state = torch.randn(H, W, C)
    edge_index = torch.randint(0, N, (2, 2000))
    node_rows = torch.randint(0, H, (N,))
    node_cols = torch.randint(0, W, (N,))

    graph = Data(x=torch.randn(N, C), edge_index=edge_index)
    graph.node_rows = node_rows
    graph.node_cols = node_cols

    feasible = torch.ones(N, dtype=torch.bool)
    feasible[0] = False

    probs, value, logits = model(state, graph, feasible)
    print(f"Action probs: {probs.shape}, sum={probs.sum():.4f}")
    print(f"Value: {value.item():.4f}")
    print("[OK] Model forward pass OK")
