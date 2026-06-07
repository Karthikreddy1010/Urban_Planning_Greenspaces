"""
graph_builder.py - Convert the spatial grid into a PyTorch Geometric graph.

Nodes  = grid cells (inside Newark boundary)
Edges  = 8-direction spatial adjacency + road connectivity
"""

import numpy as np
import geopandas as gpd
import torch
from torch_geometric.data import Data

from config import TARGET_CRS
import data_loader


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(grid_gdf: gpd.GeoDataFrame,
                feature_tensor: np.ndarray,
                valid_mask: np.ndarray,
                road_gdf: gpd.GeoDataFrame = None) -> Data:
    """
    Build a PyTorch Geometric ``Data`` object.

    Parameters
    ----------
    grid_gdf : GeoDataFrame
        The grid with row/col indices and metadata.
    feature_tensor : ndarray (H, W, C)
        The normalised feature tensor.
    valid_mask : ndarray (H, W) bool
        True for cells inside Newark boundary.
    road_gdf : GeoDataFrame, optional
        Road network for connectivity edges.  If None, loaded from disk.

    Returns
    -------
    data : torch_geometric.data.Data
        .x           - (N, C) node features
        .edge_index  - (2, E) edge list
        .pos         - (N, 2) centroid coordinates
        .node_rows   - (N,) row indices
        .node_cols   - (N,) col indices
        .grid_rows   - total grid rows (H)
        .grid_cols   - total grid cols (W)
    """
    print("[graph_builder] Building graph ...")

    rows, cols, C = feature_tensor.shape

    # ── Map valid cells to node IDs ──────────────────────────────────────
    cell_to_node = {}  # (r, c) → node_id
    node_features = []
    node_positions = []  # (cx, cy) for each node
    node_rows_list = []
    node_cols_list = []

    for _, cell in grid_gdf.iterrows():
        r, c = int(cell["row"]), int(cell["col"])
        if valid_mask[r, c]:
            nid = len(cell_to_node)
            cell_to_node[(r, c)] = nid
            node_features.append(feature_tensor[r, c, :])
            node_positions.append([cell["centroid_x"], cell["centroid_y"]])
            node_rows_list.append(r)
            node_cols_list.append(c)

    N = len(cell_to_node)
    print(f"  {N} nodes created")

    x = torch.tensor(np.array(node_features), dtype=torch.float32)
    pos = torch.tensor(np.array(node_positions), dtype=torch.float32)

    # ── Spatial edges (8-direction queen adjacency) ──────────────────────
    edge_src, edge_dst = [], []
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    for (r, c), nid in cell_to_node.items():
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cell_to_node:
                edge_src.append(nid)
                edge_dst.append(cell_to_node[(nr, nc)])

    spatial_edge_count = len(edge_src)
    print(f"  {spatial_edge_count} spatial edges (8-direction)")

    # ── Road connectivity edges ──────────────────────────────────────────
    if road_gdf is None:
        try:
            road_gdf = data_loader.load_road_network()
        except Exception as e:
            print(f"  [warn] Could not load road network: {e}")
            road_gdf = None

    if road_gdf is not None and len(road_gdf) > 0:
        road_edges = _build_road_edges(road_gdf, grid_gdf, cell_to_node)
        edge_src.extend(road_edges[0])
        edge_dst.extend(road_edges[1])
        road_edge_count = len(road_edges[0])
        print(f"  {road_edge_count} road connectivity edges")

    # ── Assemble edge_index ──────────────────────────────────────────────
    if len(edge_src) > 0:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        # Remove duplicate edges
        edge_index = _deduplicate_edges(edge_index)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    print(f"  Total edges: {edge_index.size(1)} (after dedup)")

    # ── Build Data object ────────────────────────────────────────────────
    data = Data(
        x=x,
        edge_index=edge_index,
        pos=pos,
    )
    # Store additional metadata as tensors
    data.node_rows = torch.tensor(node_rows_list, dtype=torch.long)
    data.node_cols = torch.tensor(node_cols_list, dtype=torch.long)
    data.grid_rows = rows
    data.grid_cols = cols
    data.num_nodes = N

    return data


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_road_edges(road_gdf: gpd.GeoDataFrame,
                      grid_gdf: gpd.GeoDataFrame,
                      cell_to_node: dict) -> tuple:
    """
    For each road segment, find which grid cells it passes through,
    then add edges between consecutive cells along the road.
    """
    resolution = grid_gdf.attrs["resolution"]
    origin_x = grid_gdf.attrs["origin_x"]
    origin_y = grid_gdf.attrs["origin_y"]

    src, dst = [], []

    for _, road in road_gdf.iterrows():
        geom = road.geometry
        if geom is None or geom.is_empty:
            continue

        # Sample points along the road
        try:
            length = geom.length
            if length < 1.0:
                continue
            n_samples = max(2, int(length / (resolution * 0.5)))
            fracs = np.linspace(0, 1, n_samples)
            points = [geom.interpolate(f, normalized=True) for f in fracs]
        except Exception:
            continue

        # Map points to grid cells
        road_cells = []
        for pt in points:
            c = int((pt.x - origin_x) / resolution)
            r = int((origin_y - pt.y) / resolution)
            if (r, c) in cell_to_node:
                nid = cell_to_node[(r, c)]
                if len(road_cells) == 0 or road_cells[-1] != nid:
                    road_cells.append(nid)

        # Add edges between consecutive cells
        for i in range(len(road_cells) - 1):
            src.append(road_cells[i])
            dst.append(road_cells[i + 1])
            src.append(road_cells[i + 1])
            dst.append(road_cells[i])

    return src, dst


def _deduplicate_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Remove duplicate edges from an edge_index tensor."""
    # Encode each edge as a single integer for fast dedup
    n = edge_index.max().item() + 1
    encoded = edge_index[0] * n + edge_index[1]
    _, unique_idx = torch.unique(encoded, return_inverse=False, sorted=True, dim=0), None
    unique_mask = torch.zeros(encoded.size(0), dtype=torch.bool)
    seen = set()
    for i in range(encoded.size(0)):
        val = encoded[i].item()
        if val not in seen:
            seen.add(val)
            unique_mask[i] = True
    return edge_index[:, unique_mask]


if __name__ == "__main__":
    from preprocess import preprocess_all
    grid_gdf, tensor, norm_params, valid_mask = preprocess_all()
    data = build_graph(grid_gdf, tensor, valid_mask)
    print(f"\n[OK] Graph: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    print(f"  Node features shape: {data.x.shape}")
