"""
preprocess.py  --  Production-level geospatial preprocessing pipeline.

Pipeline overview
=================
  STEP 1  Load all raw datasets, standardise CRS to EPSG:32618 (UTM 18N).
  STEP 2  Create a 100 m x 100 m fishnet grid over Newark boundary.
  STEP 3  Raster alignment -- reproject Landsat / NLCD to the grid.
  STEP 4  Vector-to-grid aggregation (population, traffic, soil, parcels).
  STEP 5  PM2.5 interpolation (IDW / RBF from AQS points).
  STEP 6  Feature engineering -- 7 channels, normalised.
  STEP 7  NaN handling and min-max normalisation.
  STEP 8  Assemble final tensor  X in (H, W, C), save .npy + metadata.
  STEP 9  Validation checks -- per-layer plots, stats, alignment audit.

Output
------
  (grid_gdf, tensor, norm_params, valid_mask)  -- cached as pickle.
  results/feature_tensor.npy                   -- the raw (H, W, C) array.
  results/tensor_metadata.json                 -- bounds, transform, channels.
  results/channel_sanity_*.png                 -- one plot per channel.
"""

import json
import os
import pickle
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.crs
import rasterio.transform
from rasterio.warp import reproject, Resampling
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from shapely.geometry import box, Point

from config import (
    TARGET_CRS, GRID_CFG, LANDSAT_CFG, DATA_PATHS,
    CACHE_DIR, RESULTS_DIR,
)
import data_loader

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# STEP 2 -- Grid Generation
# ============================================================================

def create_grid(boundary_gdf: gpd.GeoDataFrame,
                resolution: float = None) -> gpd.GeoDataFrame:
    """
    Create a regular *resolution* m x *resolution* m fishnet grid over
    the Newark boundary.  Only cells whose centroid falls inside the
    boundary polygon are retained.

    Returns
    -------
    grid_gdf : GeoDataFrame
        Columns: geometry, row, col, grid_id, centroid_x, centroid_y.
        attrs:   rows, cols, resolution, origin_x, origin_y.
    """
    resolution = resolution or GRID_CFG.resolution
    boundary_union = boundary_gdf.geometry.union_all()
    minx, miny, maxx, maxy = boundary_union.bounds

    # Snap to resolution
    minx = np.floor(minx / resolution) * resolution
    miny = np.floor(miny / resolution) * resolution
    maxx = np.ceil(maxx / resolution) * resolution
    maxy = np.ceil(maxy / resolution) * resolution

    cols_count = int((maxx - minx) / resolution)
    rows_count = int((maxy - miny) / resolution)

    print(f"[preprocess] Grid envelope: {cols_count} x {rows_count} cells "
          f"({resolution} m resolution)")

    cells = []
    for r in range(rows_count):
        for c in range(cols_count):
            x0 = minx + c * resolution
            y0 = maxy - (r + 1) * resolution      # row 0 = top
            x1 = x0 + resolution
            y1 = y0 + resolution
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if boundary_union.contains(Point(cx, cy)):
                cells.append({
                    "geometry": box(x0, y0, x1, y1),
                    "row": r,
                    "col": c,
                    "centroid_x": cx,
                    "centroid_y": cy,
                })

    grid_gdf = gpd.GeoDataFrame(cells, crs=TARGET_CRS)
    grid_gdf["grid_id"] = range(len(grid_gdf))
    print(f"  {len(grid_gdf)} cells inside Newark boundary")

    grid_gdf.attrs["rows"] = rows_count
    grid_gdf.attrs["cols"] = cols_count
    grid_gdf.attrs["resolution"] = resolution
    grid_gdf.attrs["origin_x"] = minx
    grid_gdf.attrs["origin_y"] = maxy       # top-left y
    return grid_gdf


# ============================================================================
# STEP 3 -- Raster-to-Grid helpers
# ============================================================================

def _reproject_raster_to_grid(raster: np.ndarray,
                              src_transform, src_crs,
                              grid_gdf: gpd.GeoDataFrame,
                              resampling=Resampling.bilinear,
                              src_nodata=None) -> np.ndarray:
    """
    Reproject *raster* from its native CRS / transform into the grid's
    coordinate space.  Returns a (rows, cols) float32 array aligned to
    the grid.  Pixels with no data are set to NaN.

    Parameters
    ----------
    resampling : Resampling enum
        Use Resampling.nearest for categorical data (land cover),
        Resampling.bilinear for continuous data (LST, impervious).
    src_nodata : scalar or None
        Value in the source raster that represents "no data".
    """
    res = grid_gdf.attrs["resolution"]
    rows = grid_gdf.attrs["rows"]
    cols = grid_gdf.attrs["cols"]
    origin_x = grid_gdf.attrs["origin_x"]
    origin_y = grid_gdf.attrs["origin_y"]

    dst_transform = rasterio.transform.from_origin(origin_x, origin_y, res, res)
    dst_crs = rasterio.crs.CRS.from_epsg(32618)

    # Work in float32; use a sentinel for nodata so rasterio can mask
    NODATA_SENTINEL = -9999.0
    src = raster.astype(np.float32).copy()
    if src_nodata is not None:
        src[src == src_nodata] = NODATA_SENTINEL
    # Also treat existing NaN as nodata
    src[np.isnan(src)] = NODATA_SENTINEL

    dst_array = np.full((rows, cols), NODATA_SENTINEL, dtype=np.float32)

    try:
        reproject(
            source=src,
            destination=dst_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=NODATA_SENTINEL,
            dst_nodata=NODATA_SENTINEL,
        )
    except Exception as e:
        print(f"  [warn] Reprojection failed: {e}")
        return np.full((rows, cols), np.nan, dtype=np.float32)

    # Replace sentinel with NaN
    dst_array[dst_array == NODATA_SENTINEL] = np.nan
    # Guard against values very close to sentinel
    dst_array[dst_array < -9990.0] = np.nan

    n_valid = np.count_nonzero(~np.isnan(dst_array))
    print(f"  Reprojected: {n_valid}/{rows*cols} pixels valid")
    return dst_array


def _check_raster_coverage(raster_bounds, raster_crs,
                           boundary_gdf: gpd.GeoDataFrame) -> bool:
    """Return True if the raster spatially overlaps the Newark boundary."""
    from shapely.geometry import box as sbox
    raster_poly = sbox(raster_bounds.left, raster_bounds.bottom,
                       raster_bounds.right, raster_bounds.top)
    raster_gdf = gpd.GeoDataFrame(geometry=[raster_poly],
                                  crs=raster_crs)
    raster_gdf = raster_gdf.to_crs(boundary_gdf.crs)
    return raster_gdf.geometry[0].intersects(
        boundary_gdf.geometry.union_all()
    )


# ============================================================================
# STEP 4 -- Channel rasterisation functions
# ============================================================================

def rasterize_population(grid_gdf: gpd.GeoDataFrame,
                         demographics_df: pd.DataFrame,
                         boundary_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Compute population density (people / km^2) per grid cell.
    Uses total population and distributes uniformly as a baseline.
    Will be refined later using impervious surface as a proxy.
    """
    rows = grid_gdf.attrs["rows"]
    cols = grid_gdf.attrs["cols"]

    total_pop = demographics_df["population"].sum()
    area_km2 = boundary_gdf.geometry.area.sum() / 1e6
    mean_density = total_pop / area_km2

    pop_grid = np.full((rows, cols), np.nan, dtype=np.float32)
    for _, cell in grid_gdf.iterrows():
        pop_grid[cell["row"], cell["col"]] = mean_density

    print(f"  Population: {total_pop:,.0f} people, "
          f"mean density {mean_density:.0f} people/km^2")
    return pop_grid


def rasterize_lst(grid_gdf: gpd.GeoDataFrame,
                  lst_data: tuple,
                  boundary_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Reproject Landsat LST to grid -> (rows, cols) array in deg C."""
    temp_c, transform, crs = lst_data
    # Check coverage
    bounds = rasterio.transform.array_bounds(temp_c.shape[0],
                                             temp_c.shape[1], transform)
    bb = type('B', (), {'left': bounds[0], 'bottom': bounds[1],
                        'right': bounds[2], 'top': bounds[3]})()
    if not _check_raster_coverage(bb, crs, boundary_gdf):
        print("  [warn] Landsat does not cover Newark -- using zeros")
        rows, cols = grid_gdf.attrs["rows"], grid_gdf.attrs["cols"]
        return np.full((rows, cols), np.nan, dtype=np.float32)

    return _reproject_raster_to_grid(temp_c, transform, crs, grid_gdf,
                                     resampling=Resampling.bilinear)


def rasterize_pm25(grid_gdf: gpd.GeoDataFrame,
                   pm25_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Interpolate PM2.5 from sparse AQS monitoring points to grid centroids
    using RBF (thin-plate spline) with IDW fallback.
    """
    rows = grid_gdf.attrs["rows"]
    cols = grid_gdf.attrs["cols"]

    pts = np.array([[g.x, g.y] for g in pm25_gdf.geometry])
    vals = pm25_gdf["pm25_value"].values.astype(np.float64)

    if len(pts) < 2:
        print("  [warn] Too few PM2.5 points -- using uniform value")
        mean_val = vals.mean() if len(vals) > 0 else 10.0
        grid = np.full((rows, cols), np.nan, dtype=np.float32)
        for _, cell in grid_gdf.iterrows():
            grid[cell["row"], cell["col"]] = mean_val
        return grid

    centroids = np.column_stack([
        grid_gdf["centroid_x"].values,
        grid_gdf["centroid_y"].values,
    ])

    try:
        rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline",
                              smoothing=1.0)
        interp_vals = rbf(centroids)
    except Exception:
        interp_vals = _idw(pts, vals, centroids)

    interp_vals = np.clip(interp_vals, 0, 100)

    grid = np.full((rows, cols), np.nan, dtype=np.float32)
    for i, (_, cell) in enumerate(grid_gdf.iterrows()):
        grid[cell["row"], cell["col"]] = interp_vals[i]

    print(f"  PM2.5: range [{np.nanmin(grid):.1f}, {np.nanmax(grid):.1f}] ug/m^3")
    return grid


def _idw(pts, vals, query, power=2.0):
    """Inverse Distance Weighting interpolation."""
    result = np.zeros(len(query))
    for i, q in enumerate(query):
        dists = np.sqrt(((pts - q) ** 2).sum(axis=1))
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / (dists ** power)
        result[i] = np.sum(weights * vals) / np.sum(weights)
    return result


def rasterize_impervious_from_nlcd(grid_gdf, imp_data, boundary_gdf):
    """
    Attempt to reproject NLCD impervious surface to grid.
    Returns (array, success_bool).
    """
    data, transform, crs = imp_data
    bounds = rasterio.transform.array_bounds(data.shape[0], data.shape[1],
                                             transform)
    bb = type('B', (), {'left': bounds[0], 'bottom': bounds[1],
                        'right': bounds[2], 'top': bounds[3]})()
    if not _check_raster_coverage(bb, crs, boundary_gdf):
        print("  [warn] NLCD impervious does NOT cover Newark")
        return None, False

    grid = _reproject_raster_to_grid(data.astype(np.float32),
                                     transform, crs, grid_gdf,
                                     resampling=Resampling.bilinear,
                                     src_nodata=250)
    grid = np.clip(grid, 0, 100)
    n_valid = np.count_nonzero(~np.isnan(grid))
    if n_valid < 10:
        print("  [warn] NLCD impervious reprojection produced no valid data")
        return None, False
    return grid, True


def rasterize_impervious_proxy(grid_gdf, green_grid, road_gdf):
    """
    Fallback: derive impervious surface proxy from:
      - green_space indicator (non-green --> likely impervious)
      - road network density (roads = fully impervious)
    Returns values in [0, 100] scale.
    """
    rows = grid_gdf.attrs["rows"]
    cols = grid_gdf.attrs["cols"]
    resolution = grid_gdf.attrs["resolution"]

    # Start with inverse of green space: non-green areas assumed 70% impervious
    imp_grid = (1.0 - green_grid) * 70.0

    # Add road coverage: cells overlapping roads get boosted
    if road_gdf is not None and len(road_gdf) > 0:
        try:
            boundary_union = grid_gdf.geometry.union_all()
            roads_clipped = road_gdf[road_gdf.geometry.intersects(
                boundary_union.buffer(200))]
            if len(roads_clipped) > 0:
                roads_buf = roads_clipped.copy()
                roads_buf["geometry"] = roads_buf.geometry.buffer(resolution / 4)
                joined = gpd.sjoin(grid_gdf[["geometry", "row", "col"]],
                                   roads_buf[["geometry"]],
                                   how="left", predicate="intersects")
                road_count = joined.groupby(["row", "col"]).size().reset_index(
                    name="road_segments")
                for _, r in road_count.iterrows():
                    row_i, col_i = int(r["row"]), int(r["col"])
                    # Increase impervious by up to 20% based on road presence
                    imp_grid[row_i, col_i] = min(100.0,
                        imp_grid[row_i, col_i] + min(r["road_segments"] * 5, 20))
        except Exception as e:
            print(f"  [warn] Road overlay failed: {e}")

    imp_grid = np.clip(imp_grid, 0, 100)
    print(f"  Impervious (proxy): range [{np.nanmin(imp_grid):.0f}, "
          f"{np.nanmax(imp_grid):.0f}]%")
    return imp_grid


def rasterize_green_space_from_nlcd(grid_gdf, lc_data, open_space_gdf,
                                    boundary_gdf):
    """
    Try NLCD landcover for green-space classification.
    Falls back to Open Space polygons + NDVI if NLCD doesn't cover Newark.
    """
    rows = grid_gdf.attrs["rows"]
    cols = grid_gdf.attrs["cols"]

    green_grid = np.zeros((rows, cols), dtype=np.float32)
    used_nlcd = False

    # --- Attempt NLCD landcover ---
    if lc_data is not None:
        lc_arr, transform, crs = lc_data
        bounds = rasterio.transform.array_bounds(lc_arr.shape[0],
                                                 lc_arr.shape[1], transform)
        bb = type('B', (), {'left': bounds[0], 'bottom': bounds[1],
                            'right': bounds[2], 'top': bounds[3]})()
        if _check_raster_coverage(bb, crs, boundary_gdf):
            lc_grid = _reproject_raster_to_grid(
                lc_arr.astype(np.float32), transform, crs, grid_gdf,
                resampling=Resampling.nearest,
                src_nodata=250)
            green_classes = {41, 42, 43, 52, 71, 81, 90, 95}
            for cls in green_classes:
                green_grid[lc_grid == cls] = 1.0
            n = int(np.nansum(green_grid))
            if n > 0:
                used_nlcd = True
                print(f"  Green space (NLCD): {n} cells")
        else:
            print("  [warn] NLCD landcover does NOT cover Newark")

    # --- Overlay open-space polygons (always applied) ---
    if open_space_gdf is not None and len(open_space_gdf) > 0:
        try:
            boundary_union = grid_gdf.geometry.union_all()
            os_clipped = open_space_gdf[
                open_space_gdf.geometry.intersects(boundary_union)]
            if len(os_clipped) > 0:
                joined = gpd.sjoin(grid_gdf, os_clipped,
                                   how="inner", predicate="intersects")
                os_cells = set()
                for _, cell in joined.iterrows():
                    os_cells.add((cell["row"], cell["col"]))
                    green_grid[cell["row"], cell["col"]] = 1.0
                print(f"  Green space (Open Space overlay): "
                      f"{len(os_cells)} additional cells")
        except Exception as e:
            print(f"  [warn] Open space overlay failed: {e}")

    n_total = int(np.nansum(green_grid))
    src = "NLCD + Open Space" if used_nlcd else "Open Space only"
    print(f"  Green space TOTAL: {n_total} cells ({src})")
    return green_grid


def rasterize_traffic(grid_gdf: gpd.GeoDataFrame,
                      traffic_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Assign traffic intensity to grid cells by buffering AADT road
    segments and spatial-joining to the grid.
    """
    rows = grid_gdf.attrs["rows"]
    cols = grid_gdf.attrs["cols"]
    resolution = grid_gdf.attrs["resolution"]

    traffic_grid = np.full((rows, cols), np.nan, dtype=np.float32)

    boundary_union = grid_gdf.geometry.union_all()
    traffic_clipped = traffic_gdf[traffic_gdf.geometry.intersects(
        boundary_union.buffer(500))].copy()

    if len(traffic_clipped) == 0:
        print("  [warn] No traffic data overlaps Newark -- using zeros")
        for _, cell in grid_gdf.iterrows():
            traffic_grid[cell["row"], cell["col"]] = 0.0
        return traffic_grid

    traffic_clipped = traffic_clipped.copy()
    traffic_clipped["geometry"] = traffic_clipped.geometry.buffer(resolution / 2)

    try:
        joined = gpd.sjoin(grid_gdf[["geometry", "row", "col"]],
                           traffic_clipped[["geometry", "aadt_value"]],
                           how="left", predicate="intersects")
        agg = joined.groupby(["row", "col"])["aadt_value"].sum().reset_index()
        for _, r in agg.iterrows():
            traffic_grid[int(r["row"]), int(r["col"])] = r["aadt_value"]
    except Exception as e:
        print(f"  [warn] Traffic spatial join failed: {e}")

    # Fill missing cells with 0
    for _, cell in grid_gdf.iterrows():
        if np.isnan(traffic_grid[cell["row"], cell["col"]]):
            traffic_grid[cell["row"], cell["col"]] = 0.0

    print(f"  Traffic: range [{np.nanmin(traffic_grid):.0f}, "
          f"{np.nanmax(traffic_grid):.0f}] AADT")
    return traffic_grid


def rasterize_feasibility(grid_gdf: gpd.GeoDataFrame,
                          green_grid: np.ndarray,
                          preserved_gdf: gpd.GeoDataFrame,
                          lc_data: tuple = None,
                          boundary_gdf: gpd.GeoDataFrame = None
                          ) -> np.ndarray:
    """
    Land feasibility mask:  1 = can place green space,  0 = cannot.

    Infeasible if:
      - Already green space
      - Preserved / protected land
      - Water (NLCD class 11, if available)
      - Outside the boundary
    """
    rows = grid_gdf.attrs["rows"]
    cols = grid_gdf.attrs["cols"]

    # Start: all cells outside boundary are 0
    feasibility = np.zeros((rows, cols), dtype=np.float32)
    valid_cells = set()
    for _, cell in grid_gdf.iterrows():
        r, c = cell["row"], cell["col"]
        valid_cells.add((r, c))
        feasibility[r, c] = 1.0      # boundary cell -> feasible by default

    # Exclude already-green cells
    feasibility[green_grid == 1] = 0.0

    # Exclude water from NLCD (class 11) if the data covers Newark
    if lc_data is not None and boundary_gdf is not None:
        lc_arr, transform, crs = lc_data
        bounds = rasterio.transform.array_bounds(
            lc_arr.shape[0], lc_arr.shape[1], transform)
        bb = type('B', (), {'left': bounds[0], 'bottom': bounds[1],
                            'right': bounds[2], 'top': bounds[3]})()
        if _check_raster_coverage(bb, crs, boundary_gdf):
            lc_grid = _reproject_raster_to_grid(
                lc_arr.astype(np.float32), transform, crs, grid_gdf,
                resampling=Resampling.nearest, src_nodata=250)
            feasibility[lc_grid == 11] = 0.0
            print("  Applied NLCD water mask (class 11)")

    # Exclude preserved land
    if preserved_gdf is not None and len(preserved_gdf) > 0:
        try:
            joined = gpd.sjoin(grid_gdf, preserved_gdf,
                               how="inner", predicate="intersects")
            n_preserved = 0
            for _, cell in joined.iterrows():
                feasibility[cell["row"], cell["col"]] = 0.0
                n_preserved += 1
            print(f"  Excluded {n_preserved} preserved-land cells")
        except Exception:
            pass

    n_feasible = int(np.sum(feasibility == 1))
    n_boundary = len(valid_cells)
    print(f"  Feasibility: {n_feasible} / {n_boundary} boundary cells feasible")
    return feasibility


# ============================================================================
# Population refinement
# ============================================================================

def refine_population_with_impervious(pop_grid, imp_grid, grid_gdf,
                                       demographics_df):
    """
    Redistribute population proportional to impervious surface cover.
    Higher impervious ~ more development ~ more people.
    """
    rows, cols = pop_grid.shape
    total_pop = demographics_df["population"].sum()

    weights = imp_grid.copy()
    weights[np.isnan(weights)] = 0
    weights = weights / 100.0

    valid_mask = np.zeros((rows, cols), dtype=bool)
    for _, cell in grid_gdf.iterrows():
        valid_mask[cell["row"], cell["col"]] = True

    weights[~valid_mask] = 0
    total_weight = weights.sum()

    if total_weight > 0:
        cell_area_km2 = (grid_gdf.attrs["resolution"] / 1000.0) ** 2
        pop_refined = (weights / total_weight) * total_pop / cell_area_km2
    else:
        pop_refined = pop_grid.copy()

    # Light smoothing
    pop_refined = gaussian_filter(pop_refined, sigma=1.0)
    pop_refined[~valid_mask] = np.nan

    print(f"  Population (refined): range [{np.nanmin(pop_refined):.0f}, "
          f"{np.nanmax(pop_refined):.0f}] people/km^2")
    return pop_refined


# ============================================================================
# STEP 8 -- Build feature tensor
# ============================================================================

def build_feature_tensor(channel_arrays: dict,
                         grid_gdf: gpd.GeoDataFrame):
    """
    Stack channel arrays into (H, W, C).  Per-channel:
      - NaN imputation with median of valid values
      - Min-max normalisation to [0, 1]

    Returns (tensor, norm_params).
    """
    rows = grid_gdf.attrs["rows"]
    cols = grid_gdf.attrs["cols"]
    C = len(GRID_CFG.channels)

    tensor = np.full((rows, cols, C), np.nan, dtype=np.float32)
    norm_params = {}

    for i, ch_name in enumerate(GRID_CFG.channels):
        arr = channel_arrays[ch_name]

        # Impute NaN with median
        valid_vals = arr[~np.isnan(arr)]
        if len(valid_vals) > 0:
            median_val = np.median(valid_vals)
            arr = np.where(np.isnan(arr), median_val, arr)
        else:
            arr = np.zeros_like(arr)

        # Min-max normalisation
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax - vmin > 1e-8:
            arr_norm = (arr - vmin) / (vmax - vmin)
        else:
            arr_norm = np.zeros_like(arr)

        tensor[:, :, i] = arr_norm
        norm_params[ch_name] = {"min": vmin, "max": vmax}

    print(f"\n[preprocess] Feature tensor shape: {tensor.shape}")
    for ch in GRID_CFG.channels:
        p = norm_params[ch]
        print(f"  {ch}: raw range [{p['min']:.4f}, {p['max']:.4f}]")

    return tensor, norm_params


# ============================================================================
# STEP 9 -- Validation & sanity plots
# ============================================================================

def validate_and_plot(tensor, norm_params, valid_mask, grid_gdf):
    """Print stats and save per-channel sanity plots."""
    H, W, C = tensor.shape

    print("\n" + "=" * 60)
    print("VALIDATION -- per-channel statistics")
    print("=" * 60)
    print(f"  Tensor shape : ({H}, {W}, {C})")
    print(f"  Valid cells  : {valid_mask.sum()}")
    print(f"  Any NaN      : {np.isnan(tensor).any()}")

    for i, ch in enumerate(GRID_CFG.channels):
        layer = tensor[:, :, i]
        masked = layer[valid_mask]
        print(f"  [{i}] {ch:25s}  "
              f"min={masked.min():.4f}  max={masked.max():.4f}  "
              f"mean={masked.mean():.4f}  std={masked.std():.4f}")

    # Save sanity plots
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()

    for i, ch in enumerate(GRID_CFG.channels):
        layer = tensor[:, :, i].copy()
        layer[~valid_mask] = np.nan
        ax = axs[i]
        im = ax.imshow(layer, cmap="viridis")
        ax.set_title(ch, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # Hide extra subplot
    if len(GRID_CFG.channels) < len(axs):
        for j in range(len(GRID_CFG.channels), len(axs)):
            axs[j].axis("off")

    plt.suptitle("Channel Sanity Check", fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "channel_sanity_all.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n[preprocess] Saved sanity plot to {plot_path}")


# ============================================================================
# STEP 5-8 -- Master pipeline
# ============================================================================

def preprocess_all(data_dir: str = None, use_cache: bool = True):
    """
    Full preprocessing pipeline.

    Returns
    -------
    grid_gdf, tensor, norm_params, valid_mask
    """
    cache_path = os.path.join(CACHE_DIR, "preprocessed.pkl")

    if use_cache and os.path.exists(cache_path):
        print("[preprocess] Loading from cache ...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # ==== STEP 1: Load datasets =============================================
    print("=" * 70)
    print("STEP 1: Loading datasets")
    print("=" * 70)
    data = data_loader.load_all()
    boundary_gdf = data["boundary"]

    # ==== STEP 2: Create grid ===============================================
    print("\n" + "=" * 70)
    print("STEP 2: Creating 100m x 100m grid")
    print("=" * 70)
    grid_gdf = create_grid(boundary_gdf)

    # ==== STEP 3-6: Rasterise features ======================================
    print("\n" + "=" * 70)
    print("STEP 3-6: Rasterising & engineering features")
    print("=" * 70)

    # --- Channel 1: Population density ---
    print("\n[Channel 1/7] Population density")
    pop_grid = rasterize_population(grid_gdf, data["demographics"],
                                     boundary_gdf)

    # --- Channel 2: LST ---
    print("\n[Channel 2/7] Land Surface Temperature (Landsat)")
    lst_grid = rasterize_lst(grid_gdf, data["lst"], boundary_gdf)

    # --- Channel 3: PM2.5 ---
    print("\n[Channel 3/7] PM2.5 (interpolated from AQS)")
    pm25_grid = rasterize_pm25(grid_gdf, data["pollution"])

    # --- Channel 5: Green space (needed before impervious fallback) ---
    print("\n[Channel 5/7] Green space indicator")
    green_grid = rasterize_green_space_from_nlcd(
        grid_gdf, data["nlcd_lc"], data["open_space"], boundary_gdf)

    # --- Channel 4: Impervious surface ---
    print("\n[Channel 4/7] Impervious surface")
    imp_grid, nlcd_ok = rasterize_impervious_from_nlcd(
        grid_gdf, data["nlcd_imp"], boundary_gdf)
    if not nlcd_ok:
        print("  Falling back to impervious proxy (green space + roads)")
        imp_grid = rasterize_impervious_proxy(
            grid_gdf, green_grid, data["roads"])

    # --- Channel 6: Traffic intensity ---
    print("\n[Channel 6/7] Traffic intensity (AADT)")
    traffic_grid = rasterize_traffic(grid_gdf, data["traffic"])

    # --- Channel 7: Feasibility mask ---
    print("\n[Channel 7/7] Land feasibility mask")
    feasibility_grid = rasterize_feasibility(
        grid_gdf, green_grid, data["preserved"],
        lc_data=data["nlcd_lc"], boundary_gdf=boundary_gdf)

    # --- Refine population using impervious proxy ---
    print("\n[Refinement] Population x impervious surface")
    pop_grid = refine_population_with_impervious(
        pop_grid, imp_grid, grid_gdf, data["demographics"])

    # ==== STEP 7-8: Build feature tensor ====================================
    print("\n" + "=" * 70)
    print("STEP 7-8: Building normalised feature tensor")
    print("=" * 70)

    channel_arrays = {
        "population_density": pop_grid,
        "lst":                lst_grid,
        "pm25":               pm25_grid,
        "impervious":         imp_grid,
        "green_space":        green_grid,
        "traffic":            traffic_grid,
        "feasibility":        feasibility_grid,
    }

    tensor, norm_params = build_feature_tensor(channel_arrays, grid_gdf)

    # Valid mask: cells inside boundary
    rows = grid_gdf.attrs["rows"]
    cols = grid_gdf.attrs["cols"]
    valid_mask = np.zeros((rows, cols), dtype=bool)
    for _, cell in grid_gdf.iterrows():
        valid_mask[cell["row"], cell["col"]] = True

    # ==== STEP 9: Validation ================================================
    print("\n" + "=" * 70)
    print("STEP 9: Validation & sanity checks")
    print("=" * 70)
    validate_and_plot(tensor, norm_params, valid_mask, grid_gdf)

    # Save tensor as .npy
    npy_path = os.path.join(RESULTS_DIR, "feature_tensor.npy")
    np.save(npy_path, tensor)
    print(f"[preprocess] Saved tensor to {npy_path}")

    # Save metadata
    meta = {
        "shape": list(tensor.shape),
        "channels": list(GRID_CFG.channels),
        "norm_params": norm_params,
        "grid": {
            "rows": rows,
            "cols": cols,
            "resolution": grid_gdf.attrs["resolution"],
            "origin_x": grid_gdf.attrs["origin_x"],
            "origin_y": grid_gdf.attrs["origin_y"],
            "crs": TARGET_CRS,
        },
        "valid_cells": int(valid_mask.sum()),
        "feasible_cells": int(np.sum(feasibility_grid == 1)),
    }
    meta_path = os.path.join(RESULTS_DIR, "tensor_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[preprocess] Saved metadata to {meta_path}")

    # Pack result and cache
    result = (grid_gdf, tensor, norm_params, valid_mask)

    print(f"\n[preprocess] Caching results to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n[OK] Preprocessing complete.")
    return result


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    grid_gdf, tensor, norm_params, valid_mask = preprocess_all(use_cache=False)
    print(f"\nFinal tensor: {tensor.shape}")
    print(f"Valid cells: {valid_mask.sum()}")
    feas_idx = GRID_CFG.channels.index("feasibility")
    feas_layer = tensor[:, :, feas_idx]
    print(f"Feasible cells: {int((feas_layer[valid_mask] > 0.5).sum())}")
