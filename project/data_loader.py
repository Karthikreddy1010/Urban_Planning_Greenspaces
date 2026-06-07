"""
data_loader.py - Load raw GIS datasets for the Urban Green Space RL system.

Every loader:
  • reads from disk using GeoPandas / Rasterio / Pandas
  • reprojects vector data to the project CRS (UTM 18N)
  • returns clean objects ready for preprocess.py
"""

import glob
import os
import re
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box

from config import DATA_PATHS, LANDSAT_CFG, TARGET_CRS

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _reproject_gdf(gdf: gpd.GeoDataFrame, target_crs: str = TARGET_CRS) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to *target_crs* if needed."""
    if gdf.crs is None:
        print("  [warn] No CRS found - assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.to_epsg() != int(target_crs.split(":")[1]):
        gdf = gdf.to_crs(target_crs)
    return gdf


def _landsat_path(band_suffix: str, scene_id: str = None) -> str:
    """Construct full path to a Landsat band file."""
    scene_id = scene_id or DATA_PATHS.landsat_scene_id
    fname = f"{scene_id}_{band_suffix}.TIF"
    return os.path.join(DATA_PATHS.landsat_dir, fname)


# ── Vector loaders ───────────────────────────────────────────────────────────

def load_newark_boundary() -> gpd.GeoDataFrame:
    """Load Newark city boundary polygon → UTM 18N."""
    print("[data_loader] Loading Newark boundary ...")
    gdf = gpd.read_file(DATA_PATHS.newark_boundary)
    gdf = _reproject_gdf(gdf)
    print(f"  Boundary area ~ {gdf.geometry.area.sum() / 1e6:.2f} km^2")
    return gdf


def load_road_network() -> gpd.GeoDataFrame:
    """Load Newark road network (LineString geometries) → UTM 18N."""
    print("[data_loader] Loading road network ...")
    gdf = gpd.read_file(DATA_PATHS.road_network)
    gdf = _reproject_gdf(gdf)
    print(f"  {len(gdf)} road segments loaded")
    return gdf


def load_pollution_points() -> gpd.GeoDataFrame:
    """
    Load AQS pollution monitoring points → UTM 18N.
    Extracts PM2.5 arithmetic mean for each station.
    """
    print("[data_loader] Loading AQS pollution points ...")
    gdf = gpd.read_file(DATA_PATHS.pollution_points)
    gdf = _reproject_gdf(gdf)

    # Keep only PM2.5 records and extract useful columns
    pm25_mask = gdf["Pollutant"] == "PM2.5"
    pm25_gdf = gdf.loc[pm25_mask].copy()

    if len(pm25_gdf) == 0:
        # Fallback: use all pollutant records
        print("  [warn] No PM2.5 records - using all pollutant records as proxy")
        pm25_gdf = gdf.copy()

    # Use arithmetic mean as the representative value
    pm25_gdf["pm25_value"] = pd.to_numeric(pm25_gdf["Arithmetic Mean"], errors="coerce")
    pm25_gdf = pm25_gdf.dropna(subset=["pm25_value"])
    print(f"  {len(pm25_gdf)} PM2.5 monitoring points")
    return pm25_gdf


def load_aadt_traffic() -> gpd.GeoDataFrame:
    """Load AADT traffic data (line geometry with traffic counts) → UTM 18N."""
    print("[data_loader] Loading AADT traffic ...")
    gdf = gpd.read_file(DATA_PATHS.aadt_traffic_shp)
    gdf = _reproject_gdf(gdf)

    # Identify the AADT column (may vary by dataset)
    aadt_cols = [c for c in gdf.columns if "AADT" in c.upper() or "ADT" in c.upper()]
    if aadt_cols:
        gdf["aadt_value"] = pd.to_numeric(gdf[aadt_cols[0]], errors="coerce").fillna(0)
    else:
        # Try numeric columns as fallback
        numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "geometry"]
        if numeric_cols:
            gdf["aadt_value"] = gdf[numeric_cols[0]].fillna(0)
        else:
            gdf["aadt_value"] = 1.0  # uniform weight
    print(f"  {len(gdf)} road segments with traffic data")
    return gdf


def load_open_space() -> gpd.GeoDataFrame:
    """Load NJ Open Space polygons → UTM 18N."""
    print("[data_loader] Loading open space ...")
    gdf = gpd.read_file(DATA_PATHS.open_space_shp)
    gdf = _reproject_gdf(gdf)
    print(f"  {len(gdf)} open-space polygons")
    return gdf


def load_preserved_land() -> gpd.GeoDataFrame:
    """Load preserved / Green Acres points → UTM 18N."""
    print("[data_loader] Loading preserved land ...")
    gdf = gpd.read_file(DATA_PATHS.preserved_shp)
    gdf = _reproject_gdf(gdf)
    print(f"  {len(gdf)} preserved-land features")
    return gdf


def load_parcels(boundary_gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
    """
    Load Essex County parcels → UTM 18N.
    Optionally clip to Newark boundary for speed.
    """
    print("[data_loader] Loading parcels (Essex County) ...")
    # Prefer the GeoJSON (already has geometry) but it's 100 MB
    # Use the shapefile which is smaller on disk
    try:
        gdf = gpd.read_file(DATA_PATHS.parcels_shp)
    except Exception:
        print("  [warn] Shapefile failed - trying GeoJSON ...")
        gdf = gpd.read_file(DATA_PATHS.newark_parcels_geojson)

    gdf = _reproject_gdf(gdf)

    if boundary_gdf is not None:
        boundary_union = boundary_gdf.geometry.union_all()
        gdf = gdf[gdf.geometry.intersects(boundary_union)].copy()
        print(f"  Clipped to {len(gdf)} parcels inside Newark")
    else:
        print(f"  {len(gdf)} parcels total (Essex County)")
    return gdf


def load_demographics() -> pd.DataFrame:
    """Load Newark demographics CSV (census tract level)."""
    print("[data_loader] Loading demographics ...")
    df = pd.read_csv(DATA_PATHS.demographics_csv)
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    # Ensure GEOID is string for matching
    df["GEOID"] = df["GEOID"].astype(str)
    print(f"  {len(df)} census tracts, columns: {list(df.columns)}")
    return df


# ── Raster loaders ───────────────────────────────────────────────────────────

def load_landsat_lst(scene_id: str = None) -> tuple:
    """
    Load Landsat Surface Temperature Band 10.
    Returns (temperature_celsius_array, rasterio_transform, crs).
    """
    scene_id = scene_id or DATA_PATHS.landsat_scene_id
    path = _landsat_path("ST_B10", scene_id)
    print(f"[data_loader] Loading Landsat LST from {os.path.basename(path)} ...")

    with rasterio.open(path) as src:
        dn = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    # Apply scale/offset → Kelvin → Celsius
    mask = dn <= 0 if nodata is None else (dn == nodata) | (dn <= 0)
    temp_k = dn * LANDSAT_CFG.st_scale + LANDSAT_CFG.st_offset
    temp_c = temp_k + LANDSAT_CFG.kelvin_to_celsius
    temp_c[mask] = np.nan

    print(f"  Shape {temp_c.shape}, range [{np.nanmin(temp_c):.1f}, {np.nanmax(temp_c):.1f}] degC")
    return temp_c, transform, crs


def load_landsat_ndvi(scene_id: str = None) -> tuple:
    """
    Compute NDVI from Landsat B4 (Red) and B5 (NIR).
    Returns (ndvi_array, transform, crs).
    """
    scene_id = scene_id or DATA_PATHS.landsat_scene_id
    b4_path = _landsat_path("SR_B4", scene_id)
    b5_path = _landsat_path("SR_B5", scene_id)
    print(f"[data_loader] Computing NDVI from {os.path.basename(b4_path)} ...")

    with rasterio.open(b4_path) as src:
        red_dn = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
    with rasterio.open(b5_path) as src:
        nir_dn = src.read(1).astype(np.float32)

    # DN → surface reflectance
    red = red_dn * LANDSAT_CFG.sr_scale + LANDSAT_CFG.sr_offset
    nir = nir_dn * LANDSAT_CFG.sr_scale + LANDSAT_CFG.sr_offset

    # Mask fill values
    valid = (red_dn > 0) & (nir_dn > 0)
    denom = nir + red
    denom[denom == 0] = 1e-10
    ndvi = (nir - red) / denom
    ndvi[~valid] = np.nan

    print(f"  NDVI range [{np.nanmin(ndvi):.3f}, {np.nanmax(ndvi):.3f}]")
    return ndvi, transform, crs


def load_nlcd_raster(data_dir: str, year: int, prefix: str) -> tuple:
    """
    Load an NLCD annual raster (landcover or impervious).
    Returns (array, transform, crs).
    """
    pattern = os.path.join(data_dir, f"*{prefix}_{year}_*.tiff")
    matches = glob.glob(pattern)
    if not matches:
        # Try without prefix
        pattern = os.path.join(data_dir, f"*{year}*.tiff")
        matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No NLCD raster found for year {year} in {data_dir}")

    path = matches[0]
    print(f"[data_loader] Loading NLCD from {os.path.basename(path)} ...")

    with rasterio.open(path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs

    print(f"  Shape {data.shape}, unique values: {len(np.unique(data))}")
    return data, transform, crs


def load_nlcd_landcover(year: int = None) -> tuple:
    """Load NLCD landcover raster for the given year."""
    year = year or DATA_PATHS.nlcd_year
    return load_nlcd_raster(DATA_PATHS.nlcd_landcover_dir, year, "LndCov")


def load_nlcd_impervious(year: int = None) -> tuple:
    """Load NLCD fractional impervious surface raster for the given year."""
    year = year or DATA_PATHS.nlcd_year
    return load_nlcd_raster(DATA_PATHS.nlcd_impervious_dir, year, "FctImp")


# ── Convenience: load everything ─────────────────────────────────────────────

def load_all():
    """
    Load all datasets and return as a dict.
    Useful for one-shot preprocessing.
    """
    boundary = load_newark_boundary()
    return {
        "boundary": boundary,
        "lst": load_landsat_lst(),
        "ndvi": load_landsat_ndvi(),
        "nlcd_lc": load_nlcd_landcover(),
        "nlcd_imp": load_nlcd_impervious(),
        "roads": load_road_network(),
        "pollution": load_pollution_points(),
        "traffic": load_aadt_traffic(),
        "open_space": load_open_space(),
        "preserved": load_preserved_land(),
        "parcels": load_parcels(boundary),
        "demographics": load_demographics(),
    }


if __name__ == "__main__":
    # Quick smoke test
    data = load_all()
    print("\n[OK] All datasets loaded successfully.")
    for k, v in data.items():
        if isinstance(v, tuple):
            print(f"  {k}: array shape {v[0].shape}")
        elif isinstance(v, gpd.GeoDataFrame):
            print(f"  {k}: {len(v)} features, CRS={v.crs}")
        elif isinstance(v, pd.DataFrame):
            print(f"  {k}: {len(v)} rows")
