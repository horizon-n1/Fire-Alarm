import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from pathlib import Path
from typing import Optional, Tuple
import logging

# ── Optional raster support (only needed if you have GeoTIFFs) ──────────────
try:
    import rasterio
    from rasterio.transform import rowcol
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logging.warning("rasterio not installed. Elevation features will be skipped.")

logger = logging.getLogger(__name__)

# ── Directory layout ─────────────────────────────────────────────────────────
RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & clean the raw FIRMS CSV
# ════════════════════════════════════════════════════════════════════════════

def load_firms_csv(filename: str = "active_fires.csv") -> pd.DataFrame:
    """
    Loads the raw FIRMS CSV, casts types, and drops unusable rows.

    Key columns we care about:
        latitude, longitude  — spatial position of each fire pixel
        bright_ti4           — brightness temperature (K), proxy for fire intensity
        frp                  — Fire Radiative Power (MW), proxy for fuel consumption rate
        confidence           — VIIRS confidence: 'low' | 'nominal' | 'high'
        acq_date, acq_time   — acquisition timestamp for temporal ordering
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}. Run ingestion.py first.")

    df = pd.read_csv(path)

    # ── Type coercion ────────────────────────────────────────────────────────
    float_cols = ["latitude", "longitude", "bright_ti4", "frp", "scan", "track"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Combine date + time into a single sortable timestamp
    if "acq_date" in df.columns and "acq_time" in df.columns:
        df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)  # e.g. "930" → "0930"
        df["timestamp"] = pd.to_datetime(
            df["acq_date"] + " " + df["acq_time"],
            format="%Y-%m-%d %H%M",
            errors="coerce"
        )

    # ── Filtering ────────────────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude", "bright_ti4"])

    # Drop low-confidence detections — they add noise to the graph
    if "confidence" in df.columns:
        df = df[df["confidence"].isin(["nominal", "high", "n", "h"])]

    logger.info(f"Loaded {before} rows → {len(df)} after cleaning.")
    return df.reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Discretise the continuous lat/lon points onto a regular grid
# ════════════════════════════════════════════════════════════════════════════

def build_grid(
    df: pd.DataFrame,
    cell_size_deg: float = 0.01          # ~1 km at mid-latitudes
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Snaps each fire point to the nearest grid cell centre.

    Returns:
        grid_df   — one row per unique (grid_row, grid_col), with aggregated features
        lat_ticks — 1-D array of unique latitude  centres
        lon_ticks — 1-D array of unique longitude centres
    """
    df = df.copy()
    df["grid_row"] = np.floor(df["latitude"]  / cell_size_deg).astype(int)
    df["grid_col"] = np.floor(df["longitude"] / cell_size_deg).astype(int)

    agg = {
        "latitude":   "mean",
        "longitude":  "mean",
        "bright_ti4": "max",   # peak brightness in cell
        "frp":        "sum",   # total fire radiative power
    }
    # Only aggregate columns that exist
    agg = {k: v for k, v in agg.items() if k in df.columns}

    grid_df = (
        df.groupby(["grid_row", "grid_col"], as_index=False)
          .agg(agg)
          .reset_index(drop=True)
    )

    # Normalise row/col to 0-based indices
    grid_df["row_idx"] = grid_df["grid_row"] - grid_df["grid_row"].min()
    grid_df["col_idx"] = grid_df["grid_col"] - grid_df["grid_col"].min()

    lat_ticks = np.sort(grid_df["latitude"].unique())
    lon_ticks = np.sort(grid_df["longitude"].unique())

    logger.info(f"Grid: {grid_df['row_idx'].max()+1} rows × {grid_df['col_idx'].max()+1} cols, "
                f"{len(grid_df)} active cells.")
    return grid_df, lat_ticks, lon_ticks


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — (Optional) Attach elevation from a GeoTIFF
# ════════════════════════════════════════════════════════════════════════════

def attach_elevation(grid_df: pd.DataFrame, dem_path: Optional[str] = None) -> pd.DataFrame:
    """
    Samples a Digital Elevation Model (DEM) GeoTIFF at each grid cell centre
    and attaches the elevation (metres) and derived slope (rise/run).

    If no DEM is provided or rasterio is unavailable, fills with zeros so the
    rest of the pipeline continues uninterrupted.
    """
    grid_df = grid_df.copy()

    if dem_path is None or not RASTERIO_AVAILABLE:
        grid_df["elevation"] = 0.0
        grid_df["slope"]     = 0.0
        logger.warning("No DEM provided — elevation & slope set to 0.")
        return grid_df

    with rasterio.open(dem_path) as src:
        lats = grid_df["latitude"].values
        lons = grid_df["longitude"].values

        # Convert geographic coords → pixel indices
        rows, cols = rowcol(src.transform, lons, lats)
        rows = np.clip(rows, 0, src.height - 1)
        cols = np.clip(cols, 0, src.width  - 1)

        elev_band = src.read(1).astype(float)
        elev_band[elev_band == src.nodata] = 0.0

        grid_df["elevation"] = elev_band[rows, cols]

    # Estimate slope via finite differences on the elevation column
    # (simplified — for full 2-D slope you'd use scipy or richdem)
    grid_df = grid_df.sort_values(["row_idx", "col_idx"])
    elev_arr = grid_df["elevation"].values
    slope    = np.gradient(elev_arr)
    grid_df["slope"] = slope

    logger.info("Elevation and slope attached.")
    return grid_df


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Build node feature matrix
# ════════════════════════════════════════════════════════════════════════════

def build_node_features(grid_df: pd.DataFrame) -> torch.Tensor:
    """
    Assembles the node feature matrix X of shape [N, F].

    Feature layout (index → meaning):
        0  bright_ti4_norm   normalised brightness temperature  (fire intensity proxy)
        1  frp_norm          normalised Fire Radiative Power    (fuel consumption proxy)
        2  elevation_norm    normalised elevation               (topographic effect)
        3  slope_norm        normalised terrain slope           (spread-rate modifier)
        4  is_on_fire        binary label — 1 if cell is active fire pixel
    """
    def minmax(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0.0

    feature_cols = {}

    feature_cols["bright_ti4_norm"] = minmax(grid_df.get("bright_ti4", pd.Series(0.0, index=grid_df.index)))
    feature_cols["frp_norm"]        = minmax(grid_df.get("frp",        pd.Series(0.0, index=grid_df.index)))
    feature_cols["elevation_norm"]  = minmax(grid_df.get("elevation",  pd.Series(0.0, index=grid_df.index)))
    feature_cols["slope_norm"]      = minmax(grid_df.get("slope",      pd.Series(0.0, index=grid_df.index)))

    # Every node in grid_df IS an active fire detection → label = 1
    feature_cols["is_on_fire"] = pd.Series(1.0, index=grid_df.index)

    X = torch.tensor(
        pd.DataFrame(feature_cols).values,
        dtype=torch.float32
    )
    logger.info(f"Node feature matrix: {X.shape}  (nodes × features)")
    return X


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Build edge index + edge attributes (the "graph wiring")
# ════════════════════════════════════════════════════════════════════════════

def build_edges(
    grid_df: pd.DataFrame,
    wind_u: float = 0.0,
    wind_v: float = 0.0,
    connectivity: int = 8,
    radius: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:

    pos  = grid_df[["row_idx", "col_idx"]].values
    elev = grid_df["elevation"].values if "elevation" in grid_df.columns else np.zeros(len(grid_df))

    rc_to_idx = {(int(r), int(c)): i for i, (r, c) in enumerate(pos)}

    offsets = [
        (dr, dc)
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if not (dr == 0 and dc == 0)
    ]

    wind_vec = np.array([wind_u, wind_v], dtype=np.float32)
    wind_mag = np.linalg.norm(wind_vec) + 1e-8

    src_list, dst_list, attr_list = [], [], []

    for i, (r, c) in enumerate(pos):
        for dr, dc in offsets:
            j = rc_to_idx.get((int(r + dr), int(c + dc)))
            if j is None:
                continue

            dist          = np.sqrt(dr**2 + dc**2)
            slope_grad    = float(elev[j] - elev[i])

            spread_dir      = np.array([dc, dr], dtype=np.float32)
            spread_dir_norm = spread_dir / (np.linalg.norm(spread_dir) + 1e-8)
            wind_alignment  = float(np.dot(wind_vec / wind_mag, spread_dir_norm))

            src_list.append(i)
            dst_list.append(j)
            attr_list.append([dist, slope_grad, wind_alignment])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attr_list,            dtype=torch.float32)

    edge_attr[:, 0] = edge_attr[:, 0] / edge_attr[:, 0].max()

    logger.info(f"Edges: {edge_index.shape[1]} (connectivity={connectivity}, radius={radius})")
    return edge_index, edge_attr


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — Assemble and save the PyTorch Geometric Data object
# ════════════════════════════════════════════════════════════════════════════

def build_graph(
    firms_csv:  str   = "active_fires.csv",
    dem_path:   Optional[str] = None,
    wind_u:     float = 0.0,
    wind_v:     float = 0.0,
    cell_size:  float = 0.01,
    radius:     int   = 2, 
    output_name: str  = "fire_graph.pt"
) -> Data:
    """
    End-to-end pipeline: CSV → cleaned grid → graph → saved .pt file.

    Args:
        firms_csv   : filename inside data/raw/
        dem_path    : optional path to a GeoTIFF DEM
        wind_u      : West→East wind component  (m/s)
        wind_v      : South→North wind component (m/s)
        cell_size   : grid resolution in degrees (~0.01° ≈ 1 km)
        output_name : filename to write inside data/processed/

    Returns:
        PyTorch Geometric Data object ready for model.py
    """
    logger.info("═══ Processor pipeline start ═══")

    df                       = load_firms_csv(firms_csv)
    grid_df, _, _            = build_grid(df, cell_size_deg=cell_size)
    grid_df                  = attach_elevation(grid_df, dem_path)
    x                        = build_node_features(grid_df)
    edge_index, edge_attr    = build_edges(grid_df, wind_u=wind_u, wind_v=wind_v, radius=radius)

    # Node positions in lat/lon — used by the Mapbox/Leaflet frontend
    pos = torch.tensor(
        grid_df[["latitude", "longitude"]].values,
        dtype=torch.float32
    )

    graph = Data(
        x          = x,           # [N, 5]  node features
        edge_index = edge_index,  # [2, E]  graph connectivity
        edge_attr  = edge_attr,   # [E, 3]  physics edge weights
        pos        = pos,         # [N, 2]  geographic coordinates
        num_nodes  = x.shape[0],
    )

    out_path = PROCESSED_DIR / output_name
    torch.save(graph, out_path)
    logger.info(f"Graph saved → {out_path}")
    logger.info(f"  Nodes : {graph.num_nodes}")
    logger.info(f"  Edges : {graph.num_edges}")
    logger.info(f"  Node features : {graph.x.shape[1]}")
    logger.info(f"  Edge features : {graph.edge_attr.shape[1]}")
    logger.info("═══ Processor pipeline complete ═══")

    return graph


# ════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    graph = build_graph(
        firms_csv  = "active_fires.csv",
        dem_path   = None,          # swap in your GeoTIFF path when you have one
        wind_u     = 3.5,           # example: mild eastward wind
        wind_v     = 1.2,
        cell_size  = 0.01,
        radius = 5,
        output_name= "fire_graph.pt"
    )

    print(graph)