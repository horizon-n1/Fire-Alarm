import streamlit as st
import torch
import json
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
from pathlib import Path
import sys

# --- Path resolution ---
# Ensures the application can find the source directory regardless of execution point
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from src.model import PyroFlowGNN
    from src import physics
except ImportError:
    # Fallback for different directory structures
    from model import PyroFlowGNN
    import physics

# --- Global Paths ---
MODELS_DIR    = ROOT / "data" / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
HISTORY_PATH  = MODELS_DIR / "training_history.json"
GRAPH_PATH    = PROCESSED_DIR / "fire_graph.pt"
MODEL_PATH    = MODELS_DIR / "pyroflow_latest.pt"


# ==============================================================================
# Page Configuration and Custom CSS
# ==============================================================================

st.set_page_config(
    page_title = "PyroFlow - Wildfire Spread GNN",
    page_icon  = "none", # Removed emoji icon
    layout     = "wide",
)

# Custom styling for a dark, professional "Command Center" aesthetic
st.markdown("""
    <style>
        .main { background-color: #0e1117; }
        .block-container { padding-top: 1rem; }
        h1 { color: #ff4b2b; }
        .metric-label { font-size: 0.85rem; color: #aaa; }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# Cached Resource Loaders
# ==============================================================================

@st.cache_resource
def load_model_and_graph():
    """
    Loads the trained GNN and the fire graph into memory once.
    Using cache_resource prevents re-loading heavy model weights on every UI interaction.
    """
    if not MODEL_PATH.exists():
        return None, None

    # Load checkpoint from disk; map to CPU for general accessibility
    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location="cpu")
    config     = checkpoint["model_config"]

    # Initialize model architecture based on saved config
    model = PyroFlowGNN(
        in_channels    = config["in_channels"],
        edge_dim       = config["edge_dim"],
        hidden_dim     = config["hidden_dim"],
        lambda_physics = config["lambda_physics"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval() # Set to evaluation mode (disables dropout, etc.)

    graph = torch.load(GRAPH_PATH, weights_only=False, map_location="cpu")
    return model, graph


@st.cache_data
def load_history():
    """Loads JSON training history for the loss curve visualization."""
    if not HISTORY_PATH.exists():
        return []
    with open(HISTORY_PATH) as f:
        return json.load(f)


# ==============================================================================
# GNN Inference Logic
# ==============================================================================

@torch.no_grad()
def run_inference(
    model:      PyroFlowGNN,
    graph,
    wind_speed: float,
    wind_dir:   float,    # degrees, 0 = North, 90 = East
    humidity:   float,    # 0.0 to 1.0
) -> pd.DataFrame:
    """
    Re-runs the GNN forward pass with updated environmental parameters.
    This effectively acts as a 'Physics Surrogate' allowing real-time simulation.
    """
    # Convert wind direction to Cartesian components for vector math
    wind_rad = np.radians(wind_dir)
    wind_u   = wind_speed * np.sin(wind_rad)   # East component
    wind_v   = wind_speed * np.cos(wind_rad)   # North component

    # Create a copy of edges to update attributes without mutating the original graph
    edge_attr = graph.edge_attr.clone()

    # Recompute wind alignment for every edge in the graph
    # This teaches the GNN how fire spreads in relation to current wind vectors
    edge_index = graph.edge_index
    pos        = graph.pos.numpy()

    wind_vec = np.array([wind_u, wind_v], dtype=np.float32)
    wind_mag = np.linalg.norm(wind_vec) + 1e-8

    for e in range(edge_index.shape[1]):
        src, dst    = edge_index[0, e].item(), edge_index[1, e].item()
        spread_dir  = pos[dst] - pos[src]           # Physical vector between nodes
        spread_norm = spread_dir / (np.linalg.norm(spread_dir) + 1e-8)
        
        # Dot product determines how much the wind 'pushes' fire along this specific edge
        alignment   = float(np.dot(wind_vec / wind_mag, spread_norm))
        edge_attr[e, 2] = alignment

    # Node Feature Modification: scale intensity by inverse humidity
    # Lower humidity results in higher brightness/intensity features
    x = graph.x.clone()
    moisture_scale      = 1.0 - (humidity * 0.6)
    x[:, 0]             = x[:, 0] * moisture_scale  # TI4 brightness normalized
    x[:, 1]             = x[:, 1] * moisture_scale  # Fire Radiative Power (FRP)

    # Perform the forward pass through the GNN
    fire_logit, spread_rate = model(x, edge_index, edge_attr)
    fire_prob = torch.sigmoid(fire_logit).numpy()

    lats = graph.pos[:, 0].numpy()
    lons = graph.pos[:, 1].numpy()
    labels = graph.x[:, 4].numpy()

    return pd.DataFrame({
        "lat":         lats,
        "lon":         lons,
        "fire_prob":   fire_prob,
        "spread_rate": spread_rate.numpy(),
        "is_fire":     labels,
    })


# ==============================================================================
# Map Visualization Builder
# ==============================================================================

def build_map(df: pd.DataFrame, threshold: float = 0.5) -> folium.Map:
    """
    Creates an interactive Folium map. 
    Node colors represent risk level; circle size represents predicted spread speed.
    """
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    m = folium.Map(
        location    = [center_lat, center_lon],
        zoom_start  = 7,
        tiles       = "CartoDB dark_matter",
    )

    for _, row in df.iterrows():
        prob  = row["fire_prob"]
        rate  = max(row["spread_rate"], 0.5)
        radius = min(rate * 3, 20)   # Caps visual radius for clarity

        if prob > threshold:
            color, fill = "#ff2200", "#ff2200"
            tooltip = f"FIRE | prob={prob:.2f} | rate={rate:.2f} m/min"
        elif prob > 0.3:
            color, fill = "#ff9900", "#ff9900"
            tooltip = f"HIGH RISK | prob={prob:.2f}"
        else:
            color, fill = "#0099ff", "#0033aa"
            tooltip = f"Safe | prob={prob:.2f}"

        folium.CircleMarker(
            location     = [row["lat"], row["lon"]],
            radius       = radius,
            color        = color,
            fill         = True,
            fill_color   = fill,
            fill_opacity = 0.7,
            tooltip      = tooltip,
        ).add_to(m)

    return m


# ==============================================================================
# Main Streamlit Dashboard Layout
# ==============================================================================

def main():
    # --- Header ---
    st.title("PyroFlow - Physics-Informed Wildfire GNN")
    st.caption("2020 California Fire Season | GNN Surrogate Model | Real-time What-If Analysis")

    model, graph = load_model_and_graph()

    # Error handling if pre-processing/training hasn't been run
    if model is None or graph is None:
        st.error("Model or graph not found. Please run trainer.py and processor.py first.")
        return

    history = load_history()

    # --- Sidebar - Environmental Simulation Controls ---
    with st.sidebar:
        st.header("What-If Controls")
        st.caption("Adjust environmental conditions to see real-time GNN re-predictions.")

        wind_speed = st.slider(
            "Wind Speed (m/s)",
            min_value = 0.0,
            max_value = 30.0,
            value     = 5.0,
            step      = 0.5,
        )

        wind_dir = st.slider(
            "Wind Direction (Degrees)",
            min_value = 0,
            max_value = 359,
            value     = 180,
            step      = 5,
            help      = "0 = North, 90 = East, 180 = South, 270 = West"
        )

        humidity = st.slider(
            "Relative Humidity (0-1)",
            min_value = 0.0,
            max_value = 1.0,
            value     = 0.15,
            step      = 0.05,
            help      = "Typical historical fire season averages were 0.10 to 0.20"
        )

        threshold = st.slider(
            "Fire Probability Threshold",
            min_value = 0.1,
            max_value = 0.9,
            value     = 0.5,
            step      = 0.05,
        )

        st.divider()
        st.markdown("**Model Specs**")
        st.markdown(f"- Nodes: {graph.num_nodes}")
        st.markdown(f"- Edges: {graph.num_edges}")
        st.markdown(f"- Architecture: 3-layer GraphSAGE")
        st.markdown(f"- Physics: Rothermel Model (1972)")

    # --- Run Inference ---
    results = run_inference(model, graph, wind_speed, wind_dir, humidity)

    # Calculate Summary Metrics
    n_fire    = (results["fire_prob"] > threshold).sum()
    avg_rate  = results.loc[results["fire_prob"] > threshold, "spread_rate"].mean()
    max_rate  = results["spread_rate"].max()
    avg_prob  = results["fire_prob"].mean()

    # --- Top Metrics Row ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Fire Nodes",    f"{n_fire} / {graph.num_nodes}")
    col2.metric("Avg Spread Rate",      f"{avg_rate:.2f} m/min" if not np.isnan(avg_rate) else "N/A")
    col3.metric("Max Spread Rate",      f"{max_rate:.2f} m/min")
    col4.metric("Avg Fire Probability", f"{avg_prob:.3f}")

    st.divider()

    # --- Main Visualization Panels ---
    map_col, chart_col = st.columns([3, 2])

    with map_col:
        st.subheader("Fire Spread Prediction Map")
        st.caption("Red: Active Fire | Orange: High Risk | Blue: Safe Area")
        fire_map = build_map(results, threshold=threshold)
        st_folium(fire_map, width=700, height=500)

    with chart_col:
        # Training analytics
        st.subheader("Training Loss Curve")
        if history:
            hist_df = pd.DataFrame(history)
            st.line_chart(
                hist_df[["epoch", "loss", "val_loss"]].set_index("epoch"),
                color=["#ff4b2b", "#ffaa00"],
            )

            st.subheader("Final Validation Metrics")
            last = history[-1]
            m1, m2, m3 = st.columns(3)
            m1.metric("F1 Score",     f"{last['val_f1']:.4f}")
            m2.metric("Recall",       f"{last['val_recall']:.4f}")
            m3.metric("Val Loss",     f"{last['val_loss']:.4f}")
        else:
            st.info("No training history JSON found in models directory.")

        st.subheader("Spread Rate Distribution")
        # Histogram of how fast the fire is predicted to move across all nodes
        st.bar_chart(
            results["spread_rate"].clip(0, 10).value_counts(bins=10).sort_index()
        )

    # --- Raw Data Inspection ---
    with st.expander("View Raw Node Predictions"):
        st.dataframe(
            results.sort_values("fire_prob", ascending=False).round(4),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()