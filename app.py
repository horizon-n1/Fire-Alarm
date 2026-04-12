import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import torch
import json
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import time
import plotly.graph_objects as go

from src.model import PyroFlowGNN
from src import physics

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

# ════════════════════════════════════════════════════════════════════════════
# Fire Spread Animation
# ════════════════════════════════════════════════════════════════════════════

def simulate_spread_steps(results, edge_index, edge_attr, n_steps=8):
    import numpy as np
    frames     = []
    probs      = results["fire_prob"].values.copy()
    lats       = results["lat"].values
    lons       = results["lon"].values
    BOOST      = 0.20
    THRESH     = 0.3
    RADIUS_KM  = 8.0    # nodes within 80 km of a fire node can ignite

    def haversine_matrix(lats, lons):
        """Returns NxN distance matrix in km."""
        R    = 6371.0
        lat  = np.radians(lats)
        lon  = np.radians(lons)
        dlat = lat[:, None] - lat[None, :]
        dlon = lon[:, None] - lon[None, :]
        a    = np.sin(dlat/2)**2 + np.cos(lat[:,None]) * np.cos(lat[None,:]) * np.sin(dlon/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    dist_matrix = haversine_matrix(lats, lons)   # [N, N]

    for step in range(n_steps):
        new_probs = probs.copy()

        for i in range(len(probs)):
            if probs[i] >= THRESH:
                # Find all nodes within RADIUS_KM
                neighbours = np.where(dist_matrix[i] < RADIUS_KM)[0]
                for j in neighbours:
                    if i == j:
                        continue
                    # Closer nodes get stronger boost
                    dist_factor = 1.0 - (dist_matrix[i, j] / RADIUS_KM)
                    boost       = BOOST * probs[i] * dist_factor
                    new_probs[j] = min(new_probs[j] + boost, 1.0)

        probs = new_probs
        frame = results.copy()
        frame["fire_prob"] = probs
        frame["step"]      = step + 1
        frames.append(frame)

    return frames

def build_animation_map(df: pd.DataFrame, step: int, total: int) -> folium.Map:
    """Builds a single animation frame map."""
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    m = folium.Map(
        location   = [center_lat, center_lon],
        zoom_start = 7,
        tiles      = "CartoDB dark_matter",
    )

    # Step counter overlay
    folium.Marker(
        location = [df["lat"].max(), df["lon"].min()],
        icon     = folium.DivIcon(html=f"""
            <div style="
                background: rgba(0,0,0,0.7);
                color: #ff4b2b;
                font-size: 16px;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 6px;
                border: 1px solid #ff4b2b;
                white-space: nowrap;
            ">
                ⏱ T+{step} hrs &nbsp;|&nbsp; Step {step}/{total}
            </div>
        """)
    ).add_to(m)

    for _, row in df.iterrows():
        prob   = row["fire_prob"]
        rate   = max(row["spread_rate"], 0.5)
        radius = min(rate * 3, 20)

        if prob > 0.75:
            color = "#ff0000"
        elif prob > 0.5:
            color = "#ff4b2b"
        elif prob > 0.3:
            color = "#ff9900"
        elif prob > 0.15:
            color = "#ffdd00"
        else:
            color = "#0033aa"

        folium.CircleMarker(
            location     = [row["lat"], row["lon"]],
            radius       = radius * prob,   # shrinks for low-prob nodes
            color        = color,
            fill         = True,
            fill_color   = color,
            fill_opacity = min(prob + 0.2, 0.9),
            tooltip      = f"prob={prob:.2f} | rate={rate:.2f} m/min",
        ).add_to(m)

    return m

def build_animation_map_plotly(df, step, total):
    fig = go.Figure(go.Scattermapbox(
        lat  = df["lat"],
        lon  = df["lon"],
        mode = "markers",
        marker = dict(
            size       = (df["fire_prob"] * 15 + 3).clip(3, 18),
            color      = df["fire_prob"],
            colorscale = "Reds",
            cmin       = 0,
            cmax       = 1,
            colorbar   = dict(title="Fire Prob"),
        ),
        text = df.apply(
            lambda r: f"prob={r['fire_prob']:.2f} | rate={r['spread_rate']:.2f} m/min",
            axis=1
        ),
    ))

    fig.update_layout(
        mapbox = dict(
            style  = "carto-darkmatter",
            center = dict(lat=df["lat"].mean(), lon=df["lon"].mean()),
            zoom   = 6,
        ),
        margin           = dict(l=0, r=0, t=40, b=0),
        height           = 460,
        title            = f"T+{step} hrs | Step {step}/{total}",
        title_font_color = "#ff4b2b",
        paper_bgcolor    = "#0e1117",
        plot_bgcolor     = "#0e1117",
    )
    return fig
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
        tab1, tab2 = st.tabs(["📍 Live Prediction Map", "🎬 Spread Animation"])

        with tab1:
            st.caption("Red: Active Fire | Orange: High Risk | Blue: Safe Area")
            fire_map = build_map(results, threshold=threshold)
            st_folium(fire_map, width=700, height=480)

        with tab2:
            st.caption("Simulates fire propagation across the graph over time.")
            n_steps = st.slider("Simulation Steps (hours)", 4, 16, 8, key="n_steps")

            cache_key = f"frames_{n_steps}_{wind_speed}_{wind_dir}_{humidity}"
            if "frames_cache_key" not in st.session_state or st.session_state.frames_cache_key != cache_key:
                st.session_state.frames           = simulate_spread_steps(
                    results    = results,
                    edge_index = graph.edge_index,
                    edge_attr  = graph.edge_attr,
                    n_steps    = n_steps,
                )
                st.session_state.frames_cache_key = cache_key
                st.session_state.anim_step        = 0
                st.session_state.playing          = False

            frames = st.session_state.frames

            col_play, col_stop, col_step = st.columns([1, 1, 3])
            with col_play:
                if st.button("▶ Play", type="primary", use_container_width=True):
                    st.session_state.playing   = True
                    st.session_state.anim_step = 0
            with col_stop:
                if st.button("⏹ Stop", use_container_width=True):
                    st.session_state.playing = False

            with col_step:
                manual_step = st.slider(
                    "Manual Step", 1, n_steps,
                    value = max(st.session_state.get("anim_step", 0) + 1, 1),
                    key   = "manual_step_slider"
                )

            # Determine which frame to show
            if st.session_state.get("playing", False):
                current_step = st.session_state.anim_step
            else:
                current_step = manual_step - 1

            # Render current frame
            frame = frames[current_step]
            fig   = build_animation_map_plotly(frame, step=current_step+1, total=n_steps)
            st.plotly_chart(fig, use_container_width=True)

            n_burning = (frame["fire_prob"] > 0.5).sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Burning Nodes", n_burning)
            c2.metric("Coverage",      f"{n_burning / len(frame) * 100:.1f}%")
            c3.metric("Time",          f"T+{current_step+1} hrs")

            # If playing, advance one frame then trigger a rerun
            if st.session_state.get("playing", False):
                next_step = st.session_state.anim_step + 1
                if next_step >= n_steps:
                    st.session_state.playing   = False
                    st.session_state.anim_step = 0
                else:
                    st.session_state.anim_step = next_step
                    time.sleep(1.2)
                    st.rerun()
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
        # Replace this:
        st.bar_chart(
            results["spread_rate"].clip(0, 10).value_counts(bins=10).sort_index()
        )

        # With this:
        spread_hist = pd.cut(
            results["spread_rate"].clip(0, 10),
            bins = 10
        ).value_counts().sort_index()
        spread_df = pd.DataFrame({
            "range": [str(i) for i in spread_hist.index],
            "count": spread_hist.values
        })
        st.bar_chart(spread_df.set_index("range"))

    # --- Raw Data Inspection ---
    with st.expander("View Raw Node Predictions"):
        st.dataframe(
            results.sort_values("fire_prob", ascending=False).round(4),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()