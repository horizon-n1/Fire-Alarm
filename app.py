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
from folium.plugins import HeatMap
from streamlit_folium import st_folium

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
    page_icon  = "🔥",
    layout     = "wide",
)

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
    if not MODEL_PATH.exists():
        return None, None

    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location="cpu")
    config     = checkpoint["model_config"]

    model = PyroFlowGNN(
        in_channels    = config["in_channels"],
        edge_dim       = config["edge_dim"],
        hidden_dim     = config["hidden_dim"],
        lambda_physics = config["lambda_physics"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    graph = torch.load(GRAPH_PATH, weights_only=False, map_location="cpu")
    return model, graph


@st.cache_data
def load_history():
    if not HISTORY_PATH.exists():
        return []
    with open(HISTORY_PATH) as f:
        return json.load(f)


# ==============================================================================
# GNN Inference
# ==============================================================================

@torch.no_grad()
def run_inference(model, graph, wind_speed, wind_dir, humidity):
    wind_rad = np.radians(wind_dir)
    wind_u   = wind_speed * np.sin(wind_rad)
    wind_v   = wind_speed * np.cos(wind_rad)

    edge_attr  = graph.edge_attr.clone()
    edge_index = graph.edge_index
    pos        = graph.pos.numpy()

    wind_vec = np.array([wind_u, wind_v], dtype=np.float32)
    wind_mag = np.linalg.norm(wind_vec) + 1e-8

    for e in range(edge_index.shape[1]):
        src, dst    = edge_index[0, e].item(), edge_index[1, e].item()
        spread_dir  = pos[dst] - pos[src]
        spread_norm = spread_dir / (np.linalg.norm(spread_dir) + 1e-8)
        alignment   = float(np.dot(wind_vec / wind_mag, spread_norm))
        edge_attr[e, 2] = alignment

    x = graph.x.clone()
    moisture_scale = 1.0 - (humidity * 0.6)
    x[:, 0]        = x[:, 0] * moisture_scale
    x[:, 1]        = x[:, 1] * moisture_scale

    fire_logit, spread_rate = model(x, edge_index, edge_attr)
    fire_prob = torch.sigmoid(fire_logit).numpy()

    return pd.DataFrame({
        "lat":         graph.pos[:, 0].numpy(),
        "lon":         graph.pos[:, 1].numpy(),
        "fire_prob":   fire_prob,
        "spread_rate": spread_rate.numpy(),
        "is_fire":     graph.x[:, 4].numpy(),
    })


# ==============================================================================
# Map Builders
# ==============================================================================

def build_map(df: pd.DataFrame, threshold: float = 0.5) -> folium.Map:
    m = folium.Map(
        location   = [df["lat"].mean(), df["lon"].mean()],
        zoom_start = 7,
        tiles      = "CartoDB dark_matter",
    )

    for _, row in df.iterrows():
        prob   = row["fire_prob"]
        rate   = max(row["spread_rate"], 0.5)
        radius = min(rate * 3, 20)

        if prob > threshold:
            color = "#ff2200"
            tooltip = f"FIRE | prob={prob:.2f} | rate={rate:.2f} m/min"
        elif prob > 0.3:
            color = "#ff9900"
            tooltip = f"HIGH RISK | prob={prob:.2f}"
        else:
            color = "#0099ff"
            tooltip = f"Safe | prob={prob:.2f}"

        folium.CircleMarker(
            location     = [row["lat"], row["lon"]],
            radius       = radius,
            color        = color,
            fill         = True,
            fill_color   = color,
            fill_opacity = 0.7,
            tooltip      = tooltip,
        ).add_to(m)

    return m


def build_heatmap(df: pd.DataFrame) -> folium.Map:
    m = folium.Map(
        location   = [float(df["lat"].mean()), float(df["lon"].mean())],
        zoom_start = 6,
        tiles      = "OpenStreetMap",   # ← switch from CartoDB
    )

    heat_data = [
        [float(row["lat"]), float(row["lon"]), float(row["fire_prob"])]
        for _, row in df.iterrows()
    ]

    HeatMap(
        data        = heat_data,
        min_opacity = 0.5,
        radius      = 25,
        blur        = 15,
        max_val     = 1.0,
    ).add_to(m)

    return m


# ==============================================================================
# Main Layout
# ==============================================================================

def main():
    st.title("🔥 PyroFlow - Physics-Informed Wildfire GNN")
    st.caption("2020 California Fire Season | GNN Surrogate Model | Real-time What-If Analysis")

    model, graph = load_model_and_graph()

    if model is None or graph is None:
        st.error("Model or graph not found. Please run trainer.py and processor.py first.")
        return

    history = load_history()

    # --- Sidebar ---
    with st.sidebar:
        st.header("What-If Controls")
        st.caption("Adjust environmental conditions to see real-time GNN re-predictions.")

        wind_speed = st.slider("Wind Speed (m/s)",         0.0,  30.0, 5.0,  0.5)
        wind_dir   = st.slider("Wind Direction (Degrees)",   0,   359, 180,    5,
                               help="0=North, 90=East, 180=South, 270=West")
        humidity   = st.slider("Relative Humidity (0-1)",  0.0,  1.0,  0.15, 0.05,
                               help="2020 CA fire season avg: 0.10–0.20")
        threshold  = st.slider("Fire Probability Threshold", 0.1, 0.9, 0.5,  0.05)

        st.divider()
        st.markdown("**Model Specs**")
        st.markdown(f"- Nodes: `{graph.num_nodes}`")
        st.markdown(f"- Edges: `{graph.num_edges}`")
        st.markdown(f"- Architecture: `3-layer GraphSAGE`")
        st.markdown(f"- Physics: `Rothermel (1972)`")

    # --- Inference ---
    results  = run_inference(model, graph, wind_speed, wind_dir, humidity)
    n_fire   = (results["fire_prob"] > threshold).sum()
    avg_rate = results.loc[results["fire_prob"] > threshold, "spread_rate"].mean()
    max_rate = results["spread_rate"].max()
    avg_prob = results["fire_prob"].mean()

    # --- Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Fire Nodes",    f"{n_fire} / {graph.num_nodes}")
    col2.metric("Avg Spread Rate",      f"{avg_rate:.2f} m/min" if not np.isnan(avg_rate) else "N/A")
    col3.metric("Max Spread Rate",      f"{max_rate:.2f} m/min")
    col4.metric("Avg Fire Probability", f"{avg_prob:.3f}")

    st.divider()

    # --- Main Panels ---
    map_col, chart_col = st.columns([3, 2])

    with map_col:
        tab1, tab2 = st.tabs(["📍 Prediction Map", "🌡️ Confidence Heatmap"])

        with tab1:
            st.caption("Red: Active Fire | Orange: High Risk | Blue: Safe")
            st_folium(build_map(results, threshold), width=700, height=480)

        with tab2:
            st.caption("Heatmap intensity = GNN fire probability confidence")
            heat_map = build_heatmap(results)
            from streamlit_folium import folium_static
            folium_static(heat_map, width=700, height=480)

    with chart_col:
        st.subheader("Training Loss Curve")
        if history:
            hist_df = pd.DataFrame(history)
            st.line_chart(
                hist_df[["epoch", "loss", "val_loss"]].set_index("epoch"),
                color = ["#ff4b2b", "#ffaa00"],
            )

            st.subheader("Final Validation Metrics")
            last = history[-1]
            m1, m2, m3 = st.columns(3)
            m1.metric("F1 Score", f"{last['val_f1']:.4f}")
            m2.metric("Recall",   f"{last['val_recall']:.4f}")
            m3.metric("Val Loss", f"{last['val_loss']:.4f}")
        else:
            st.info("No training history found.")

        st.subheader("Spread Rate Distribution")
        spread_hist = pd.cut(
            results["spread_rate"].clip(0, 10), bins=10
        ).value_counts().sort_index()
        spread_df = pd.DataFrame({
            "range": [str(i) for i in spread_hist.index],
            "count": spread_hist.values,
        })
        st.bar_chart(spread_df.set_index("range"))

    with st.expander("🔍 Raw Node Predictions"):
        st.dataframe(
            results.sort_values("fire_prob", ascending=False).round(4),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()