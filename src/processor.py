"""
processor.py
Converts raw fire CSV data and topographical bounds into a PyTorch Geometric Graph.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path

# --- VS Code / Direct Run Fix ---
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
import src.constants as const

def load_fire_data(filepath):
    """Load the raw fire data CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file missing: {filepath}. Run ingestion.py first.")
    return pd.read_csv(filepath)

def create_grid_graph(df, max_grid_size=100):
    """
    Converts GPS coordinates into a spatial graph for the GNN.
    
    Args:
        df: Pandas DataFrame containing 'latitude' and 'longitude'.
        max_grid_size: Limits the matrix to prevent CPU/RAM crashes.
        
    Returns:
        torch_geometric.data.Data: The graph object.
    """
    # 1. Determine geographic boundaries
    min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
    min_lon, max_lon = df['longitude'].min(), df['longitude'].max()

    # Expand bounds slightly to give the fire room to spread
    padding = 0.05 
    min_lat -= padding; max_lat += padding
    min_lon -= padding; max_lon += padding

    # Create a grid
    grid_h = min(max_grid_size, int((max_lat - min_lat) * 111)) # ~111km per lat degree
    grid_w = min(max_grid_size, int((max_lon - min_lon) * 85))  # ~85km per lon degree at mid-latitudes
    
    num_nodes = grid_h * grid_w
    print(f"Building {grid_h}x{grid_w} spatial grid ({num_nodes} nodes)...")

    # 2. Map active fires to grid cells
    # Calculate cell indices for each fire coordinate
    df['grid_x'] = ((df['longitude'] - min_lon) / (max_lon - min_lon) * (grid_w - 1)).astype(int)
    df['grid_y'] = ((df['latitude'] - min_lat) / (max_lat - min_lat) * (grid_h - 1)).astype(int)
    
    # 3. Create Node Features (X)
    # Features per node: [is_burning, fuel_type, slope, moisture]
    # For the hackathon, we initialize with baseline values and map the active fires
    node_features = np.zeros((num_nodes, 4), dtype=np.float32)
    
    # Baseline defaults: Brush fuel (ID=2), flat slope (0.0), dry moisture (0.10)
    node_features[:, 1] = 2.0  
    node_features[:, 2] = 0.0  
    node_features[:, 3] = 0.10 

    # Map the fires onto the 'is_burning' feature (Index 0)
    for _, row in df.iterrows():
        node_idx = row['grid_y'] * grid_w + row['grid_x']
        node_features[node_idx, 0] = 1.0 # 1.0 means actively burning

    x = torch.tensor(node_features)

    # 4. Build the Edge Index (Adjacency Matrix)
    # Connect each cell to its 8 immediate neighbors (N, S, E, W, NE, NW, SE, SW)
    edges = []
    
    # Helper to convert (y, x) to flat node index
    def get_id(y, x): return y * grid_w + x

    for y in range(grid_h):
        for x in range(grid_w):
            curr_id = get_id(y, x)
            # Define 8-way neighbors
            neighbors = [
                (y-1, x), (y+1, x), (y, x-1), (y, x+1),
                (y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)
            ]
            for ny, nx in neighbors:
                if 0 <= ny < grid_h and 0 <= nx < grid_w:
                    neighbor_id = get_id(ny, nx)
                    edges.append([curr_id, neighbor_id])

    # PyTorch Geometric requires edge_index to be shape [2, num_edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 5. Package into PyG Data Object
    graph_data = Data(x=x, edge_index=edge_index)
    return graph_data

def save_graph(graph_data, filename="graph_data.pt"):
    """Saves the PyTorch Geometric graph to disk."""
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / filename
    torch.save(graph_data, file_path)
    print(f"Graph saved to {file_path}")

if __name__ == "__main__":
    print("Running Graph Processor Sanity Check...")
    
    raw_data_path = Path(__file__).parent.parent / "data" / "raw" / "active_fires.csv"
    
    try:
        df = load_fire_data(raw_data_path)
        
        # We process the graph
        graph = create_grid_graph(df)
        
        print("\nGraph Structural Integrity:")
        print(f" - Nodes: {graph.num_nodes}")
        print(f" - Edges: {graph.num_edges}")
        print(f" - Feature Dimensions: {graph.num_node_features}")
        
        active_fire_count = int(graph.x[:, 0].sum().item())
        print(f" - Active Fire Nodes Mapped: {active_fire_count}")
        
        # Save it for the GNN to pick up later
        save_graph(graph)
        print("\nSUCCESS: Graph processing pipeline is operational.")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")