import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, BatchNorm
from torch_geometric.data import Data
from pathlib import Path

try:
    from src import constants as const
    from src import physics
except ImportError:
    import constants as const
    import physics

# ════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — Edge Feature Injection
# PyTorch Geometric's SAGEConv doesn't natively use edge_attr, so we
# project edge features into the node space before each conv layer.
# ════════════════════════════════════════════════════════════════════════════

class EdgeFeatureInjector(nn.Module):
    """
    Projects edge attributes [E, edge_dim] into edge-weighted messages
    that get summed into destination nodes before graph convolution.

    This is how we inject slope + wind_alignment into message passing
    without leaving the standard PyG API.
    """
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )

    def forward(
        self,
        x:          torch.Tensor,   # [N, node_dim]
        edge_index: torch.Tensor,   # [2, E]
        edge_attr:  torch.Tensor    # [E, edge_dim]
    ) -> torch.Tensor:
        src, dst    = edge_index
        edge_msg    = self.edge_mlp(edge_attr)          # [E, node_dim]

        # Weight each message by wind_alignment (col 2 of edge_attr)
        # High wind_alignment = fire more likely to spread this direction
        wind_weight = torch.sigmoid(edge_attr[:, 2:3])  # [E, 1]  range (0,1)
        edge_msg    = edge_msg * wind_weight             # scale by physics

        # Scatter-sum into destination nodes
        out = torch.zeros_like(x)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)
        return x + out                                  # residual


# ════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — GNN Encoder
# Three-layer GraphSAGE with edge injection between every layer.
# GraphSAGE was chosen over GCN because it aggregates neighbour features
# independently of node degree — important for sparse fire graphs where
# isolated nodes would get washed out by degree normalisation.
# ════════════════════════════════════════════════════════════════════════════

class PyroGNNEncoder(nn.Module):
    """
    Encodes each node into a latent fire-state embedding.

    Architecture:
        EdgeInject → SAGEConv → BN → ReLU → Dropout
        EdgeInject → SAGEConv → BN → ReLU → Dropout
        EdgeInject → SAGEConv → BN → ReLU
    """
    def __init__(
        self,
        in_channels:  int = 5,      # matches node feature dim from processor.py
        hidden_dim:   int = 64,
        out_dim:      int = 32,
        edge_dim:     int = 3,      # matches edge feature dim from processor.py
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Edge injectors — one per layer
        self.edge_inj1 = EdgeFeatureInjector(in_channels,  edge_dim)
        self.edge_inj2 = EdgeFeatureInjector(hidden_dim,   edge_dim)
        self.edge_inj3 = EdgeFeatureInjector(hidden_dim,   edge_dim)

        # Graph convolution layers
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim,  hidden_dim)
        self.conv3 = SAGEConv(hidden_dim,  out_dim)

        # Batch normalisation stabilises training on small graphs
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(out_dim)

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor
    ) -> torch.Tensor:

        # Layer 1
        x = self.edge_inj1(x, edge_index, edge_attr)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.edge_inj2(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3
        x = self.edge_inj3(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        return x  # [N, out_dim]


# ════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — Spread Predictor Head
# Takes the latent embedding and predicts:
#   (a) fire probability at T+1  (primary task)
#   (b) spread rate in m/min     (auxiliary task, supervised by Rothermel)
# ════════════════════════════════════════════════════════════════════════════

class SpreadPredictorHead(nn.Module):
    """
    Two-headed MLP decoder:
        fire_prob  — sigmoid output, binary cross-entropy target
        spread_rate — ReLU output (non-negative), Rothermel physics target
    """
    def __init__(self, in_dim: int = 32):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
        )

        # Head A — will this node be on fire at T+1?
        self.fire_head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            # No sigmoid here — applied in loss for numerical stability
        )

        # Head B — how fast is fire spreading (m/min)?
        self.rate_head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.ReLU(),   # spread rate is always ≥ 0
        )

    def forward(self, x: torch.Tensor):
        shared      = self.shared(x)
        fire_logit  = self.fire_head(shared).squeeze(-1)   # [N]
        spread_rate = self.rate_head(shared).squeeze(-1)   # [N]  m/min
        return fire_logit, spread_rate


# ════════════════════════════════════════════════════════════════════════════
# BLOCK 4 — Physics-Informed Loss
# The key differentiator from a standard GNN classifier.
# Total loss = BCE(fire prediction) + λ * Rothermel_penalty
# ════════════════════════════════════════════════════════════════════════════

class PhysicsInformedLoss(nn.Module):
    """
    Combines:
        L_data    — Binary Cross-Entropy on fire probability predictions
        L_physics — Penalty when predicted spread rate violates Rothermel

    L_total = L_data + lambda_physics * L_physics

    The Rothermel penalty works by:
        1. Computing the physics-based spread rate from node features
        2. Penalising the squared difference between the GNN's predicted
           spread rate and what Rothermel says it should be
        3. This forces the GNN to internalise physical fire behaviour
           rather than just fitting historical patterns
    """
    def __init__(self, lambda_physics: float = 0.5):
        super().__init__()
        self.lambda_physics = lambda_physics

    def forward(
        self,
        fire_logit:       torch.Tensor,   # [N]   raw logits from fire_head
        spread_rate_pred: torch.Tensor,   # [N]   predicted spread (m/min)
        fire_labels:      torch.Tensor,   # [N]   ground truth 0/1
        node_features:    torch.Tensor,   # [N,5] raw node features
        edge_attr:        torch.Tensor,   # [E,3] for wind extraction
    ) -> tuple:

        # ── Data loss ───────────────────────────────────────────────────────
        L_data = F.binary_cross_entropy_with_logits(
            fire_logit,
            fire_labels.float()
        )

        # ── Physics loss ─────────────────────────────────────────────────────
        # Extract slope from node features (index 3 = slope_norm)
        slope_norm = node_features[:, 3]

        # Extract wind speed proxy from edge_attr — take mean wind_alignment
        # across all edges as a scalar wind influence per-graph
        wind_speed_proxy = edge_attr[:, 2].mean().item()

        # Compute Rothermel spread rate for each node
        # We use brightness temperature as a fuel proxy (index 0)
        bright_norm = node_features[:, 0]

        rothermel_rate = physics.compute_spread_rate(
            wind_speed  = wind_speed_proxy * 10.0,  # scale to m/s range
            slope       = slope_norm.detach().cpu().numpy(),
            fuel_moisture = 0.08,                   # 8% — dry summer default
        )

        rothermel_tensor = torch.tensor(
            rothermel_rate,
            dtype  = torch.float32,
            device = spread_rate_pred.device
        )

        # Penalise deviation from Rothermel prediction
        L_physics = F.mse_loss(spread_rate_pred, rothermel_tensor)

        L_total = L_data + self.lambda_physics * L_physics

        return L_total, L_data, L_physics


# ════════════════════════════════════════════════════════════════════════════
# BLOCK 5 — Full PyroFlow Model (assembles all blocks)
# ════════════════════════════════════════════════════════════════════════════

class PyroFlowGNN(nn.Module):
    """
    Full physics-informed GNN for wildfire spread prediction.

    Forward pass:
        graph → GNN encoder → spread predictor → (fire_logit, spread_rate)

    Usage:
        model  = PyroFlowGNN()
        logits, rates = model(graph.x, graph.edge_index, graph.edge_attr)
    """
    def __init__(
        self,
        in_channels:     int   = 5,
        hidden_dim:      int   = 64,
        out_dim:         int   = 32,
        edge_dim:        int   = 3,
        dropout:         float = 0.3,
        lambda_physics:  float = 0.5,
    ):
        super().__init__()

        self.encoder = PyroGNNEncoder(
            in_channels = in_channels,
            hidden_dim  = hidden_dim,
            out_dim     = out_dim,
            edge_dim    = edge_dim,
            dropout     = dropout,
        )
        self.head = SpreadPredictorHead(in_dim=out_dim)
        self.loss_fn = PhysicsInformedLoss(lambda_physics=lambda_physics)

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor
    ):
        embedding              = self.encoder(x, edge_index, edge_attr)
        fire_logit, spread_rate = self.head(embedding)
        return fire_logit, spread_rate

    def compute_loss(
        self,
        fire_logit:       torch.Tensor,
        spread_rate_pred: torch.Tensor,
        fire_labels:      torch.Tensor,
        node_features:    torch.Tensor,
        edge_attr:        torch.Tensor,
    ):
        return self.loss_fn(
            fire_logit, spread_rate_pred,
            fire_labels, node_features, edge_attr
        )


# ════════════════════════════════════════════════════════════════════════════
# Smoke test — verifies shapes are correct before trainer.py
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

    print("Loading graph...")
    graph = torch.load(PROCESSED_DIR / "fire_graph.pt", weights_only=False)
    print(f"Graph: {graph}")

    print("\nInitialising PyroFlowGNN...")
    model = PyroFlowGNN(
        in_channels    = graph.x.shape[1],       # 5
        edge_dim       = graph.edge_attr.shape[1], # 3
        lambda_physics = 0.5,
    )
    print(model)

    print("\nRunning forward pass...")
    model.train()
    fire_logit, spread_rate = model(graph.x, graph.edge_index, graph.edge_attr)

    # All nodes are labelled 1 (active fire) in this single-snapshot graph
    labels = graph.x[:, 4]   # is_on_fire column from processor.py

    L_total, L_data, L_physics = model.compute_loss(
        fire_logit, spread_rate, labels, graph.x, graph.edge_attr
    )

    print(f"\n✓ fire_logit  : {fire_logit.shape}")
    print(f"✓ spread_rate : {spread_rate.shape}")
    print(f"✓ L_data      : {L_data.item():.4f}")
    print(f"✓ L_physics   : {L_physics.item():.4f}")
    print(f"✓ L_total     : {L_total.item():.4f}")
    print("\nmodel.py smoke test passed — ready for trainer.py")