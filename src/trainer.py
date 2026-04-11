import torch
import torch.optim as optim
from torch_geometric.data import Data
from pathlib import Path
import logging
import json
from datetime import datetime

try:
    from src.model import PyroFlowGNN
except ImportError:
    from model import PyroFlowGNN

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — Temporal graph splitting
# Our single graph snapshot needs to be split into T and T+1 pairs.
# We simulate this by masking nodes — half the graph is "current state",
# the other half provides ground truth labels for the next timestep.
# ════════════════════════════════════════════════════════════════════════════

def create_temporal_split(graph: Data, train_ratio: float = 0.8):
    """
    Splits nodes into train/val masks for supervised learning.

    Since we have a single-snapshot graph (not a time series), we treat
    this as a transductive node classification problem:
        - Train on 80% of nodes with known fire labels
        - Validate on remaining 20%

    In a production system you'd have multiple timestep snapshots and
    split by time instead. For the hackathon this is the correct approach.
    """
    N = graph.num_nodes
    perm = torch.randperm(N)

    train_size  = int(N * train_ratio)
    train_idx   = perm[:train_size]
    val_idx     = perm[train_size:]

    train_mask  = torch.zeros(N, dtype=torch.bool)
    val_mask    = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True

    return train_mask, val_mask


# ════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — Single epoch
# ════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model:      PyroFlowGNN,
    graph:      Data,
    optimizer:  torch.optim.Optimizer,
    train_mask: torch.Tensor,
    device:     torch.device,
) -> dict:
    model.train()
    optimizer.zero_grad()

    x          = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    edge_attr  = graph.edge_attr.to(device)
    labels     = graph.x[:, 4].to(device)   # is_on_fire column

    fire_logit, spread_rate = model(x, edge_index, edge_attr)

    # Compute loss only on training nodes
    L_total, L_data, L_physics = model.compute_loss(
        fire_logit[train_mask],
        spread_rate[train_mask],
        labels[train_mask],
        x[train_mask],
        edge_attr,
    )

    L_total.backward()

    # Gradient clipping — essential for small graphs where gradients spike
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    # Compute training accuracy
    with torch.no_grad():
        preds    = (torch.sigmoid(fire_logit[train_mask]) > 0.5).float()
        accuracy = (preds == labels[train_mask]).float().mean().item()

    return {
        "loss":       L_total.item(),
        "L_data":     L_data.item(),
        "L_physics":  L_physics.item(),
        "accuracy":   accuracy,
    }


# ════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — Validation
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    model:    PyroFlowGNN,
    graph:    Data,
    val_mask: torch.Tensor,
    device:   torch.device,
) -> dict:
    model.eval()

    x          = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    edge_attr  = graph.edge_attr.to(device)
    labels     = graph.x[:, 4].to(device)

    fire_logit, spread_rate = model(x, edge_index, edge_attr)

    L_total, L_data, L_physics = model.compute_loss(
        fire_logit[val_mask],
        spread_rate[val_mask],
        labels[val_mask],
        x[val_mask],
        edge_attr,
    )

    preds    = (torch.sigmoid(fire_logit[val_mask]) > 0.5).float()
    accuracy = (preds == labels[val_mask]).float().mean().item()

    # Precision / recall — more meaningful than accuracy for fire detection
    tp = ((preds == 1) & (labels[val_mask] == 1)).sum().item()
    fp = ((preds == 1) & (labels[val_mask] == 0)).sum().item()
    fn = ((preds == 0) & (labels[val_mask] == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "val_loss":      L_total.item(),
        "val_L_data":    L_data.item(),
        "val_L_physics": L_physics.item(),
        "val_accuracy":  accuracy,
        "val_precision": precision,
        "val_recall":    recall,
        "val_f1":        f1,
    }


# ════════════════════════════════════════════════════════════════════════════
# BLOCK 4 — Full training loop
# ════════════════════════════════════════════════════════════════════════════

def train(
    graph_path:      str   = "fire_graph.pt",
    epochs:          int   = 200,
    lr:              float = 1e-3,
    weight_decay:    float = 1e-4,
    hidden_dim:      int   = 64,
    lambda_physics:  float = 0.5,
    train_ratio:     float = 0.8,
    patience:        int   = 30,      # early stopping
    save_best:       bool  = True,
) -> dict:
    """
    Full training loop with early stopping and model checkpointing.

    Args:
        graph_path     : filename inside data/processed/
        epochs         : max training epochs
        lr             : Adam learning rate
        weight_decay   : L2 regularisation
        hidden_dim     : GNN hidden layer width
        lambda_physics : weight of Rothermel physics loss
        train_ratio    : fraction of nodes used for training
        patience       : early stopping patience (epochs without improvement)
        save_best      : whether to checkpoint the best model

    Returns:
        history dict with per-epoch metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load graph ───────────────────────────────────────────────────────
    graph_file = PROCESSED_DIR / graph_path
    if not graph_file.exists():
        raise FileNotFoundError(f"Graph not found: {graph_file}. Run processor.py first.")

    graph = torch.load(graph_file, weights_only=False)
    logger.info(f"Graph loaded: {graph}")

    # ── Masks ────────────────────────────────────────────────────────────
    train_mask, val_mask = create_temporal_split(graph, train_ratio)
    logger.info(f"Train nodes: {train_mask.sum().item()} | Val nodes: {val_mask.sum().item()}")

    # ── Model ────────────────────────────────────────────────────────────
    model = PyroFlowGNN(
        in_channels    = graph.x.shape[1],
        edge_dim       = graph.edge_attr.shape[1],
        hidden_dim     = hidden_dim,
        lambda_physics = lambda_physics,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr           = lr,
        weight_decay = weight_decay,
    )

    # Cosine annealing — smoothly decays LR, works well for small graphs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ────────────────────────────────────────────────────
    history       = []
    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None

    logger.info("═══ Training start ═══")

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, graph, optimizer, train_mask, device)
        val_metrics   = validate(model, graph, val_mask, device)
        scheduler.step()

        row = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(row)

        # ── Logging ──────────────────────────────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:>4} | "
                f"loss={train_metrics['loss']:.4f} "
                f"L_phy={train_metrics['L_physics']:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} "
                f"val_f1={val_metrics['val_f1']:.4f} "
                f"val_recall={val_metrics['val_recall']:.4f}"
            )

        # ── Early stopping ────────────────────────────────────────────────
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            patience_ctr  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # ── Save best model ───────────────────────────────────────────────────
    if save_best and best_state is not None:
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = MODELS_DIR / f"pyroflow_{timestamp}.pt"

        torch.save({
            "model_state":  best_state,
            "model_config": {
                "in_channels":    graph.x.shape[1],
                "edge_dim":       graph.edge_attr.shape[1],
                "hidden_dim":     hidden_dim,
                "lambda_physics": lambda_physics,
            },
            "best_val_loss": best_val_loss,
            "epochs_trained": len(history),
        }, model_path)

        logger.info(f"Best model saved → {model_path}")

        # Also save as latest for app.py to load
        latest_path = MODELS_DIR / "pyroflow_latest.pt"
        torch.save(torch.load(model_path, weights_only=False), latest_path)

    # ── Save training history ─────────────────────────────────────────────
    history_path = MODELS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved → {history_path}")

    final = history[-1]
    logger.info("═══ Training complete ═══")
    logger.info(f"Best val_loss : {best_val_loss:.4f}")
    logger.info(f"Final val_f1  : {final['val_f1']:.4f}")
    logger.info(f"Final val_recall : {final['val_recall']:.4f}")

    return history


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    history = train(
        graph_path     = "fire_graph.pt",
        epochs         = 200,
        lr             = 1e-3,
        lambda_physics = 0.5,
        patience       = 30,
    )