import torch
import numpy as np
import src.constants as const
try:
    from src import constants as const      # works when called from project root
except ImportError:
    import constants as const               # works when running physics.py directly

def calculate_slope_factor(slope_gradient):
    """Calculate terrain slope multiplier."""
    return const.SLOPE_COEFF * torch.pow(slope_gradient, 2)

def calculate_wind_factor(wind_speed):
    """Calculate wind speed multiplier."""
    return const.WIND_COEFF * torch.pow(wind_speed, 1.5)

def calculate_rothermel_ros(base_ros, wind_speed, slope_gradient, moisture):
    """Calculate the final Rate of Spread (m/min)."""
    phi_s = calculate_slope_factor(slope_gradient)
    phi_w = calculate_wind_factor(wind_speed)
    
    # Apply exponential moisture dampening
    moisture_damping = torch.exp(const.MOISTURE_COEFF * moisture)
    
    ros = base_ros * (1.0 + phi_w + phi_s) * moisture_damping
    
    # Ensure Rate of Spread is non-negative
    return torch.nn.functional.relu(ros)


def compute_spread_rate(
    wind_speed:    float,              # m/s
    slope,                             # scalar or np.ndarray (normalised 0-1)
    fuel_moisture: float = 0.08,       # fraction, e.g. 0.08 = 8%
) -> np.ndarray:
    """
    Simplified Rothermel fire spread rate model.
    Returns spread rate in m/min for each node.

    R = R0 * phi_wind * phi_slope / xi_moisture

    Where:
        R0          — base spread rate from fuel properties
        phi_wind    — wind multiplier
        phi_slope   — slope multiplier (uphill accelerates fire)
        xi_moisture — moisture damping (wet fuel burns slower)
    """
    slope = np.atleast_1d(np.array(slope, dtype=np.float32))

    # ── Base spread rate (m/min) ─────────────────────────────────────────
    # Derived from Rothermel 1972 for medium grass fuel (fuel model 2)
    R0 = 0.5   # m/min baseline

    # ── Wind multiplier ──────────────────────────────────────────────────
    # Rothermel: phi_w = C * U^B where U is wind speed
    # Simplified coefficients for grass fuel
    C, B = 0.4, 1.5
    phi_wind = 1.0 + C * (max(wind_speed, 0.0) ** B)

    # ── Slope multiplier ─────────────────────────────────────────────────
    # phi_s = 1 + 5.275 * beta^(-0.3) * tan(theta)^2
    # Simplified: linear scaling of normalised slope
    phi_slope = 1.0 + 3.0 * np.clip(slope, 0.0, 1.0)

    # ── Moisture damping ─────────────────────────────────────────────────
    # Fire dies out above ~30% moisture content
    fuel_moisture = np.clip(fuel_moisture, 0.01, 0.30)
    xi_moisture   = 1.0 + 5.0 * fuel_moisture   # higher moisture = lower rate

    spread_rate = (R0 * phi_wind * phi_slope) / xi_moisture

    return spread_rate.astype(np.float32)