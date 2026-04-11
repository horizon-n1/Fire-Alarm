"""
constants.py
Physical constants and fuel models for the Physics-Informed GNN loss function.
Based on a simplified Rothermel Wildfire Spread Model.
"""

# --- Environment Constants ---
GRAVITY = 9.81           # Standard gravity (m/s^2)
CELL_SIZE_METERS = 100.0 # Spatial resolution per graph node

# --- Fuel Models (Simplified NFFL) ---
# base_ros_r0: Base Rate of Spread (meters/minute)
FUEL_MODELS = {
    "GRASS": {
        "id": 1,
        "base_ros_r0": 2.5,
        "heat_yield": 15000,
        "moisture_extinction": 0.12 
    },
    "BRUSH": {
        "id": 2,
        "base_ros_r0": 1.0,
        "heat_yield": 18000,
        "moisture_extinction": 0.20
    },
    "TIMBER": {
        "id": 3,
        "base_ros_r0": 0.3,
        "heat_yield": 22000,
        "moisture_extinction": 0.25
    }
}

# --- Rothermel Coefficients ---
# Equation: ROS = R0 * (1 + wind_factor + slope_factor)
WIND_COEFF = 0.05      # Wind speed multiplier
SLOPE_COEFF = 5.275    # Uphill gradient multiplier
MOISTURE_COEFF = -2.5  # Moisture exponential decay factor