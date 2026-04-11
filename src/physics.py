import torch
import src.constants as const

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