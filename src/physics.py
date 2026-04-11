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

if __name__ == "__main__":
    print("Running Physics Engine Sanity Check...")
    
    # Create dummy tensors with requires_grad=True to track backpropagation
    dummy_base_ros = torch.tensor([2.5, 1.0, 0.3], requires_grad=True)
    dummy_wind = torch.tensor([5.0, 10.0, 0.0], requires_grad=True)
    dummy_slope = torch.tensor([0.1, 0.3, -0.1], requires_grad=True)
    dummy_moisture = torch.tensor([0.05, 0.15, 0.20], requires_grad=True)
    
    # Execute physics calculation
    predicted_ros = calculate_rothermel_ros(
        dummy_base_ros, dummy_wind, dummy_slope, dummy_moisture
    )
    
    print(f"Base ROS Input:  {dummy_base_ros.detach().numpy()}")
    print(f"Calculated ROS:  {predicted_ros.detach().numpy()}")
    
    # Verify PyTorch gradient graph integrity
    pseudo_loss = predicted_ros.sum()
    pseudo_loss.backward()
    
    if dummy_wind.grad is not None:
        print("SUCCESS: Gradients are flowing. Engine is GNN-compatible.")
    else:
        print("ERROR: Gradient graph broken.")