from models.data import generate_synthetic_data
from physics import coil_field
from control import current_control


def plasma_control_loop(model, R_coil=1.0, Z0_coil=0.0, N_turns=100, initial_current=1e4):
    """Simulates real-time plasma control."""
    current = initial_current
    
    for step in range(100):  # Simulation of 100 time steps
        # 1. Measurement of plasma parameters (simulated)
        plasma_params , _ = generate_synthetic_data(samples=1)   # Simulated temporal data
        
        # 2. Disruption prediction
        disruption_probability = model.predict(plasma_params).item()
        
        # 3. Current adjustment via PID
        current_correction = current_control(disruption_probability)
        current += current_correction
        
        # 4. Calculation of new magnetic field
        Br, Bz = coil_field(r=0.5, z=0.0, R=R_coil, Z0=Z0_coil, I=current, N=N_turns)
        
        print(f"Step {step}: Disruption Prob={disruption_probability:.3f}, Current={current:.1f} A, Br={Br:.2e} T, Bz={Bz:.2e} T")
    

