import math
import random
import numpy as np
from environment.wrapper import OrbitWarsWrapper

def run_benchmark(num_samples=1000):
    config = {"shipSpeed": 6.0, "sunRadius": 10.0, "boardSize": 100.0}
    wrapper = OrbitWarsWrapper(config)
    center = config["boardSize"] / 2.0
    
    hits = 0
    total_error = 0.0
    valid_samples = 0
    
    print(f"Running Aiming Benchmark with {num_samples} samples...")
    
    for _ in range(num_samples):
        # 1. Generate Source Planet
        s_rad = random.uniform(15, 45)
        s_angle = random.uniform(0, 2 * math.pi)
        source_pos = (center + s_rad * math.cos(s_angle), center + s_rad * math.sin(s_angle))
        source_radius = random.uniform(1.0, 3.0)
        
        # 2. Generate Target Planet
        t_rad = random.uniform(15, 45)
        t_angle = random.uniform(0, 2 * math.pi)
        t_ang_vel = random.uniform(0.01, 0.05) * random.choice([-1, 1])
        target_radius = random.uniform(1.0, 3.0)
        
        target_data = {
            'id': 999,
            'x': center + t_rad * math.cos(t_angle),
            'y': center + t_rad * math.sin(t_angle),
            'radius': target_radius,
            'owner': -1,
            'production': 0,
            'ships': 10,
            'angular_velocity': t_ang_vel,
            'source_ships': random.randint(20, 1000)
        }
        
        obs = {'planet_angular_velocities': {999: t_ang_vel}, 'comets': []}
        
        # 3. Calculate Aim Angle
        angle, t_est, tx_est, ty_est = wrapper.get_intercept_params(source_pos, source_radius, target_data, 1.0, obs)
        
        # 4. Check Path Safety (Sun Collision)
        dist_est = math.hypot(tx_est - source_pos[0], ty_est - source_pos[1])
        if not wrapper.is_path_safe(source_pos[0], source_pos[1], angle, dist_est):
            continue # Skip samples that would hit the sun
            
        valid_samples += 1
        
        # 5. Verification: Where is the planet REALLY at t_est?
        real_tx, real_ty = wrapper.predict_future_position(target_data, t_est, obs)
        
        # 6. Verification: Where is the fleet REALLY at t_est?
        # Fleet starts at source SURFACE.
        fleet_speed = wrapper.calculate_speed(target_data['source_ships'])
        # In reality, dist traveled = speed * t_est
        # Let's check the distance between fleet end pos and planet center
        dist_to_target = math.hypot(real_tx - (source_pos[0] + math.cos(angle) * (source_radius + fleet_speed * t_est)),
                                    real_ty - (source_pos[1] + math.sin(angle) * (source_radius + fleet_speed * t_est)))
        
        # A hit is if the fleet (point) is inside the target planet's radius
        if dist_to_target <= target_radius + 0.5: # 0.5 margin for discrete steps
            hits += 1
        
        total_error += dist_to_target
        
    if valid_samples > 0:
        accuracy = (hits / valid_samples) * 100
        avg_error = total_error / valid_samples
        print(f"\nResults:")
        print(f"  Valid Samples (no sun collision): {valid_samples}")
        print(f"  Hits: {hits}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Avg Error (dist): {avg_error:.4f}")
    else:
        print("No valid samples generated.")

if __name__ == "__main__":
    run_benchmark()
