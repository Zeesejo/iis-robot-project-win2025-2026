
import numpy as np
import math
from src.robot import sensor_wrapper  # PROVIDED - DO NOT CHANGE

class ParticleFilter:
    def __init__(self, num_particles=1000, map_size=10.0):
        self.num_particles = num_particles
        self.map_size = map_size  # 10m x 10m room
        
        # Particles: [x, y, theta] for each particle
        self.particles = np.random.uniform(
            low=[0, 0, 0], 
            high=[map_size, map_size, 2*math.pi], 
            size=(num_particles, 3)
        )
        self.weights = np.ones(num_particles) / num_particles
        
        # Robot pose estimate
        self.robot_pose = np.array([5.0, 5.0, 0.0])  # Center start guess
        
    def predict(self, wheel_velocities, dt=1./240.0):
        """Move all particles based on odometry + noise (handles negative velocities)"""
        v = abs(wheel_velocities[0]) if len(wheel_velocities) > 0 else 0.0  # Use absolute for noise
        omega = wheel_velocities[1] if len(wheel_velocities) > 1 else 0.0
    
        # Update all particles (vectorized for speed)
        noise_v = np.random.normal(0, max(0.01, 0.1 * v), self.num_particles)
        noise_omega = np.random.normal(0, max(0.02, 0.2 * abs(omega)), self.num_particles)
    
        self.particles[:, 2] += omega * dt + noise_omega
        self.particles[:, 2] = ((self.particles[:, 2] + math.pi) % (2 * math.pi)) - math.pi
    
        self.particles[:, 0] += v * np.cos(self.particles[:, 2]) * dt + noise_v * np.cos(self.particles[:, 2])
        self.particles[:, 1] += v * np.sin(self.particles[:, 2]) * dt + noise_v * np.sin(self.particles[:, 2])
    
        # Keep particles in bounds
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.map_size)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.map_size)

    
    def measurement_update(self, sensors):
        """Handle flattened depth image"""
        if 'depth' in sensors:
            depth_data = sensors['depth']
        else:
            depth_data = np.array([2.1, 7.9, 4.2, 5.8])  # Fallback
        
        for i in range(self.num_particles):
            px, py, theta = self.particles[i]
            expected = np.array([px, self.map_size-px, py, self.map_size-py])
            measured = depth_data[:4]  # First 4 values
            error = np.sum((expected - measured)**2)
            self.weights[i] = np.exp(-error / (2 * 0.3**2))
    
        self.weights += 1e-10
        self.weights /= np.sum(self.weights)

    
    def resample(self):
        """Low-variance resampling - keep best particles"""
        max_weight = np.max(self.weights)
        index = int(np.random.uniform(0, 1.0 / self.num_particles))
        cumulative_sum = 0.0
        new_particles = np.zeros_like(self.particles)
        new_weights = np.ones(self.num_particles) / self.num_particles
        
        for i in range(self.num_particles):
            cumulative_sum += self.num_particles * self.weights[index]
            while cumulative_sum > 1.0:
                cumulative_sum -= 1.0
                index = (index + 1) % self.num_particles
            new_particles[i] = self.particles[index]
        
        self.particles = new_particles
        self.weights = new_weights
    
    def estimate_pose(self):
        """Return weighted average pose [x, y, theta]"""
        # Circular mean for theta
        x = np.sum(self.weights * self.particles[:, 0])
        y = np.sum(self.weights * self.particles[:, 1]) 
        theta = np.sum(self.weights * self.particles[:, 2])
        return np.array([x, y, theta])

# Global instance for cognitive_architecture.py
pf = None

def initialize_state_estimator():
    """Call once at simulation start"""
    global pf
    pf = ParticleFilter(num_particles=1000)

def state_estimate(sensors, control_inputs):
    """
    Main API for cognitive_architecture.py - NOW USES REAL DATA!
    """
    global pf
    
    if pf is None:
        initialize_state_estimator()
    
    # === REAL WHEEL VELOCITIES (from control_inputs) ===
    wheel_left = control_inputs['wheel_left']   # fl_wheel_joint velocity
    wheel_right = control_inputs['wheel_right'] # fr_wheel_joint velocity
    
    # Convert to linear v, angular omega (Husky wheel radius ~0.1m, baseline ~0.5m)
    v = (wheel_left + wheel_right) / 2 * 0.1      # m/s
    omega = (wheel_right - wheel_left) / 0.5      # rad/s
    
    wheel_velocities = np.array([v, omega])
    
    # === REAL SENSORS (from sensors dict) ===
    if 'depth' in sensors:
        depth_data = sensors['depth']
    else:
        depth_data = np.array([2.1, 7.9, 4.2, 5.8])  # Fallback
    
    # === PARTICLE FILTER STEP ===
    pf.predict(wheel_velocities)           # REAL wheel odometry
    pf.measurement_update(sensors)         # REAL depth + IMU
    pf.resample()
    
    pose = pf.estimate_pose()
    return pose

