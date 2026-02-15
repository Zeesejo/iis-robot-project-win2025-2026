
import numpy as np
import math
from src.robot import sensor_wrapper  # PROVIDED - DO NOT CHANGE

# Robot physical constants
WHEEL_RADIUS = 0.1      # meters
WHEEL_BASELINE = 0.45   # distance between left and right wheels (2 * 0.225)


class ParticleFilter:
    def __init__(self, num_particles=1000, map_size=10.0):
        self.num_particles = num_particles
        self.map_size = map_size  # 10m x 10m room
        
        # Particles: [x, y, theta] for each particle
        # Initialize near origin (0, 0) where robot actually starts
        self.particles = np.random.normal(
            loc=[0.0, 0.0, 0.0],
            scale=[0.3, 0.3, 0.2],
            size=(num_particles, 3)
        )
        # Clip to valid room bounds: room is 10x10 centered at origin → [-5, 5]
        self.particles[:, 0] = np.clip(self.particles[:, 0], -map_size/2, map_size/2)
        self.particles[:, 1] = np.clip(self.particles[:, 1], -map_size/2, map_size/2)
        self.particles[:, 2] = ((self.particles[:, 2] + math.pi) % (2 * math.pi)) - math.pi
        
        self.weights = np.ones(num_particles) / num_particles
        
        # Robot pose estimate - start at origin
        self.robot_pose = np.array([0.0, 0.0, 0.0])
        
    def predict(self, wheel_velocities, dt=1./240.0):
        """
        Move all particles based on differential drive odometry + noise.
        
        Args:
            wheel_velocities: list of 4 wheel angular velocities [fl, fr, bl, br] in rad/s,
                              OR a [v, omega] pair (auto-detected by length).
            dt: simulation timestep
        """
        if len(wheel_velocities) == 4:
            # Convert 4 wheel velocities to [v, omega] using differential drive model
            # Left wheels: fl(0), bl(2); Right wheels: fr(1), br(3)
            left_avg = (wheel_velocities[0] + wheel_velocities[2]) / 2.0
            right_avg = (wheel_velocities[1] + wheel_velocities[3]) / 2.0
            v = (left_avg + right_avg) / 2.0 * WHEEL_RADIUS       # linear velocity (m/s)
            omega = (right_avg - left_avg) / WHEEL_BASELINE * WHEEL_RADIUS  # angular velocity (rad/s)
        elif len(wheel_velocities) >= 2:
            v = wheel_velocities[0]
            omega = wheel_velocities[1]
        else:
            v = 0.0
            omega = 0.0
    
        # Update all particles (vectorized for speed)
        noise_v = np.random.normal(0, max(0.01, 0.1 * abs(v)), self.num_particles)
        noise_omega = np.random.normal(0, max(0.02, 0.2 * abs(omega)), self.num_particles)
    
        self.particles[:, 2] += (omega + noise_omega) * dt
        self.particles[:, 2] = ((self.particles[:, 2] + math.pi) % (2 * math.pi)) - math.pi
    
        self.particles[:, 0] += (v + noise_v) * np.cos(self.particles[:, 2]) * dt
        self.particles[:, 1] += (v + noise_v) * np.sin(self.particles[:, 2]) * dt
    
        # Keep particles in room bounds [-5, 5]
        self.particles[:, 0] = np.clip(self.particles[:, 0], -self.map_size/2, self.map_size/2)
        self.particles[:, 1] = np.clip(self.particles[:, 1], -self.map_size/2, self.map_size/2)

    
    def measurement_update(self, sensors):
        """
        Update particle weights using lidar data.
        Compares expected wall distances to measured lidar distances.
        
        Args:
            sensors: dict with 'lidar' key containing list of 36 ray distances
        """
        if 'lidar' not in sensors:
            return  # No lidar data, skip update
        
        lidar_data = np.array(sensors['lidar'])
        half = self.map_size / 2.0
        
        # Use 4 cardinal direction rays for wall distance comparison
        # Ray indices for 36 rays: 0=forward, 9=left, 18=back, 27=right
        cardinal_indices = [0, 9, 18, 27]
        measured = lidar_data[cardinal_indices]
        
        for i in range(self.num_particles):
            px, py, theta = self.particles[i]
            # Expected distances to walls from particle position & heading
            # Forward (theta), Left (theta+90), Back (theta+180), Right (theta+270)
            angles = [theta, theta + math.pi/2, theta + math.pi, theta - math.pi/2]
            expected = []
            for a in angles:
                cos_a = math.cos(a)
                sin_a = math.sin(a)
                # Ray-cast to room walls [-5, 5] x [-5, 5]
                dists = []
                if cos_a > 1e-6:
                    dists.append((half - px) / cos_a)
                elif cos_a < -1e-6:
                    dists.append((-half - px) / cos_a)
                if sin_a > 1e-6:
                    dists.append((half - py) / sin_a)
                elif sin_a < -1e-6:
                    dists.append((-half - py) / sin_a)
                # Take the nearest positive wall hit
                pos_dists = [d for d in dists if d > 0]
                expected.append(min(pos_dists) if pos_dists else 5.0)
            
            expected = np.array(expected)
            # Clamp expected to lidar range
            expected = np.clip(expected, 0, 5.0)
            error = np.sum((expected - measured)**2)
            self.weights[i] = np.exp(-error / (2 * 1.0**2))
    
        self.weights += 1e-10
        self.weights /= np.sum(self.weights)

    
    def resample(self):
        """Low-variance resampling - keep best particles"""
        indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            replace=True,
            p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def estimate_pose(self):
        """Return weighted average pose [x, y, theta]"""
        x = np.sum(self.weights * self.particles[:, 0])
        y = np.sum(self.weights * self.particles[:, 1])
        # Circular mean for theta
        sin_mean = np.sum(self.weights * np.sin(self.particles[:, 2]))
        cos_mean = np.sum(self.weights * np.cos(self.particles[:, 2]))
        theta = np.arctan2(sin_mean, cos_mean)
        return np.array([x, y, theta])

# Global instance for cognitive_architecture.py
pf = None

def initialize_state_estimator():
    """Call once at simulation start"""
    global pf
    pf = ParticleFilter(num_particles=1000)

def state_estimate(sensors, control_inputs):
    """
    Main API for cognitive_architecture.py
    
    Args:
        sensors: dict with 'lidar', 'imu' keys
        control_inputs: dict with 'wheel_left' and 'wheel_right' angular velocities
    """
    global pf
    
    if pf is None:
        initialize_state_estimator()
    
    # Convert wheel velocities to [v, omega]
    wheel_left = control_inputs.get('wheel_left', 0.0)
    wheel_right = control_inputs.get('wheel_right', 0.0)
    
    v = (wheel_left + wheel_right) / 2.0 * WHEEL_RADIUS
    omega = (wheel_right - wheel_left) / WHEEL_BASELINE * WHEEL_RADIUS
    
    wheel_velocities = np.array([v, omega])
    
    # Particle filter step
    pf.predict(wheel_velocities)
    pf.measurement_update(sensors)
    pf.resample()
    
    pose = pf.estimate_pose()
    return pose

