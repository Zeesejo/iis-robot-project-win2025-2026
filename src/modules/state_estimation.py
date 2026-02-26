import numpy as np
import math
from src.robot import sensor_wrapper  # PROVIDED - DO NOT CHANGE

# Robot physical constants
WHEEL_RADIUS   = 0.1     # meters
WHEEL_BASELINE = 0.45    # distance between left and right wheels (2 * 0.225)

# Noise tuning – small values keep particles tight around odometry
_SIGMA_V     = 0.05   # fractional linear-velocity noise  (5 %)
_SIGMA_OMEGA = 0.10   # fractional angular-velocity noise (10 %)
_SIGMA_V_MIN     = 0.005   # floor when stationary
_SIGMA_OMEGA_MIN = 0.010

# [F44] Tightened from 0.3 -> 0.15 m for faster wall-anchoring.
# With 36 rays all contributing, a tighter sigma gives sharper weight peaks
# and faster convergence without starving low-probability particles.
_SIGMA_LIDAR = 0.15


class ParticleFilter:
    def __init__(self, num_particles=500, map_size=10.0):
        self.num_particles = num_particles
        self.map_size      = map_size       # 10 m x 10 m room

        # Initialise particles tightly around the known start pose (0, 0, 0)
        self.particles = np.zeros((num_particles, 3))
        self.particles[:, 0] = np.random.normal(0.0, 0.1, num_particles)
        self.particles[:, 1] = np.random.normal(0.0, 0.1, num_particles)
        self.particles[:, 2] = np.random.normal(0.0, 0.05, num_particles)
        self._wrap_angles()

        self.weights    = np.ones(num_particles) / num_particles
        self.robot_pose = np.array([0.0, 0.0, 0.0])

    # -- helpers ----------------------------------------------------------

    def _wrap_angles(self):
        self.particles[:, 2] = (
            (self.particles[:, 2] + math.pi) % (2 * math.pi)
        ) - math.pi

    def _clip_positions(self):
        half = self.map_size / 2.0
        self.particles[:, 0] = np.clip(self.particles[:, 0], -half, half)
        self.particles[:, 1] = np.clip(self.particles[:, 1], -half, half)

    # -- predict ----------------------------------------------------------

    def predict(self, wheel_velocities, dt=1.0 / 240.0):
        """
        Differential-drive motion update.

        Accepts either:
          * 4 wheel angular velocities [fl, fr, bl, br]  (rad/s)
          * [v, omega] already in m/s and rad/s
        """
        if len(wheel_velocities) == 4:
            left_avg  = (wheel_velocities[0] + wheel_velocities[2]) / 2.0
            right_avg = (wheel_velocities[1] + wheel_velocities[3]) / 2.0
            v     = (left_avg + right_avg) / 2.0 * WHEEL_RADIUS
            omega = (right_avg - left_avg)  / WHEEL_BASELINE * WHEEL_RADIUS
        elif len(wheel_velocities) >= 2:
            v, omega = wheel_velocities[0], wheel_velocities[1]
        else:
            v = omega = 0.0

        n = self.num_particles
        sigma_v     = max(_SIGMA_V_MIN,     _SIGMA_V     * abs(v))
        sigma_omega = max(_SIGMA_OMEGA_MIN, _SIGMA_OMEGA * abs(omega))

        dv     = v     + np.random.normal(0, sigma_v,     n)
        domega = omega + np.random.normal(0, sigma_omega, n)

        # Use the heading at the START of the step for position integration
        theta_old = self.particles[:, 2].copy()

        self.particles[:, 2] += domega * dt
        self._wrap_angles()

        # Use mid-step heading for more accurate arc integration
        theta_mid = (theta_old + self.particles[:, 2]) / 2.0
        self.particles[:, 0] += dv * np.cos(theta_mid) * dt
        self.particles[:, 1] += dv * np.sin(theta_mid) * dt
        self._clip_positions()

    # -- measurement update -----------------------------------------------

    def measurement_update(self, sensors):
        """
        Vectorised weight update using all 36 lidar rays vs. wall model.

        The room is axis-aligned, so the expected wall distance in direction a
        from position (px, py) is the minimum positive slab distance.
        Using all 36 rays gives the filter much more information to anchor
        the pose against wall geometry, reducing drift.
        """
        if 'lidar' not in sensors:
            return

        lidar_data = np.array(sensors['lidar'], dtype=float)
        n_rays     = len(lidar_data)
        if n_rays == 0:
            return

        half = self.map_size / 2.0

        # Ray angles in robot frame: ray 0 = forward (0 deg), evenly spaced CCW
        ray_angles_rel = np.linspace(0, 2 * math.pi, n_rays, endpoint=False)

        # Particles: shape (P, 3)
        px    = self.particles[:, 0]   # (P,)
        py    = self.particles[:, 1]
        theta = self.particles[:, 2]

        # Absolute ray angles for every particle: (P, R)
        abs_angles = theta[:, None] + ray_angles_rel[None, :]  # broadcast

        cos_a = np.cos(abs_angles)   # (P, R)
        sin_a = np.sin(abs_angles)

        # Wall slab distances - vectorised
        with np.errstate(divide='ignore', invalid='ignore'):
            dx_pos = np.where(cos_a >  1e-6, ( half - px[:, None]) / cos_a, np.inf)
            dx_neg = np.where(cos_a < -1e-6, (-half - px[:, None]) / cos_a, np.inf)
            dy_pos = np.where(sin_a >  1e-6, ( half - py[:, None]) / sin_a, np.inf)
            dy_neg = np.where(sin_a < -1e-6, (-half - py[:, None]) / sin_a, np.inf)

        # Nearest positive wall hit per (particle, ray)
        d_min = np.minimum(
            np.minimum(dx_pos, dx_neg),
            np.minimum(dy_pos, dy_neg)
        )
        d_min = np.clip(d_min, 0.0, half * math.sqrt(2))

        # Gaussian likelihood: compare expected vs. measured
        err   = d_min - lidar_data[None, :]          # (P, R)
        log_w = -0.5 * np.sum(err ** 2, axis=1) / (_SIGMA_LIDAR ** 2)

        # Numerically stable weight update
        log_w -= log_w.max()
        self.weights = np.exp(log_w)
        self.weights += 1e-200
        self.weights /= self.weights.sum()

    # -- resample ---------------------------------------------------------

    def resample(self):
        """Systematic (low-variance) resampling."""
        n   = self.num_particles
        cdf = np.cumsum(self.weights)
        u0  = np.random.uniform(0, 1.0 / n)
        us  = u0 + np.arange(n) / n
        indices = np.searchsorted(cdf, us)
        self.particles = self.particles[indices]
        self.weights   = np.ones(n) / n

    # -- estimate ---------------------------------------------------------

    def estimate_pose(self):
        """Weighted mean pose [x, y, theta]."""
        x = np.dot(self.weights, self.particles[:, 0])
        y = np.dot(self.weights, self.particles[:, 1])
        sin_t = np.dot(self.weights, np.sin(self.particles[:, 2]))
        cos_t = np.dot(self.weights, np.cos(self.particles[:, 2]))
        theta = np.arctan2(sin_t, cos_t)
        self.robot_pose = np.array([x, y, theta])
        return self.robot_pose


# -- module-level API ---------------------------------------------------------

pf = None


def initialize_state_estimator():
    """Call once at simulation start."""
    global pf
    pf = ParticleFilter(num_particles=500)


def state_estimate(sensors, control_inputs):
    """
    Main API called by cognitive_architecture.py every simulation step.

    Args:
        sensors:        dict with 'lidar', 'imu', 'joint_states' keys
        control_inputs: dict with 'wheel_left' and 'wheel_right' (rad/s)

    Returns:
        np.array([x, y, theta])  - estimated robot pose in world frame
    """
    global pf
    if pf is None:
        initialize_state_estimator()

    wheel_left  = control_inputs.get('wheel_left',  0.0)
    wheel_right = control_inputs.get('wheel_right', 0.0)

    # Convert wheel angular velocities (rad/s) -> [v (m/s), omega (rad/s)]
    v     = (wheel_left + wheel_right) / 2.0 * WHEEL_RADIUS
    omega = (wheel_right - wheel_left) / WHEEL_BASELINE * WHEEL_RADIUS

    pf.predict(np.array([v, omega]))
    pf.measurement_update(sensors)
    pf.resample()

    return pf.estimate_pose()
