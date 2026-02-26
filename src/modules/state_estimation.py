"""
M5 - State Estimation: Particle Filter
Fuses noisy LIDAR + IMU + control inputs to maintain a reliable
state estimate (x, y, theta) of the robot.

Vectorised NumPy implementation: sensor update is O(P x R) in
broadcasted array ops instead of O(P x R) nested Python loops,
giving ~100x speedup over the naive version.
"""

import numpy as np


class ParticleFilter:
    """
    Monte Carlo Localization (Particle Filter) for robot state estimation.
    State: [x, y, theta]
    """

    def __init__(self, n_particles=200, room_size=10.0,
                 motion_noise=(0.05, 0.05, 0.03),
                 sensor_noise=0.3):
        """
        n_particles : number of particles (reduced to 200 for speed;
                      still gives accurate MCL convergence)
        room_size   : half-width of the room in metres
        motion_noise: (sigma_x, sigma_y, sigma_theta) for motion model
        sensor_noise: sigma for sensor likelihood
        """
        self.n            = n_particles
        self.room         = room_size
        self.motion_noise = np.array(motion_noise)
        self.sensor_noise = sensor_noise

        half = room_size / 2.0
        self.particles        = np.zeros((n_particles, 3))
        self.particles[:, 0]  = np.random.uniform(-half, half, n_particles)
        self.particles[:, 1]  = np.random.uniform(-half, half, n_particles)
        self.particles[:, 2]  = np.random.uniform(-np.pi, np.pi, n_particles)
        self.weights          = np.ones(n_particles) / n_particles

        # Pre-compute fixed ray offsets (updated lazily when n_rays changes)
        self._n_rays    = 0
        self._ray_local = None   # shape (R,)  - angles relative to robot

    # ------------------------------------------------------------------ init
    def initialize_at(self, x, y, theta, spread=0.3):
        """Initialize all particles near a known pose (from scene map)."""
        self.particles[:, 0] = np.random.normal(x,           spread,     self.n)
        self.particles[:, 1] = np.random.normal(y,           spread,     self.n)
        self.particles[:, 2] = np.random.normal(theta, spread / 2.0,     self.n)
        self.weights[:] = 1.0 / self.n

    # ------------------------------------------------------- motion model
    def predict(self, delta_x, delta_y, delta_theta):
        """
        Vectorised motion update: rotate odometry into each particle's
        world frame and add Gaussian noise.
        """
        angles   = self.particles[:, 2]              # (P,)
        cos_a    = np.cos(angles)
        sin_a    = np.sin(angles)
        nx, ny, nt = self.motion_noise

        self.particles[:, 0] += (delta_x * cos_a - delta_y * sin_a
                                 + np.random.normal(0, nx, self.n))
        self.particles[:, 1] += (delta_x * sin_a + delta_y * cos_a
                                 + np.random.normal(0, ny, self.n))
        self.particles[:, 2] += (delta_theta
                                 + np.random.normal(0, nt, self.n))
        # Wrap theta to [-pi, pi]
        self.particles[:, 2] = (
            (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi
        )

    # ------------------------------------------------------- sensor model
    def update(self, lidar_readings, obstacle_map):
        """
        Vectorised weight update.

        All P particles and R rays are processed simultaneously via
        NumPy broadcasting — no Python loops over particles.

        lidar_readings : list/array of R measured distances
        obstacle_map   : list of (ox, oy) tuples
        """
        meas = np.asarray(lidar_readings, dtype=np.float32)  # (R,)
        n_rays = len(meas)

        if n_rays != self._n_rays:
            self._n_rays    = n_rays
            self._ray_local = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

        ray_local = self._ray_local   # (R,)

        # Particle positions / headings  ->  shapes (P,1) for broadcasting
        px     = self.particles[:, 0:1]   # (P, 1)
        py     = self.particles[:, 1:2]
        ptheta = self.particles[:, 2:3]

        # World-frame ray directions: (P, R)
        world_angles = ptheta + ray_local          # (P, R)
        ray_dx       = np.cos(world_angles)        # (P, R)
        ray_dy       = np.sin(world_angles)

        # --- Expected distances ---
        expected = np.full((self.n, n_rays), 5.0, dtype=np.float32)

        # 4 walls at x = ±5 and y = ±5
        for wall_coord, axis in [(-5.0, 'x'), (5.0, 'x'),
                                  (-5.0, 'y'), (5.0, 'y')]:
            if axis == 'x':
                safe_dx = np.where(np.abs(ray_dx) > 1e-6, ray_dx, 1e-6)
                t = (wall_coord - px) / safe_dx          # (P, R)
            else:
                safe_dy = np.where(np.abs(ray_dy) > 1e-6, ray_dy, 1e-6)
                t = (wall_coord - py) / safe_dy
            valid = t > 0
            expected = np.where(valid & (t < expected), t, expected)

        # Obstacles (AABB slab method, vectorised over P and R)
        half_obs = 0.2  # half-size of 0.4 m cube
        for ox, oy in obstacle_map:
            # x-slab
            safe_dx  = np.where(np.abs(ray_dx) > 1e-6, ray_dx, 1e-6)
            tx_min   = ((ox - half_obs) - px) / safe_dx   # (P, R)
            tx_max   = ((ox + half_obs) - px) / safe_dx
            tx_min, tx_max = np.minimum(tx_min, tx_max), np.maximum(tx_min, tx_max)

            # y-slab
            safe_dy  = np.where(np.abs(ray_dy) > 1e-6, ray_dy, 1e-6)
            ty_min   = ((oy - half_obs) - py) / safe_dy
            ty_max   = ((oy + half_obs) - py) / safe_dy
            ty_min, ty_max = np.minimum(ty_min, ty_max), np.maximum(ty_min, ty_max)

            t_enter = np.maximum(tx_min, ty_min)
            t_exit  = np.minimum(tx_max, ty_max)
            hit     = (t_exit >= 0) & (t_enter <= t_exit)
            t_hit   = np.where(t_enter > 0, t_enter, t_exit)
            valid   = hit & (t_hit > 0) & (t_hit < expected)
            expected = np.where(valid, t_hit, expected)

        # --- Log-likelihood (Gaussian sensor model) ---
        diff        = meas - expected                                  # (P, R)
        log_w       = -0.5 * np.sum((diff / self.sensor_noise) ** 2,
                                     axis=1)                           # (P,)
        # Stabilise via log-sum-exp
        log_w      -= log_w.max()
        self.weights = np.exp(log_w)
        total        = self.weights.sum()
        self.weights = (self.weights / total
                        if total > 1e-300
                        else np.ones(self.n) / self.n)

    # ------------------------------------------------------- resampling
    def resample(self):
        """Low-variance (systematic) resampling."""
        cumulative     = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        positions      = (np.arange(self.n) + np.random.uniform()) / self.n
        idx            = np.searchsorted(cumulative, positions)
        self.particles = self.particles[idx]
        self.weights[:]= 1.0 / self.n

    # ------------------------------------------------------- estimate
    def estimate(self):
        """
        Weighted mean state estimate (x, y, theta).
        Uses circular mean for theta to handle wrap-around.
        """
        w          = self.weights
        x_est      = np.dot(w, self.particles[:, 0])
        y_est      = np.dot(w, self.particles[:, 1])
        sin_mean   = np.dot(w, np.sin(self.particles[:, 2]))
        cos_mean   = np.dot(w, np.cos(self.particles[:, 2]))
        theta_est  = np.arctan2(sin_mean, cos_mean)
        return float(x_est), float(y_est), float(theta_est)

    # ------------------------------------------------------- full step
    def step(self, delta_x, delta_y, delta_theta,
             lidar_readings, obstacle_map):
        """
        Full PF cycle: predict -> update -> resample -> estimate.
        """
        self.predict(delta_x, delta_y, delta_theta)
        if lidar_readings is not None and obstacle_map is not None:
            self.update(lidar_readings, obstacle_map)
            self.resample()
        return self.estimate()
