"""
M5 - State Estimation: Particle Filter
Fuses noisy LIDAR + IMU + control inputs to maintain a reliable
state estimate (x, y, theta) of the robot.
"""

import numpy as np


class ParticleFilter:
    """
    Monte Carlo Localization (Particle Filter) for robot state estimation.
    State: [x, y, theta]
    """

    def __init__(self, n_particles=300, room_size=10.0,
                 motion_noise=(0.05, 0.05, 0.03),
                 sensor_noise=0.3):
        """
        n_particles: number of particles
        room_size: half-width of room in meters
        motion_noise: (x, y, theta) std devs for motion model
        sensor_noise: std dev for sensor likelihood
        """
        self.n = n_particles
        self.room = room_size
        self.motion_noise = motion_noise
        self.sensor_noise = sensor_noise

        # Initialize particles uniformly in room
        half = room_size / 2.0
        self.particles = np.zeros((n_particles, 3))
        self.particles[:, 0] = np.random.uniform(-half, half, n_particles)
        self.particles[:, 1] = np.random.uniform(-half, half, n_particles)
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, n_particles)
        self.weights = np.ones(n_particles) / n_particles

    def initialize_at(self, x, y, theta, spread=0.3):
        """Initialize particles near a known pose (from scene map)."""
        self.particles[:, 0] = np.random.normal(x, spread, self.n)
        self.particles[:, 1] = np.random.normal(y, spread, self.n)
        self.particles[:, 2] = np.random.normal(theta, spread / 2, self.n)
        self.weights = np.ones(self.n) / self.n

    # -------- MOTION MODEL --------
    def predict(self, delta_x, delta_y, delta_theta):
        """
        Move all particles according to odometry + noise.
        delta_x, delta_y in robot frame; delta_theta: rotation.
        """
        nx, ny, nt = self.motion_noise
        angles = self.particles[:, 2]

        # Rotate odometry into world frame per particle
        dx_world = delta_x * np.cos(angles) - delta_y * np.sin(angles)
        dy_world = delta_x * np.sin(angles) + delta_y * np.cos(angles)

        self.particles[:, 0] += dx_world + np.random.normal(0, nx, self.n)
        self.particles[:, 1] += dy_world + np.random.normal(0, ny, self.n)
        self.particles[:, 2] += delta_theta + np.random.normal(0, nt, self.n)
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

    # -------- SENSOR MODEL --------
    def update(self, lidar_readings, obstacle_map):
        """
        Update particle weights based on LIDAR vs expected distances.
        obstacle_map: list of (ox, oy) obstacle positions from scene map.
        lidar_readings: list of measured distances (N rays).
        """
        n_rays = len(lidar_readings)
        ray_angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

        new_weights = np.ones(self.n)
        for i, particle in enumerate(self.particles):
            px, py, ptheta = particle
            expected = self._expected_lidar(px, py, ptheta, ray_angles, obstacle_map)
            diff = np.array(lidar_readings) - np.array(expected)
            log_likelihood = -0.5 * np.sum((diff / self.sensor_noise) ** 2)
            new_weights[i] = np.exp(log_likelihood)

        # Normalize
        total = new_weights.sum()
        if total > 1e-300:
            self.weights = new_weights / total
        else:
            self.weights = np.ones(self.n) / self.n

    def _expected_lidar(self, px, py, ptheta, ray_angles, obstacle_map, max_range=5.0):
        """Compute expected LIDAR distances at a given particle pose."""
        expected = []
        for angle in ray_angles:
            world_angle = ptheta + angle
            dx = np.cos(world_angle)
            dy = np.sin(world_angle)
            min_dist = max_range

            # Wall distances
            for wall_dist in [5.0]:
                # Check 4 walls
                for t in self._ray_wall_intersections(px, py, dx, dy):
                    if 0 < t < min_dist:
                        min_dist = t

            # Obstacle distances
            for ox, oy in obstacle_map:
                t = self._ray_box_intersection(px, py, dx, dy, ox, oy, 0.4)
                if t is not None and 0 < t < min_dist:
                    min_dist = t

            expected.append(min_dist)
        return expected

    def _ray_wall_intersections(self, px, py, dx, dy):
        """Return distances to all 4 room walls from ray origin."""
        ts = []
        for wall_x in [-5.0, 5.0]:
            if abs(dx) > 1e-6:
                t = (wall_x - px) / dx
                if t > 0:
                    ts.append(t)
        for wall_y in [-5.0, 5.0]:
            if abs(dy) > 1e-6:
                t = (wall_y - py) / dy
                if t > 0:
                    ts.append(t)
        return ts

    def _ray_box_intersection(self, px, py, dx, dy, cx, cy, size):
        """AABB ray-box intersection test for a box at (cx,cy) with given size."""
        half = size / 2.0
        tmin_x = ((cx - half) - px) / dx if abs(dx) > 1e-6 else -np.inf
        tmax_x = ((cx + half) - px) / dx if abs(dx) > 1e-6 else np.inf
        if tmin_x > tmax_x:
            tmin_x, tmax_x = tmax_x, tmin_x
        tmin_y = ((cy - half) - py) / dy if abs(dy) > 1e-6 else -np.inf
        tmax_y = ((cy + half) - py) / dy if abs(dy) > 1e-6 else np.inf
        if tmin_y > tmax_y:
            tmin_y, tmax_y = tmax_y, tmin_y
        tmin = max(tmin_x, tmin_y)
        tmax = min(tmax_x, tmax_y)
        if tmax < 0 or tmin > tmax:
            return None
        return tmin if tmin > 0 else tmax

    # -------- RESAMPLING --------
    def resample(self):
        """Low-variance resampling."""
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        idx = np.searchsorted(cumulative,
                              np.random.uniform(0, 1.0 / self.n) +
                              np.arange(self.n) / self.n)
        self.particles = self.particles[idx]
        self.weights = np.ones(self.n) / self.n

    # -------- ESTIMATE --------
    def estimate(self):
        """
        Return the weighted mean state estimate (x, y, theta).
        Uses circular mean for theta.
        """
        x_est = np.sum(self.weights * self.particles[:, 0])
        y_est = np.sum(self.weights * self.particles[:, 1])
        sin_mean = np.sum(self.weights * np.sin(self.particles[:, 2]))
        cos_mean = np.sum(self.weights * np.cos(self.particles[:, 2]))
        theta_est = np.arctan2(sin_mean, cos_mean)
        return float(x_est), float(y_est), float(theta_est)

    def step(self, delta_x, delta_y, delta_theta, lidar_readings, obstacle_map):
        """
        Full PF cycle: predict -> update -> resample -> estimate.
        """
        self.predict(delta_x, delta_y, delta_theta)
        if lidar_readings is not None and obstacle_map is not None:
            self.update(lidar_readings, obstacle_map)
            self.resample()
        return self.estimate()
