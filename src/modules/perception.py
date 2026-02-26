"""
M4 - Perception Module
Object detection by color/size from RGBD, RANSAC plane fitting,
and PCA on point clouds for pose estimation.
"""

import numpy as np


# ===================== DEPTH -> POINT CLOUD =====================

def depth_to_pointcloud(depth, rgb, proj_matrix, view_matrix,
                         img_w=320, img_h=240):
    """
    Convert a PyBullet depth buffer + RGB image to a 3D point cloud
    with colour annotations.

    proj_matrix may be passed as:
      - a flat 16-element tuple/list (raw PyBullet output) -> reshaped here
      - a (4,4) numpy array                                -> used directly

    Returns: points (N,3), colors (N,3)
    """
    depth = np.array(depth)
    rgb_arr = np.array(rgb).reshape(img_h, img_w, 4)[:, :, :3] / 255.0

    # Normalise proj_matrix to a (4,4) array regardless of input format
    pm = np.array(proj_matrix, dtype=np.float64)
    if pm.ndim == 1:
        # PyBullet returns a flat column-major 16-element tuple
        pm = pm.reshape(4, 4).T  # transpose: PyBullet is column-major

    # Standard OpenGL projection entries
    fx = pm[0, 0]
    fy = pm[1, 1]
    cx, cy = img_w / 2.0, img_h / 2.0

    near, far = 0.1, 10.0
    # Convert PyBullet's normalized depth to metric distance
    z = far * near / (far - (far - near) * depth)   # shape (img_h, img_w)

    u_idx, v_idx = np.meshgrid(np.arange(img_w), np.arange(img_h))
    x = (u_idx - cx) * z / (fx * img_w * 0.5)
    y = (v_idx - cy) * z / (fy * img_h * 0.5)

    # Flatten everything to 1-D before stacking / masking
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = z.ravel()                               # shape (N,)
    colors  = rgb_arr.reshape(-1, 3)                 # shape (N, 3)

    points = np.stack([x_flat, y_flat, z_flat], axis=-1)  # shape (N, 3)

    # Remove points at max range (background) — all arrays are now 1-D
    valid = z_flat < (far * 0.99)                    # shape (N,)
    return points[valid], colors[valid]


# ===================== COLOR-BASED DETECTION =====================

def detect_by_color(points, colors, target_color, tol=0.25):
    """
    Filter point cloud by RGB color proximity.
    target_color: [r, g, b] in [0,1]
    tol: float - colour distance threshold (higher = wider detection).
         Can be tuned at runtime by VisionThresholdTuner (M9).
    Returns: filtered points (N,3)
    """
    target = np.array(target_color)
    diff = np.linalg.norm(colors - target, axis=1)
    mask = diff < tol
    return points[mask]


def detect_table(points, colors, tol=0.25):
    """Detect the brown table surface.

    Args:
        tol: colour tolerance, wired in from M9 VisionThresholdTuner.
    """
    return detect_by_color(points, colors, [0.5, 0.3, 0.1], tol=tol)


def detect_target(points, colors, tol=0.3):
    """Detect the red target cylinder.

    Args:
        tol: colour tolerance, wired in from M9 VisionThresholdTuner.
    """
    return detect_by_color(points, colors, [1.0, 0.0, 0.0], tol=tol)


def detect_obstacles(points, colors, tol=0.3):
    """Detect obstacle colors (blue, pink, orange, yellow, black).

    Args:
        tol: colour tolerance applied to all obstacle classes.
    """
    obstacle_colors = [
        [0.0, 0.0, 1.0],
        [1.0, 0.4, 0.7],
        [1.0, 0.5, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    detected = []
    for col in obstacle_colors:
        pts = detect_by_color(points, colors, col, tol=tol)
        if len(pts) > 10:
            detected.append(pts)
    return detected


# ===================== RANSAC PLANE FITTING =====================

def ransac_plane(points, n_iter=200, dist_thresh=0.02):
    """
    RANSAC-based plane fitting on a 3D point cloud.
    Returns: (normal, d) of best plane, inlier mask.
    """
    if len(points) < 3:
        return None, None

    best_inliers = None
    best_count = 0
    best_normal = None
    best_d = None

    for _ in range(n_iter):
        idx = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        d = -np.dot(normal, p1)
        dists = np.abs(np.dot(points, normal) + d)
        inliers = dists < dist_thresh
        count = np.sum(inliers)
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_normal = normal
            best_d = d

    return (best_normal, best_d), best_inliers


def fit_table_plane(points, colors, tol=0.25):
    """
    Detect table surface plane using RANSAC.
    Returns plane parameters and table center.

    Args:
        tol: colour tolerance passed to detect_table.
    """
    table_pts = detect_table(points, colors, tol=tol)
    if len(table_pts) < 10:
        return None, None
    (normal, d), inliers = ransac_plane(table_pts)
    if inliers is None:
        return None, None
    table_inlier_pts = table_pts[inliers]
    center = table_inlier_pts.mean(axis=0)
    return (normal, d), center


# ===================== PCA FOR POSE ESTIMATION =====================

def pca_pose(points):
    """
    Run PCA on a point cloud to estimate object pose.
    Returns: center (3,), principal axes (3,3), extents (3,)
    """
    if len(points) < 3:
        return None, None, None
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by descending eigenvalue
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    extents = 2.0 * np.sqrt(np.maximum(eigenvalues, 0))
    return center, eigenvectors, extents


def estimate_grasp_pose(target_points):
    """
    Use PCA on the target cylinder point cloud to get optimal grasp center + orientation.
    Returns: grasp_position (3,), grasp_orientation_matrix (3,3)
    """
    center, axes, extents = pca_pose(target_points)
    if center is None:
        return None, None
    return center, axes


def estimate_obstacle_pose(obstacle_points):
    """
    Use PCA on obstacle point cloud to find avoidance pose.
    Returns: center, axes, extents
    """
    return pca_pose(obstacle_points)
