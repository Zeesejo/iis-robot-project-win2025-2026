import numpy as np
import cv2
SCENE_OBJECTS = {
    'table':      {'rgb': [0.5, 0.3, 0.1], 'type': 'plane',    'dims': [1.5, 0.8, 0.625]},
    'target':     {'rgb': [1.0, 0.0, 0.0], 'type': 'cylinder', 'dims': [0.04, 0.04, 0.12]},
    'blue_obs':   {'rgb': [0.0, 0.0, 1.0], 'type': 'cube',     'dims': [0.4, 0.4, 0.4]},
    'pink_obs':   {'rgb': [1.0, 0.75, 0.8],'type': 'cube',     'dims': [0.4, 0.4, 0.4]},
    'orange_obs': {'rgb': [1.0, 0.64, 0.0],'type': 'cube',     'dims': [0.4, 0.4, 0.4]},
    'yellow_obs': {'rgb': [1.0, 1.0, 0.0], 'type': 'cube',     'dims': [0.4, 0.4, 0.4]},
    'black_obs':  {'rgb': [0.0, 0.0, 0.0], 'type': 'cube',     'dims': [0.4, 0.4, 0.4]},
}
COLOR_RANGES_HSV = {
    'red':    [(np.array([0,   150, 100]), np.array([10,  255, 255])),
               (np.array([170, 150, 100]), np.array([180, 255, 255]))],
    'brown':  [(np.array([10,  80,  30]),  np.array([25,  180, 180]))],
    'blue':   [(np.array([100, 150, 80]),  np.array([130, 255, 255]))],
    'pink':   [(np.array([140, 60,  120]), np.array([170, 255, 255]))],
    'orange': [(np.array([8,   220, 180]), np.array([14,  255, 255]))],
    'yellow': [(np.array([25,  150, 120]), np.array([35,  255, 255]))],
    'black':  [(np.array([0,   0,   0]),   np.array([180, 60,  40]))],
}
OBSTACLE_COLORS = {'blue', 'pink', 'orange', 'yellow', 'black'}
CAM_WIDTH   = 320
CAM_HEIGHT  = 240
CAM_FOV_DEG = 60 
CAM_NEAR    = 0.1
CAM_FAR     = 10.0
class SiftFeatureExtractor:
    def __init__(self):
        self.sift = cv2.SIFT_create()
    def compute_sift(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        kp, des = self.sift.detectAndCompute(gray, None)
        return kp, des
    def match_and_classify(self, test_des, knowledge_base, ratio_threshold=0.75):
        if test_des is None:
            return "Unknown", 0
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        best_count, best_cat = -1, "Unknown"
        for category, desc_list in knowledge_base.items():
            total = 0
            for train_des in desc_list:
                if train_des is None:
                    continue
                matches = bf.knnMatch(test_des, train_des, k=2)
                good = [m for m, n in matches if m.distance < ratio_threshold * n.distance]
                total += len(good)
            avg = total / len(desc_list) if desc_list else 0
            if avg > best_count:
                best_count = avg
                best_cat = category
        return best_cat, best_count
def edge_contour_segmentation(rgb_image, min_contour_area=500, ratio_threshold=5.0):
    gray    = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY) if len(rgb_image.shape) > 2 else rgb_image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, dtype=np.uint8)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ar = float(w) / h if h > 0 else 0
        if ar > ratio_threshold:
            continue
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    return mask, edges
def detect_objects_by_color(bgr_image, min_area=200):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    detections = []
    for color_name, ranges in COLOR_RANGES_HSV.items():
        color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in ranges:
            color_mask |= cv2.inRange(hsv, lo, hi)
        ksz = (3, 3) if color_name == 'red' else (5, 5)
        k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksz)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, k)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN,  k)
        effective_min = min_area * 4 if color_name == 'black' else min_area
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < effective_min:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append({
                'color':   color_name,
                'bbox':    (x, y, w, h),
                'area':    area,
                'contour': cnt,
                'mask':    color_mask,
            })
    return detections

class RANSAC_Segmentation:
    def __init__(self, points, max_iterations=1000,
                 distance_threshold=0.01, min_inliers_ratio=0.3):
        self.points             = points
        self.distance_threshold = distance_threshold
        self.max_iterations     = max_iterations
        self.N                  = len(points)
        self.min_inliers        = int(min_inliers_ratio * self.N)
        self._early_exit_count  = int(0.95 * self.N)
    def _fit_plane(self, samples):
        p1, p2, p3 = samples
        v1, v2     = p2 - p1, p3 - p1
        normal     = np.cross(v1, v2)
        norm       = np.linalg.norm(normal)
        if norm == 0:
            return None
        A, B, C = normal / norm
        D = -(A * p1[0] + B * p1[1] + C * p1[2])
        return A, B, C, D
    def _get_inliers(self, points, model):
        A, B, C, D = model
        dist = np.abs(A * points[:, 0] + B * points[:, 1]
                      + C * points[:, 2] + D)
        mask = dist < self.distance_threshold
        return int(np.sum(mask)), mask
    def run(self):
        best_count = 0
        best_mask  = np.zeros(self.N, dtype=bool)
        best_model = None
        if self.N < 3:
            return best_mask, None
        for _ in range(self.max_iterations):
            idx   = np.random.choice(self.N, 3, replace=False)
            model = self._fit_plane(self.points[idx])
            if model is None:
                continue
            count, mask = self._get_inliers(self.points, model)
            if count > best_count:
                best_count = count
                best_model = model
                best_mask  = mask
                if best_count >= self._early_exit_count:
                    break
        return best_mask, best_model
    
def compute_pca(points):
    if len(points) < 3:
        raise ValueError("Not enough points for PCA.")
    center   = np.mean(points, axis=0)
    centered = points - center
    cov      = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order        = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    projected    = np.dot(centered, eigenvectors)
    mins, maxs   = np.min(projected, axis=0), np.max(projected, axis=0)
    dimensions   = maxs - mins
    obb_offset   = (mins + maxs) / 2.0
    obb_center   = center + np.dot(obb_offset, eigenvectors.T)
    return obb_center, eigenvectors, dimensions
def refine_cylinder_points(points, obb_center, obb_vectors, obb_dims,
                            tolerance=1.2):
    if len(points) == 0:
        return points
    v1          = obb_vectors[:, 0]
    est_radius  = (obb_dims[1] + obb_dims[2]) / 4.0
    centered    = points - obb_center
    proj_v1     = np.dot(centered, v1)
    parallel    = proj_v1[:, np.newaxis] * v1
    radial_dist = np.linalg.norm(centered - parallel, axis=1)
    half_len    = obb_dims[0] / 2.0
    radial_ok   = radial_dist < (est_radius * tolerance)
    axial_ok    = ((proj_v1 >= -half_len * tolerance) &
                   (proj_v1 <=  half_len * tolerance))
    return points[radial_ok & axial_ok]
def refine_box_points(points, obb_center, obb_vectors, obb_dims,
                      tolerance=1.2):
    if len(points) == 0:
        return points
    centered = points - obb_center
    mask     = np.ones(len(points), dtype=bool)
    for axis_idx in range(3):
        v        = obb_vectors[:, axis_idx]
        proj     = np.dot(centered, v)
        half_dim = obb_dims[axis_idx] / 2.0 * tolerance
        mask    &= (proj >= -half_dim) & (proj <= half_dim)
    return points[mask]

def depth_to_point_cloud(depth_buffer, width=CAM_WIDTH, height=CAM_HEIGHT, fov=CAM_FOV_DEG, near=CAM_NEAR, far=CAM_FAR):
    depth_buffer = np.asarray(depth_buffer, dtype=np.float32)

    # OpenGL depth [0..1] -> linear depth (positive distance)
    z = (2.0 * near * far) / (far + near - (2.0 * depth_buffer - 1.0) * (far - near))
    z2d = z.reshape((height, width))

    # Your projection uses aspect = 1.0, so fx = fy
    fy = (height / 2.0) / np.tan(np.radians(fov / 2.0))
    fx = fy
    cx, cy = width / 2.0, height / 2.0

    v, u = np.indices((height, width))

    # OpenGL camera coordinates:
    #  x right, y up, camera looks along -Z
    x = (u - cx) * z2d / fx
    y = -(v - cy) * z2d / fy
    z_cam = -z2d

    points = np.stack([x, y, z_cam], axis=-1).reshape(-1, 3)

    # Validity should use the positive distance z2d
    valid = (z2d.flatten() > near) & (z2d.flatten() < far * 0.99)
    return points[valid], valid

def _view_to_Tcw(view_matrix):
    # viewMatrix is OpenGL column-major; maps WORLD -> CAMERA
    return np.array(view_matrix, dtype=np.float32).reshape((4, 4), order='F')

def cam_to_world(p_cam, view_matrix):
    T_cw = _view_to_Tcw(view_matrix)   # world -> camera
    T_wc = np.linalg.inv(T_cw)         # camera -> world
    p4 = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float32)
    pw = T_wc @ p4
    return pw[:3]

# PerceptionModule
class PerceptionModule:
    def __init__(self):
        self.sift_extractor    = SiftFeatureExtractor()
        self.knowledge_base    = {}
        self._scene_map_locked = False
        self.scene_map = {
            'table':     None,
            'obstacles': {},     # color → pose dict; locked after 3 entries
        }
    def build_knowledge_base(self, category_images):
        for cat, img_list in category_images.items():
            self.knowledge_base[cat] = []
            for img in img_list:
                _, des = self.sift_extractor.compute_sift(img)
                if des is not None:
                    self.knowledge_base[cat].append(des)

    def classify_roi(self, roi_image):
        _, des = self.sift_extractor.compute_sift(roi_image)
        return self.sift_extractor.match_and_classify(des, self.knowledge_base)
    def _extract_mask_points(self, colour_mask_2d, valid_flat,
                              all_points_3d, plane_mask):
        # colour_mask_2d is (H,W) uint8; flatten to (H*W,) bool
        colour_flat  = (colour_mask_2d.flatten() > 0)   # (H*W,)
        # Which rows of all_points_3d come from colour pixels?
        in_cloud     = colour_flat[valid_flat]           # (N,)  N = len(all_points_3d)
        not_plane    = ~plane_mask                       # (N,)
        return all_points_3d[in_cloud & not_plane]

    def process_frame(self, rgb, depth, width=CAM_WIDTH, height=CAM_HEIGHT, view_matrix=None):
        results = {
            'detections':     [],
            'table_plane':    None,
            'target_pose':    None,
            'obstacle_poses': [],
            'scene_map':      self.scene_map,
        }
        try:
            rgb_array = np.reshape(rgb,   (height, width, 4)).astype(np.uint8)
            depth_arr = np.array(depth, dtype=np.float32).reshape(height, width)
        except Exception:
            return results

        bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
        detections = detect_objects_by_color(bgr, min_area=50)
        results['detections'] = detections
        seg_mask, _ = edge_contour_segmentation(bgr, min_contour_area=300)
        all_points_3d, valid_flat = depth_to_point_cloud(
            depth_arr.flatten(), width, height)

        if len(all_points_3d) < 10:
            return results
        ransac = RANSAC_Segmentation(
            points=all_points_3d,
            max_iterations=500,
            distance_threshold=0.02,
            min_inliers_ratio=0.2,
        )
        plane_mask, plane_model = ransac.run()

        if plane_model is not None:
            inliers = all_points_3d[plane_mask]
            if len(inliers) > 50:
                z_med = float(np.median(inliers[:, 2]))
                # table top is z=0.625 (allow tolerance)
                if abs(z_med - 0.625) < 0.10:
                    results['table_plane'] = {
                        'model': plane_model,
                        'num_inliers': int(np.sum(plane_mask)),
                        'z_med': z_med
                    }
        else:
            plane_mask = np.zeros(len(all_points_3d), dtype=bool)
        red_target_pose = None
        red_detections  = [d for d in detections if d['color'] == 'red']
        red_detections.sort(key=lambda d: d['area'], reverse=True)
        for det in red_detections[:1]:
            red_pts = self._extract_mask_points(
                det['mask'], valid_flat, all_points_3d, plane_mask)
            if len(red_pts) < 5:
                continue
            median_z = np.median(red_pts[:, 2])
            red_pts = red_pts[np.abs(red_pts[:, 2] - median_z) < 0.08]

            if len(red_pts) < 5:
                continue
            
            center, vectors, dims = compute_pca(red_pts)
            refined = refine_cylinder_points(red_pts, center, vectors, dims)
            if len(refined) >= 5:
                center, vectors, dims = compute_pca(refined)
            if np.max(dims) > 0.5:
                continue

            center_out = center
            if view_matrix is not None:
                center_out = cam_to_world(center, view_matrix)
            depth_m = float(-center[2])

            red_target_pose = {
                'center':     center_out.tolist(),
                'axes':       vectors.tolist(),
                'dimensions': dims.tolist(),
                'depth_m':      depth_m,
                'center_cam': center.tolist()
            }

        results['target_pose'] = red_target_pose
        obstacle_poses = []
        for det in detections:
            if det['color'] not in OBSTACLE_COLORS:
                continue
            obs_pts = self._extract_mask_points(
                det['mask'], valid_flat, all_points_3d, plane_mask)
            if len(obs_pts) < 10:
                continue
            median_z = np.median(obs_pts[:, 2])
            obs_pts  = obs_pts[np.abs(obs_pts[:, 2] - median_z) < 0.35]
            if len(obs_pts) < 10:
                continue
            # try:
            oc, ov, od = compute_pca(obs_pts)
            refined_obs = refine_box_points(obs_pts, oc, ov, od)
            if len(refined_obs) >= 10:
                oc, ov, od = compute_pca(refined_obs)
            if np.max(od) > 0.8:
                continue

            oc_out = oc
            if view_matrix is not None:
                oc_out = cam_to_world(oc, view_matrix)

            pose = {
                'color':      det['color'],
                'center':     oc_out.tolist(),
                'axes':       ov.tolist(),
                'dimensions': od.tolist(),
            }
            obstacle_poses.append(pose)
            if not self._scene_map_locked:
                print(f"[M4-PCA] Obstacle ({det['color']}) "
                        f"center=({oc[0]:.3f},{oc[1]:.3f},{oc[2]:.3f}) "
                        f"dims=({od[0]:.3f},{od[1]:.3f},{od[2]:.3f})")
                if det['color'] not in self.scene_map['obstacles']:
                    self.scene_map['obstacles'][det['color']] = pose
        results['obstacle_poses'] = obstacle_poses
        if (not self._scene_map_locked and
                len(self.scene_map['obstacles']) >= 3):
            self._scene_map_locked = True
            print(f"[M4] Static scene map locked "
                  f"({len(self.scene_map['obstacles'])} obstacles): "
                  f"{list(self.scene_map['obstacles'].keys())}")
        return results