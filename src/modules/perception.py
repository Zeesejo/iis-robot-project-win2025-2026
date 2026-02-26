import numpy as np
import cv2

SCENE_OBJECTS = {
    'table':      {'rgb': [0.5, 0.3, 0.1], 'type': 'plane',   'dims': [1.5, 0.8, 0.625]},
    'target':     {'rgb': [1.0, 0.0, 0.0], 'type': 'cylinder', 'dims': [0.04, 0.04, 0.12]},
    'blue_obs':   {'rgb': [0.0, 0.0, 1.0], 'type': 'cube',    'dims': [0.4, 0.4, 0.4]},
    'pink_obs':   {'rgb': [1.0, 0.75, 0.8],'type': 'cube',    'dims': [0.4, 0.4, 0.4]},
    'orange_obs': {'rgb': [1.0, 0.64, 0.0],'type': 'cube',    'dims': [0.4, 0.4, 0.4]},
    'yellow_obs': {'rgb': [1.0, 1.0, 0.0], 'type': 'cube',    'dims': [0.4, 0.4, 0.4]},
    'black_obs':  {'rgb': [0.0, 0.0, 0.0], 'type': 'cube',    'dims': [0.4, 0.4, 0.4]},
}

# [F13] Red HSV tightened to exclude orange (H=0-8 + H=165-180).
# [F15-A] Lower saturation/value thresholds relaxed to S>=100, V>=60 so the
#         small cylinder (~8-12 px wide at 2m) is still detected at long range
#         where PyBullet lighting slightly desaturates colours.
#         Orange safety: orange lower bound is S>=150 and H>=9, so there is
#         zero overlap with red (H=0-8 or H=165-180, S>=100).
COLOR_RANGES_HSV = {
    'red':    [(np.array([0,   100,  60]),  np.array([8,   255, 255])),
               (np.array([165, 100,  60]),  np.array([180, 255, 255]))],
    'brown':  [(np.array([10,  80,  30]),   np.array([25,  180, 180]))],
    'blue':   [(np.array([100, 120,  70]),  np.array([130, 255, 255]))],
    'pink':   [(np.array([140,  40, 100]),  np.array([160, 255, 255]))],
    'orange': [(np.array([9,   150, 150]),  np.array([25,  255, 255]))],
    'yellow': [(np.array([25,  120, 120]),  np.array([35,  255, 255]))],
    'black':  [(np.array([0,     0,   0]),  np.array([180,  80,  50]))],
}


class SiftFeatureExtractor:

    def __init__(self):
        self.sift = cv2.SIFT_create()

    def compute_sift(self, image):
        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        kp, des = self.sift.detectAndCompute(gray_image, None)
        return kp, des

    def match_and_classify(self, test_des, knowledge_base, ratio_threshold=0.75):
        if test_des is None:
            return "Unknown", 0
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        best_match_count = -1
        best_category = "Unknown"

        for category, descriptor_list in knowledge_base.items():
            total_matches = 0
            for train_des in descriptor_list:
                if train_des is None:
                    continue
                matches = bf.knnMatch(test_des, train_des, k=2)
                good = [m for m, n in matches if m.distance < ratio_threshold * n.distance]
                total_matches += len(good)
            avg = total_matches / len(descriptor_list) if descriptor_list else 0
            if avg > best_match_count:
                best_match_count = avg
                best_category = category

        return best_category, best_match_count


def edge_contour_segmentation(rgb_image, min_contour_area=500, ratio_threshold=5.0):
    if len(rgb_image.shape) > 2:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = rgb_image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edge_map = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edge_map.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, dtype=np.uint8)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio > ratio_threshold:
            continue
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    return mask, edge_map


def detect_objects_by_color(bgr_image, min_area=200):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    detections = []

    for color_name, ranges in COLOR_RANGES_HSV.items():
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges:
            combined_mask |= cv2.inRange(hsv, lower, upper)
        # Use smaller kernel for red to preserve small objects
        kernel_size = (3, 3) if color_name == 'red' else (5, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append({
                'color': color_name, 'bbox': (x, y, w, h),
                'area': area, 'contour': cnt, 'mask': combined_mask,
            })
    return detections


class RANSAC_Segmentation:
    def __init__(self, points, max_iterations=1000, distance_threshold=0.01, min_inliers_ratio=0.3):
        self.points = points
        self.distance_threshold = distance_threshold
        self.max_iterations = max_iterations
        self.N = len(points)
        self.min_inliers = int(min_inliers_ratio * self.N)

    def _fit_plane(self, samples):
        """Calculates plane coefficients (A, B, C, D) from 3 points."""
        p1, p2, p3 = samples[0], samples[1], samples[2]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0:
            return None
        A, B, C = normal / norm
        D = -(A * p1[0] + B * p1[1] + C * p1[2])
        return A, B, C, D

    def _get_inliers(self, points, plane_model):
        """Determines inliers by perpendicular distance to the plane."""
        A, B, C, D = plane_model
        distances = np.abs(A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D)
        inliers_mask = distances < self.distance_threshold
        return np.sum(inliers_mask), inliers_mask

    def run(self):
        """Executes RANSAC loop, returns (inlier_mask, plane_model)."""
        best_count = 0
        best_mask = np.zeros(self.N, dtype=bool)
        best_model = None
        if self.N < 3:
            return best_mask, None
        for _ in range(self.max_iterations):
            idx = np.random.choice(self.N, 3, replace=False)
            model = self._fit_plane(self.points[idx])
            if model is None:
                continue
            count, mask = self._get_inliers(self.points, model)
            if count > best_count:
                best_count = count
                best_model = model
                best_mask = mask
                if best_count >= self.min_inliers:
                    break
        return best_mask, best_model


def compute_pca(points):
    if len(points) < 3:
        raise ValueError("Not enough points for PCA.")
    center = np.mean(points, axis=0)
    centered = points - center
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    projected = np.dot(centered, eigenvectors)
    mins = np.min(projected, axis=0)
    maxs = np.max(projected, axis=0)
    dimensions = maxs - mins
    obb_offset = (mins + maxs) / 2
    obb_center = center + np.dot(obb_offset, eigenvectors.T)
    return obb_center, eigenvectors, dimensions


def refine_object_points(points, obb_center, obb_vectors, obb_dims, tolerance=1.1):
    """Geometric refinement - removes outlier points."""
    if len(points) == 0:
        return points
    v1 = obb_vectors[:, 0]
    est_radius = (obb_dims[1] + obb_dims[2]) / 4.0
    centered = points - obb_center
    proj_v1 = np.dot(centered, v1)
    parallel = proj_v1[:, np.newaxis] * v1
    radial_dist = np.linalg.norm(centered - parallel, axis=1)
    half_len = obb_dims[0] / 2
    radial_ok = radial_dist < (est_radius * tolerance)
    axial_ok = (proj_v1 >= -half_len * tolerance) & (proj_v1 <= half_len * tolerance)
    return points[radial_ok & axial_ok]


def depth_to_point_cloud(depth_buffer, width=320, height=240, fov=60, near=0.1, far=10.0):
    """Convert PyBullet depth buffer to 3D point cloud."""
    depth_linear = far * near / (far - (far - near) * depth_buffer)
    depth_2d = np.reshape(depth_linear, (height, width))

    aspect = width / height
    fy = height / (2.0 * np.tan(np.radians(fov / 2.0)))
    fx = fy * aspect
    cx, cy = width / 2.0, height / 2.0

    v, u = np.indices((height, width))
    z = depth_2d
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = (z.flatten() > near) & (z.flatten() < far * 0.99)
    return points[valid], valid


def _extract_masked_points(depth_arr, color_mask_2d, width, height, fov=60, near=0.1, far=10.0):
    """
    [F13] Extract 3D points only from pixels where color_mask_2d is non-zero.
    Ensures PCA runs on the actual detected colour region, not the whole scene.
    """
    fy = height / (2.0 * np.tan(np.radians(fov / 2.0)))
    fx = fy * (width / height)
    cx, cy = width / 2.0, height / 2.0

    ys, xs = np.where(color_mask_2d > 0)
    if len(xs) == 0:
        return np.empty((0, 3))

    raw_d = depth_arr[ys, xs].astype(np.float32)
    valid = (raw_d > 0) & (raw_d < 1.0)
    xs, ys, raw_d = xs[valid], ys[valid], raw_d[valid]
    if len(xs) == 0:
        return np.empty((0, 3))

    z = far * near / (far - (far - near) * raw_d)
    depth_ok = (z > near) & (z < far * 0.99)
    xs, ys, z = xs[depth_ok], ys[depth_ok], z[depth_ok]
    if len(xs) == 0:
        return np.empty((0, 3))

    x3 = (xs - cx) * z / fx
    y3 = (ys - cy) * z / fy
    return np.stack([x3, y3, z], axis=1)


class PerceptionModule:
    def __init__(self):
        self.sift_extractor = SiftFeatureExtractor()
        self.knowledge_base = {}

    def build_knowledge_base(self, category_images):
        for cat, img_list in category_images.items():
            self.knowledge_base[cat] = []
            for img in img_list:
                _, des = self.sift_extractor.compute_sift(img)
                if des is not None:
                    self.knowledge_base[cat].append(des)

    def classify_roi(self, roi_image):
        """Classify a region-of-interest using the SIFT knowledge base."""
        _, des = self.sift_extractor.compute_sift(roi_image)
        return self.sift_extractor.match_and_classify(des, self.knowledge_base)

    def process_frame(self, rgb, depth, width=320, height=240):
        rgb_array = np.reshape(rgb, (height, width, 4)).astype(np.uint8)
        bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)

        detections = detect_objects_by_color(bgr)
        seg_mask, _ = edge_contour_segmentation(bgr, min_contour_area=300)

        depth_arr = np.array(depth).reshape(height, width)
        points_3d, valid_mask = depth_to_point_cloud(depth_arr, width, height)

        results = {'detections': detections, 'table_plane': None,
                   'target_pose': None, 'obstacle_poses': []}

        if len(points_3d) < 10:
            return results

        ransac = RANSAC_Segmentation(
            points=points_3d, max_iterations=500,
            distance_threshold=0.02, min_inliers_ratio=0.2
        )
        plane_mask, plane_model = ransac.run()

        if plane_model is not None:
            results['table_plane'] = {
                'model': plane_model,
                'num_inliers': int(np.sum(plane_mask))
            }

        object_points = points_3d[~plane_mask]
        if len(object_points) < 10:
            return results

        # [F13] PCA on RED-MASKED pixels only
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        red_mask_2d = np.zeros((height, width), dtype=np.uint8)
        for (lo, hi) in COLOR_RANGES_HSV['red']:
            red_mask_2d |= cv2.inRange(hsv, lo, hi)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        red_mask_2d = cv2.morphologyEx(red_mask_2d, cv2.MORPH_CLOSE, k3)
        red_mask_2d = cv2.morphologyEx(red_mask_2d, cv2.MORPH_OPEN,  k3)

        red_pts = _extract_masked_points(depth_arr, red_mask_2d, width, height)

        if len(red_pts) >= 10:
            try:
                center, vectors, dims = compute_pca(red_pts)
                refined = refine_object_points(red_pts, center, vectors, dims)
                if len(refined) >= 10:
                    center, vectors, dims = compute_pca(refined)
                results['target_pose'] = {
                    'center': center.tolist(),
                    'axes':   vectors.tolist(),
                    'dimensions': dims.tolist()
                }
            except ValueError:
                pass
        else:
            try:
                center, vectors, dims = compute_pca(object_points)
                refined = refine_object_points(object_points, center, vectors, dims)
                if len(refined) > 10:
                    center, vectors, dims = compute_pca(refined)
                results['target_pose'] = {
                    'center': center.tolist(),
                    'axes':   vectors.tolist(),
                    'dimensions': dims.tolist()
                }
            except ValueError:
                pass

        for det in detections:
            if det['color'] in ['blue', 'pink', 'orange', 'yellow', 'black']:
                x, y, w, h = det['bbox']
                roi_depth = depth_arr[y:y+h, x:x+w]
                if roi_depth.size == 0:
                    continue
                roi_pts, _ = depth_to_point_cloud(roi_depth.flatten(), w, h)
                if len(roi_pts) < 10:
                    continue
                try:
                    oc, ov, od = compute_pca(roi_pts)
                    results['obstacle_poses'].append({
                        'color': det['color'],
                        'center': oc.tolist(),
                        'axes':   ov.tolist(),
                        'dimensions': od.tolist()
                    })
                except ValueError:
                    continue

        return results


if __name__ == '__main__':

    np.random.seed(42)
    num_plane, num_obj = 5000, 1500

    px = np.random.uniform(-0.75, 0.75, num_plane)
    py = np.random.uniform(-0.4, 0.4, num_plane)
    pz = 0.625 + np.random.normal(0, 0.005, num_plane)
    plane_pts = np.stack([px, py, pz], axis=1)

    theta = np.random.uniform(0, 2 * np.pi, num_obj)
    r = 0.04 * np.sqrt(np.random.rand(num_obj))
    cx_ = r * np.cos(theta) + 0.2
    cy_ = r * np.sin(theta) - 0.1
    cz = np.random.uniform(0.625, 0.625 + 0.12, num_obj)
    cyl_pts = np.stack([cx_, cy_, cz], axis=1)

    full_cloud = np.concatenate([plane_pts, cyl_pts], axis=0)
    np.random.shuffle(full_cloud)
    print(f"\nSynthetic scene: {len(full_cloud)} pts ({num_plane} plane, {num_obj} cylinder)")

    ransac = RANSAC_Segmentation(full_cloud, max_iterations=500,
                                  distance_threshold=0.02, min_inliers_ratio=0.3)
    plane_mask, model = ransac.run()
    obj_pts = full_cloud[~plane_mask]
    print(f"RANSAC -> Plane: {np.sum(plane_mask)} pts, Objects: {len(obj_pts)} pts")
    if model:
        print(f"  Plane (A,B,C,D): ({model[0]:.4f}, {model[1]:.4f}, {model[2]:.4f}, {model[3]:.4f})")

    c1, v1, d1 = compute_pca(obj_pts)
    print(f"Preliminary PCA -> center={c1.round(4)}, dims={d1.round(4)}")
    refined = refine_object_points(obj_pts, c1, v1, d1, tolerance=1.1)
    c2, v2, d2 = compute_pca(refined)
    print(f"Final PCA -> center={c2.round(4)}, dims={d2.round(4)}")

    test_img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (50, 50), (150, 150), (0, 0, 255), -1)
    cv2.rectangle(test_img, (200, 100), (280, 180), (0, 165, 255), -1)
    dets = detect_objects_by_color(test_img, min_area=100)
    print(f"\nColor detection: {len(dets)} objects found")
    for d in dets:
        print(f"  Color={d['color']}, bbox={d['bbox']}, area={d['area']}")
    sift = SiftFeatureExtractor()
    kp, des = sift.compute_sift(test_img)
    print(f"\nSIFT: {len(kp) if kp else 0} keypoints")
    mask, edges = edge_contour_segmentation(test_img, min_contour_area=100)
    print(f"Edge segmentation: {np.sum(mask > 0)} mask pixels")
