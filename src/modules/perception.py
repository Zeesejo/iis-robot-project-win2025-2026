
import numpy as np
import cv2
from sklearn.decomposition import PCA
class PerceptionSystem:
    def __init__(self):
        self.color_ranges = {
            'red': {
                'lower': np.array([150, 0, 0]),
                'upper': np.array([255, 100, 100])
            },
            'blue': {
                'lower': np.array([0, 0, 150]),
                'upper': np.array([100, 100, 255])
            },
            'brown': {
                'lower': np.array([80, 40, 10]),
                'upper': np.array([180, 120, 80])
            },
            'pink': {
                'lower': np.array([180, 100, 150]),
                'upper': np.array([255, 200, 255])
            },
            'orange': {
                'lower': np.array([200, 100, 0]),
                'upper': np.array([255, 180, 100])
            },
            'yellow': {
                'lower': np.array([200, 200, 0]),
                'upper': np.array([255, 255, 150])
            },
            'black': {
                'lower': np.array([0, 0, 0]),
                'upper': np.array([50, 50, 50])
            }
        }
        
        # Expected object sizes (approximate volumes in m^3)
        self.object_sizes = {
            'target': {'min': 0.0005, 'max': 0.002},      # Small cylinder
            'obstacle': {'min': 0.05, 'max': 0.07},       # 0.4m cube
            'table': {'min': 0.5, 'max': 1.5}             # Large table
        }
        
        # Camera intrinsics (based on sensor_wrapper defaults)
        self.camera_width = 320
        self.camera_height = 240
        self.fov = 60  # degrees
        self.near_plane = 0.1
        self.far_plane = 10.0
        
        # RANSAC parameters
        self.ransac_max_iterations = 1000
        self.ransac_distance_threshold = 0.02  # 2cm
        self.ransac_min_inliers = 500
        
    def preprocess_depth(self, depth_buffer):
        # Convert PyBullet depth buffer to actual depth
        # Formula from PyBullet documentation
        depth_meters = self.far_plane * self.near_plane / (
            self.far_plane - (self.far_plane - self.near_plane) * depth_buffer
        )
        # Apply median filter to reduce noise (Law of Large Numbers)
        depth_meters = cv2.medianBlur(depth_meters.astype(np.float32), 5)
        
        return depth_meters
    
    def rgb_to_point_cloud(self, rgb_image, depth_map):
        h, w = depth_map.shape
        # Compute focal length from FOV
        fx = w / (2 * np.tan(np.radians(self.fov / 2)))
        fy = fx  # Assume square pixels
        cx, cy = w / 2, h / 2
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        # Convert to 3D coordinates
        z = depth_map
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        # Stack into point cloud
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        # Extract colors (convert RGBA to RGB if needed)
        if rgb_image.shape[-1] == 4:
            colors = rgb_image[:, :, :3].reshape(-1, 3)
        else:
            colors = rgb_image.reshape(-1, 3)
        # Filter out invalid points (depth = 0 or too far)
        valid_mask = (z.flatten() > self.near_plane) & (z.flatten() < self.far_plane)
        return points[valid_mask], colors[valid_mask]
    
    def detect_objects_by_color(self, rgb_image, depth_map, target_color):
        if target_color not in self.color_ranges:
            return []
        # Extract RGB channels (handle RGBA if present)
        if rgb_image.shape[-1] == 4:
            rgb = rgb_image[:, :, :3]
        else:
            rgb = rgb_image
        
        # Create color mask
        lower = self.color_ranges[target_color]['lower']
        upper = self.color_ranges[target_color]['upper']
        mask = cv2.inRange(rgb, lower, upper)
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter small noise
            if area < 50:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract point cloud for this region
            region_mask = np.zeros_like(mask)
            cv2.drawContours(region_mask, [contour], -1, 255, -1)
            
            # Get 3D points
            points_3d = []
            for i in range(y, y + h):
                for j in range(x, x + w):
                    if region_mask[i, j] > 0 and depth_map[i, j] > 0:
                        # Convert pixel to 3D point
                        fx = self.camera_width / (2 * np.tan(np.radians(self.fov / 2)))
                        z = depth_map[i, j]
                        x_3d = (j - self.camera_width / 2) * z / fx
                        y_3d = (i - self.camera_height / 2) * z / fx
                        points_3d.append([x_3d, y_3d, z])
            
            if len(points_3d) < 10:
                continue
            
            points_3d = np.array(points_3d)
            
            # Compute centroid
            centroid = np.mean(points_3d, axis=0)
            
            # Estimate volume (bounding box volume)
            volume = np.prod(np.ptp(points_3d, axis=0))
            
            detections.append({
                'color': target_color,
                'centroid': centroid,
                'points': points_3d,
                'volume': volume,
                'pixel_area': area,
                'bbox_2d': (x, y, w, h)
            })
        
        return detections
    
    def classify_object_by_size(self, detection):
        volume = detection['volume']
        for obj_type, size_range in self.object_sizes.items():
            if size_range['min'] <= volume <= size_range['max']:
                return obj_type
        
        return 'unknown'
    
    def detect_table_ransac(self, rgb_image, depth_map):
        # Convert to point cloud
        points, colors = self.rgb_to_point_cloud(rgb_image, depth_map)
        
        if len(points) < 3:
            return None, None, None
        
        # Look for brown-colored regions (table)
        brown_mask = np.all(
            (colors >= self.color_ranges['brown']['lower']) &
            (colors <= self.color_ranges['brown']['upper']),
            axis=1
        )
        
        # If brown detection fails, use all points
        if np.sum(brown_mask) < 100:
            candidate_points = points
        else:
            candidate_points = points[brown_mask]
        
        # RANSAC algorithm
        best_model = None
        best_inliers_count = 0
        best_inliers_mask = np.zeros(len(candidate_points), dtype=bool)
        
        for iteration in range(self.ransac_max_iterations):
            # Randomly sample 3 points
            if len(candidate_points) < 3:
                break
                
            indices = np.random.choice(len(candidate_points), 3, replace=False)
            samples = candidate_points[indices]
            
            # Fit plane
            plane_model = self._fit_plane(samples)
            
            if plane_model is None:
                continue
            
            # Find inliers
            inliers_count, inliers_mask = self._get_plane_inliers(
                candidate_points, plane_model
            )
            
            # Update best model
            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count
                best_model = plane_model
                best_inliers_mask = inliers_mask
                
                # Early exit if we have enough inliers
                if best_inliers_count >= self.ransac_min_inliers:
                    break
        
        if best_model is None:
            return None, None, None
        
        inlier_points = candidate_points[best_inliers_mask]
        
        return best_model, inlier_points, best_inliers_mask
    
    def _fit_plane(self, samples):
        if samples.shape[0] != 3:
            return None
        
        p1, p2, p3 = samples[0], samples[1], samples[2]
        
        # Compute two vectors on the plane
        v1 = p2 - p1
        v2 = p3 - p1
        
        # Normal vector via cross product
        normal = np.cross(v1, v2)
        
        # Normalize
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return None
        
        A, B, C = normal / norm
        
        # Compute D using one of the points
        D = -(A * p1[0] + B * p1[1] + C * p1[2])
        
        return (A, B, C, D)
    
    def _get_plane_inliers(self, points, plane_model):
        A, B, C, D = plane_model
        # Compute perpendicular distance from each point to plane
        distances = np.abs(A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D)
        # Inliers are points within threshold distance
        inliers_mask = distances < self.ransac_distance_threshold
        inliers_count = np.sum(inliers_mask)
        return inliers_count, inliers_mask
    
    def estimate_pose_pca(self, point_cloud, object_type='obstacle'):
        if len(point_cloud) < 3:
            return None
        
        # Compute centroid
        centroid = np.mean(point_cloud, axis=0)
        
        # Center the points
        centered_points = point_cloud - centroid
        
        # Apply PCA
        pca = PCA(n_components=3)
        pca.fit(centered_points)
        
        # Principal axes (eigenvectors)
        principal_axes = pca.components_  # Shape: (3, 3)
        
        # Explained variance gives us the extent along each axis
        eigenvalues = pca.explained_variance_
        
        # Compute extents (approximate object dimensions)
        extents = 2 * np.sqrt(eigenvalues)  # 95% confidence interval
        
        # Build rotation matrix from principal axes
        # First principal component = longest axis
        # For grasping: align gripper with shortest axis
        # For avoidance: use longest axis for path planning
        
        if object_type == 'target':
            # For grasping: approach along shortest axis (most stable)
            sorted_indices = np.argsort(extents)
            approach_axis = principal_axes[sorted_indices[0]]  # Shortest
            grasp_axis = principal_axes[sorted_indices[2]]     # Longest
            
            # Build orientation matrix
            z_axis = approach_axis / np.linalg.norm(approach_axis)
            x_axis = grasp_axis / np.linalg.norm(grasp_axis)
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)  # Recompute for orthogonality
            
            orientation = np.column_stack([x_axis, y_axis, z_axis])
            
        else:  # obstacle or table
            # For avoidance: use principal axes directly
            orientation = principal_axes.T
        
        # Ensure proper rotation matrix (orthonormal)
        U, _, Vt = np.linalg.svd(orientation)
        orientation = U @ Vt
        
        return {
            'position': centroid,
            'orientation': orientation,
            'principal_axes': principal_axes,
            'extents': extents,
            'eigenvalues': eigenvalues
        }
    
    def detect_all_objects(self, rgb_image, depth_map):
        # Preprocess depth
        depth_meters = self.preprocess_depth(depth_map)
        scene_objects = {
            'target': None,
            'table': None,
            'obstacles': []
        }
        plane_model, table_points, _ = self.detect_table_ransac(rgb_image, depth_meters)
        
        if plane_model is not None and table_points is not None:
            table_pose = self.estimate_pose_pca(table_points, object_type='table')
            scene_objects['table'] = {
                'plane_model': plane_model,
                'points': table_points,
                'pose': table_pose
            }
        
        # Detect red target object
        red_detections = self.detect_objects_by_color(rgb_image, depth_meters, 'red')
        for detection in red_detections:
            obj_type = self.classify_object_by_size(detection)
            if obj_type == 'target':
                pose = self.estimate_pose_pca(detection['points'], object_type='target')
                scene_objects['target'] = {
                    'color': 'red',
                    'centroid': detection['centroid'],
                    'points': detection['points'],
                    'volume': detection['volume'],
                    'pose': pose
                }
                break
        
        # Detect obstacles (blue, pink, orange, yellow, black)
        obstacle_colors = ['blue', 'pink', 'orange', 'yellow', 'black']
        
        for color in obstacle_colors:
            detections = self.detect_objects_by_color(rgb_image, depth_meters, color)
            for detection in detections:
                obj_type = self.classify_object_by_size(detection)
                if obj_type == 'obstacle':
                    pose = self.estimate_pose_pca(detection['points'], object_type='obstacle')
                    scene_objects['obstacles'].append({
                        'color': color,
                        'centroid': detection['centroid'],
                        'points': detection['points'],
                        'volume': detection['volume'],
                        'pose': pose
                    })
        
        return scene_objects
    
    def visualize_detections(self, rgb_image, scene_objects):
        img = rgb_image.copy()
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        # Draw target
        if scene_objects['target'] is not None:
            centroid = scene_objects['target']['centroid']
            # Project 3D to 2D (simplified)
            fx = self.camera_width / (2 * np.tan(np.radians(self.fov / 2)))
            u = int(centroid[0] * fx / centroid[2] + self.camera_width / 2)
            v = int(centroid[1] * fx / centroid[2] + self.camera_height / 2)
            
            cv2.circle(img, (u, v), 10, (255, 0, 0), 2)
            cv2.putText(img, 'TARGET', (u + 15, v), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 0, 0), 2)
        # Draw obstacles
        for i, obs in enumerate(scene_objects['obstacles']):
            centroid = obs['centroid']
            fx = self.camera_width / (2 * np.tan(np.radians(self.fov / 2)))
            u = int(centroid[0] * fx / centroid[2] + self.camera_width / 2)
            v = int(centroid[1] * fx / centroid[2] + self.camera_height / 2)
            
            cv2.circle(img, (u, v), 8, (0, 0, 255), 2)
            cv2.putText(img, f'OBS{i}', (u + 15, v), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (0, 0, 255), 1)
        
        return img


# Convenience function for easy import
def create_perception_system():
    return PerceptionSystem()
