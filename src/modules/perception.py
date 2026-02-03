import numpy as np
import cv2
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict, Optional


class PerceptionModule:
    OBJECT_COLORS = {
        'table': np.array([0.5, 0.3, 0.1]),      # Brown
        'target': np.array([1.0, 0.0, 0.0]),     # Red
        'obstacle_blue': np.array([0.0, 0.0, 1.0]),   # Blue
        'obstacle_pink': np.array([1.0, 0.75, 0.8]),  # Pink
        'obstacle_orange': np.array([1.0, 0.5, 0.0]), # Orange
        'obstacle_yellow': np.array([1.0, 1.0, 0.0]), # Yellow
        'obstacle_black': np.array([0.0, 0.0, 0.0]),  # Black
        'floor': np.array([0.2, 0.2, 0.2]),      # Gray
    }
    OBJECT_SPECS = {
        'table': {'height': 0.625, 'width': 1.5, 'depth': 0.8},
        'target': {'radius': 0.04, 'height': 0.12},  # Cylinder
        'obstacle': {'size': 0.4}  # Cube side length
    }
    
    def __init__(self, color_tolerance: float = 0.2, ransac_threshold: float = 0.01):
        self.color_tolerance = color_tolerance
        self.ransac_threshold = ransac_threshold
        
    def detect_objects_by_color(self, rgb_image: np.ndarray, 
                                depth_image: np.ndarray,
                                segmentation_mask: np.ndarray = None) -> Dict[str, List[Dict]]:
        detected_objects = {
            'table': [],
            'target': [],
            'obstacles': []
        }
        
        if rgb_image.max() > 1.0:
            rgb_image = rgb_image / 255.0
        for obj_name, color in self.OBJECT_COLORS.items():
            mask = self._create_color_mask(rgb_image, color)
            
            if mask.sum() > 0:
                points_3d = self._depth_to_pointcloud(depth_image, mask)
                
                if len(points_3d) > 10:  
                    # Compute object properties
                    centroid = np.mean(points_3d, axis=0)
                    bbox_min = np.min(points_3d, axis=0)
                    bbox_max = np.max(points_3d, axis=0)
                    size = bbox_max - bbox_min
                    
                    obj_info = {
                        'name': obj_name,
                        'color': color,
                        'centroid': centroid,
                        'bbox_min': bbox_min,
                        'bbox_max': bbox_max,
                        'size': size,
                        'points': points_3d,
                        'pixel_count': mask.sum()
                    }
                    
                    # Categorize detected object
                    if 'table' in obj_name:
                        detected_objects['table'].append(obj_info)
                    elif 'target' in obj_name:
                        detected_objects['target'].append(obj_info)
                    elif 'obstacle' in obj_name:
                        detected_objects['obstacles'].append(obj_info)
        
        return detected_objects
    
    def _create_color_mask(self, rgb_image: np.ndarray, target_color: np.ndarray) -> np.ndarray:
        color_diff = np.linalg.norm(rgb_image - target_color, axis=2)
        mask = color_diff < self.color_tolerance
        return mask
    def _depth_to_pointcloud(self, depth_image: np.ndarray, 
                            mask: np.ndarray = None,
                            fx: float = 525.0, 
                            fy: float = 525.0,
                            cx: float = None, 
                            cy: float = None) -> np.ndarray:
        h, w = depth_image.shape
        
        if cx is None:
            cx = w / 2.0
        if cy is None:
            cy = h / 2.0
        
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        if mask is not None:
            u = u[mask]
            v = v[mask]
            depth = depth_image[mask]
        else:
            u = u.flatten()
            v = v.flatten()
            depth = depth_image.flatten()
        
        valid = (depth > 0) & (depth < 100) 
        u = u[valid]
        v = v[valid]
        depth = depth[valid]
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        points_3d = np.stack([x, y, z], axis=1)
        
        return points_3d
    
    def identify_table_plane_ransac(self, points_3d: np.ndarray,
                                   max_iterations: int = 1000,
                                   min_inliers: int = 100) -> Optional[Dict]:
        if len(points_3d) < 3:
            return None
        
        best_plane = None
        best_inliers = 0
        
        for _ in range(max_iterations):
            idx = np.random.choice(len(points_3d), 3, replace=False)
            p1, p2, p3 = points_3d[idx]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6: 
                continue
            normal = normal / norm
            d = -np.dot(normal, p1)
            distances = np.abs(np.dot(points_3d, normal) + d)
            inliers = distances < self.ransac_threshold
            num_inliers = np.sum(inliers)
            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_plane = {
                    'normal': normal,
                    'd': d,
                    'inliers': inliers,
                    'num_inliers': num_inliers,
                    'centroid': np.mean(points_3d[inliers], axis=0)
                }
        if best_plane and best_plane['num_inliers'] >= min_inliers:
            return best_plane
        
        return None
    
    def compute_grasp_pose_pca(self, points_3d: np.ndarray) -> Dict:
        if len(points_3d) < 3:
            raise ValueError("Need at least 3 points for PCA")
        centroid = np.mean(points_3d, axis=0)
        centered_points = points_3d - centroid
        pca = PCA(n_components=3)
        pca.fit(centered_points)
        principal_axes = pca.components_ 
        explained_variance = pca.explained_variance_
        projections = np.dot(centered_points, principal_axes.T)
        dimensions = np.max(projections, axis=0) - np.min(projections, axis=0)
        approach_idx = np.argmin(explained_variance)
        approach_vector = principal_axes[approach_idx]
        if approach_vector[2] < 0:
            approach_vector = -approach_vector
        grasp_width = np.min(dimensions)
        
        return {
            'centroid': centroid,
            'principal_axes': principal_axes,
            'dimensions': dimensions,
            'approach_vector': approach_vector,
            'grasp_width': grasp_width,
            'explained_variance': explained_variance
        }
    
    def compute_avoidance_pose_pca(self, points_3d: np.ndarray) -> Dict:
        if len(points_3d) < 3:
            raise ValueError("Need at least 3 points for PCA")
        centroid = np.mean(points_3d, axis=0)
        centered_points = points_3d - centroid
        pca = PCA(n_components=3)
        pca.fit(centered_points)
        principal_axes = pca.components_
        projections = np.dot(centered_points, principal_axes.T)
        dimensions = np.max(projections, axis=0) - np.min(projections, axis=0)
        bounding_radius = np.max(np.linalg.norm(centered_points, axis=1))
        longest_axis_idx = np.argmax(dimensions)
        longest_axis = principal_axes[longest_axis_idx]
        xy_component = longest_axis[:2]
        if np.linalg.norm(xy_component) > 1e-6:
            xy_normalized = xy_component / np.linalg.norm(xy_component)
            avoidance_directions = [
                np.array([xy_normalized[1], -xy_normalized[0], 0]),
                np.array([-xy_normalized[1], xy_normalized[0], 0]) 
            ]
        else:
            avoidance_directions = [
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([-1, 0, 0]),
                np.array([0, -1, 0])
            ]
        
        return {
            'centroid': centroid,
            'principal_axes': principal_axes,
            'dimensions': dimensions,
            'avoidance_directions': avoidance_directions,
            'bounding_radius': bounding_radius
        }
    
    def process_scene(self, rgb_image: np.ndarray, 
                     depth_image: np.ndarray) -> Dict:
        scene = {
            'table': None,
            'target': None,
            'obstacles': [],
            'timestamp': None
        }
        detected = self.detect_objects_by_color(rgb_image, depth_image)
        if detected['table']:
            table_obj = detected['table'][0]
            table_points = table_obj['points']
            
            plane_info = self.identify_table_plane_ransac(table_points)
            
            if plane_info:
                table_obj['plane'] = plane_info
                scene['table'] = table_obj
        if detected['target']:
            target_obj = detected['target'][0]
            target_points = target_obj['points']
            
            try:
                grasp_pose = self.compute_grasp_pose_pca(target_points)
                target_obj['grasp_pose'] = grasp_pose
                scene['target'] = target_obj
            except ValueError:
                scene['target'] = target_obj
        for obstacle_obj in detected['obstacles']:
            obstacle_points = obstacle_obj['points']
            
            try:
                avoidance_pose = self.compute_avoidance_pose_pca(obstacle_points)
                obstacle_obj['avoidance_pose'] = avoidance_pose
                scene['obstacles'].append(obstacle_obj)
            except ValueError:
                scene['obstacles'].append(obstacle_obj)
        
        return scene
    
    def visualize_detection(self, rgb_image: np.ndarray, 
                           scene: Dict) -> np.ndarray:
        vis_image = rgb_image.copy()
        
        if vis_image.max() <= 1.0:
            vis_image = (vis_image * 255).astype(np.uint8)
        if scene['table']:
            cv2.putText(vis_image, "TABLE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (139, 76, 25), 2)
        if scene['target']:
            cv2.putText(vis_image, "TARGET", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        for i, obs in enumerate(scene['obstacles']):
            cv2.putText(vis_image, f"OBSTACLE {i+1}", (10, 90 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return vis_image

def preprocess_sensor_data(rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    depth_filtered = cv2.medianBlur(depth.astype(np.float32), 5)
    mask = (depth_filtered == 0)
    if mask.sum() > 0:
        depth_filtered = cv2.inpaint(depth_filtered, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return rgb, depth_filtered


def extract_target_position(scene: Dict) -> Optional[np.ndarray]:
    if scene['target'] and 'grasp_pose' in scene['target']:
        return scene['target']['grasp_pose']['centroid']
    elif scene['target']:
        return scene['target']['centroid']
    return None

def extract_obstacle_positions(scene: Dict) -> List[np.ndarray]:
    positions = []
    for obs in scene['obstacles']:
        if 'avoidance_pose' in obs:
            positions.append(obs['avoidance_pose']['centroid'])
        else:
            positions.append(obs['centroid'])
    return positions
if __name__ == "__main__":
    perception = PerceptionModule(color_tolerance=0.2, ransac_threshold=0.01)