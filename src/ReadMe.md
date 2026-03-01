
## Pipeline Overview

```
Input Frame (RGB + Depth)
        │
        ▼
Reshape & Convert Image (RGBA → BGR)
        │
        ▼
Color-Based Object Detection (HSV Threshold + Morphology)
        │
        ▼
Edge & Contour Segmentation (Canny + Contours)
        │
        ▼
Depth → 3D Point Cloud (Camera Projection Model)
        │
        ▼
RANSAC Plane Segmentation (Table Plane Detection)
        │
        ▼
Extract Object Points (Mask + 3D, Remove Plane Inliers)
        │
   ┌────┴────┐
   ▼         ▼
Target     Obstacles
(Red       (Colored
Cylinder)   Cubes)
   │         │
   ▼         ▼
PCA Pose   PCA Pose
Estim +    Estim +
Cylinder   Box
Refinement Refinement
   │         │
   └────┬────┘
        ▼
  Scene Map Update (Lock after 3 observations)
        │
        ▼
  Final Results (Poses + Detections)
```
