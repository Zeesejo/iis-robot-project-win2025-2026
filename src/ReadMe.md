                    ┌─────────────────────┐
                    │   Input Frame       │
                    │  (RGB + Depth)     │
                    └─────────┬──────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Reshape & Convert Image │
                 │ RGBA → BGR (OpenCV)    │
                 └─────────┬──────────────┘
                           │
                           ▼
            ┌────────────────────────────────┐
            │ Color-Based Object Detection   │
            │ (HSV Threshold + Morphology)  │
            └─────────┬──────────────────────┘
                      │
                      ▼
         ┌──────────────────────────────┐
         │ Edge & Contour Segmentation  │
         │ (Canny + Contours)           │
         └─────────┬────────────────────┘
                   │
                   ▼
        ┌─────────────────────────────────┐
        │ Depth → 3D Point Cloud          │
        │ (Camera Projection Model)       │
        └─────────┬───────────────────────┘
                  │
                  ▼
       ┌────────────────────────────────────┐
       │ RANSAC Plane Segmentation          │
       │ (Table Plane Detection)            │
       └─────────┬──────────────────────────┘
                 │
                 ▼
     ┌──────────────────────────────────────┐
     │ Extract Object Points (Mask + 3D)    │
     │ Remove Plane Inliers                 │
     └─────────┬────────────────────────────┘
               │
      ┌────────┴─────────┐
      │                  │
      ▼                  ▼
┌──────────────┐  ┌─────────────────┐
│ Target (Red) │  │ Obstacles       │
│ Cylinder     │  │ (Colored Cubes) │
└──────┬───────┘  └────────┬────────┘
       │                    │
       ▼                    ▼
┌────────────────┐  ┌────────────────────┐
│ PCA Pose Estim │  │ PCA Pose Estimation│
│ + Cylinder Ref │  │ + Box Refinement   │
└──────┬─────────┘  └────────┬───────────┘
       │                      │
       └──────────┬───────────┘
                  ▼
        ┌───────────────────────┐
        │ Scene Map Update      │
        │ (Lock after 3 obs)    │
        └──────────┬────────────┘
                   ▼
           ┌───────────────────┐
           │ Final Results     │
           │ (Poses + Detections)
           └───────────────────┘
