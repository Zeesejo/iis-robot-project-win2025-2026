import cv2
import numpy as np

# Load the frame
img = cv2.imread('debug_frame_600.jpg')
print(f"Image shape: {img.shape}")

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Current red HSV range (widened)
red_lower1 = np.array([0, 80, 50])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 80, 50])
red_upper2 = np.array([180, 255, 255])

# Create masks
mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
red_mask = mask1 | mask2

# Find red pixels
red_pixels = np.sum(red_mask > 0)
print(f"Red pixels (current range): {red_pixels}")

# Find all unique colors in HSV with significant presence
print("\n=== Color analysis ===")
for h_start in range(0, 180, 20):
    h_end = min(h_start + 20, 180)
    mask = cv2.inRange(hsv, np.array([h_start, 50, 50]), np.array([h_end, 255, 255]))
    pixels = np.sum(mask > 0)
    if pixels > 100:
        print(f"Hue {h_start:3d}-{h_end:3d}: {pixels:6d} pixels")

# Check for very bright red (low saturation might be the issue)
print("\n=== Testing wider red ranges ===")
test_ranges = [
    ("Very wide red (S>30, V>30)", [(0, 30, 30), (10, 255, 255)], [(170, 30, 30), (180, 255, 255)]),
    ("Wide red (S>50, V>50)", [(0, 50, 50), (15, 255, 255)], [(165, 50, 50), (180, 255, 255)]),
    ("Current red", [(0, 80, 50), (10, 255, 255)], [(170, 80, 50), (180, 255, 255)])
]

for name, (l1, u1), (l2, u2) in test_ranges:
    m1 = cv2.inRange(hsv, np.array(l1), np.array(u1))
    m2 = cv2.inRange(hsv, np.array(l2), np.array(u2))
    combined = m1 | m2
    px = np.sum(combined > 0)
    print(f"{name}: {px} pixels")

# Sample actual values at center of image (where cylinder might be)
h, w = img.shape[:2]
cy, cx = h//2, w//2
sample_region = hsv[cy-20:cy+20, cx-20:cx+20]
print(f"\n=== Center region HSV stats (±20px from center) ===")
print(f"H mean: {np.mean(sample_region[:,:,0]):.1f}, min: {np.min(sample_region[:,:,0])}, max: {np.max(sample_region[:,:,0])}")
print(f"S mean: {np.mean(sample_region[:,:,1]):.1f}, min: {np.min(sample_region[:,:,1])}, max: {np.max(sample_region[:,:,1])}")
print(f"V mean: {np.mean(sample_region[:,:,2]):.1f}, min: {np.min(sample_region[:,:,2])}, max: {np.max(sample_region[:,:,2])}")
