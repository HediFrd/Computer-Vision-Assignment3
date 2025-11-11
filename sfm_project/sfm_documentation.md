# Structure-from-Motion (SfM) Project Documentation
## AI6121 - 3D Vision Course
### Lee Wee Nam Library Reconstruction - NTU

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation & Setup](#installation--setup)
3. [Core Components](#core-components)
4. [Algorithm Details](#algorithm-details)
5. [Usage Guide](#usage-guide)
6. [Advanced Techniques](#advanced-techniques)
7. [Troubleshooting](#troubleshooting)
8. [References](#references)

---

## Project Overview

This project implements a complete Structure-from-Motion pipeline from scratch using Python, OpenCV, NumPy, and SciPy. The implementation reconstructs 3D scenes from video frames without relying on pre-built SfM libraries.

### Project Objectives
- Extract multi-view 2D images from a 40-second video of Lee Wee Nam Library
- Detect and match features across frames
- Estimate camera poses and 3D scene structure
- Generate a point cloud representation of the scene
- Provide detailed analysis and evaluation

### Key Features
✓ Complete from-scratch implementation (no COLMAP/Meshroom required)
✓ SIFT feature detection and matching
✓ Robust fundamental matrix estimation with RANSAC
✓ Direct Linear Transform (DLT) triangulation
✓ Multi-view 3D reconstruction capability
✓ Point cloud visualization and export
✓ Extensive documentation and error handling

---

## Installation & Setup

### System Requirements
```
Python 3.7+
RAM: 4 GB minimum (8 GB recommended)
Disk: 10 GB for video and results
OS: Windows, macOS, or Linux
```

### Step 1: Clone/Download Project Files
```bash
# Create project directory
mkdir sfm_project
cd sfm_project

# Place the following files:
# - sfm_pipeline.py (main implementation)
# - sfm_advanced_guide.py (advanced techniques)
# - sfm_setup.py (setup utilities)
```

### Step 2: Install Dependencies
```bash
# Method 1: Automated setup
python sfm_setup.py

# Method 2: Manual installation
pip install opencv-python==4.8.0.76
pip install numpy==1.24.3
pip install scipy==1.11.2
pip install matplotlib==3.7.2

# Optional (for advanced visualization)
pip install open3d==0.17.0
pip install scikit-learn==1.3.0
```

### Step 3: Verify Installation
```python
import cv2
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")
print("All dependencies installed successfully!")
```

---

## Core Components

### 1. Video Frame Extraction
**Function**: `extract_frames_from_video(video_path, output_dir, frame_skip=5)`

Extracts individual frames from the video at regular intervals to create a multi-view dataset.

```python
frame_list = extract_frames_from_video(
    "video.mp4", 
    output_dir="frames/", 
    frame_skip=5  # Extract every 5th frame
)
```

**Parameters**:
- `frame_skip`: Controls frame sampling rate
  - Value of 5 for 30fps video ≈ 6fps sample rate
  - Higher values: fewer frames, faster processing
  - Lower values: more frames, more comprehensive coverage

### 2. Feature Detection & Extraction
**Function**: `detect_and_extract_features(image_path)`

Uses SIFT (Scale-Invariant Feature Transform) to detect distinctive keypoints and extract their descriptors.

```python
kp, des, img = detect_and_extract_features("frame_00001.jpg")
print(f"Found {len(kp)} keypoints")
```

**SIFT Advantages**:
- Invariant to scale, rotation, and illumination changes
- Robust to noise and occlusion
- Produces distinctive descriptors
- Well-suited for architectural scenes

### 3. Feature Matching
**Function**: `match_features(des1, des2, ratio_test_threshold=0.7)`

Matches SIFT descriptors between consecutive frames using Lowe's ratio test.

```python
matches = match_features(des1, des2, ratio_test_threshold=0.7)
pts1, pts2 = extract_matched_points(kp1, kp2, matches)
```

**Lowe's Ratio Test**:
- Compares distance to nearest and second-nearest neighbor
- Filters ambiguous matches: d_nearest < 0.7 * d_second_nearest
- Removes ~90% of false matches while keeping good correspondences

### 4. Camera Calibration
**Function**: `get_camera_intrinsics(image_shape)`

Estimates camera intrinsic parameters for uncalibrated cameras.

```python
K = get_camera_intrinsics((height, width))
# K = [[f, 0, cx],
#      [0, f, cy],
#      [0, 0, 1]]
```

**For uncalibrated scenarios**:
- Focal length: f ≈ max(height, width)
- Principal point: (cx, cy) ≈ (width/2, height/2)
- Zero skew assumption

### 5. Fundamental Matrix Estimation
**Functions**: 
- `compute_fundamental_matrix_8point(pts1, pts2)` - Normalized 8-point algorithm
- `compute_fundamental_matrix_ransac(pts1, pts2)` - RANSAC-based robust estimation

```python
F_normalized, inlier_mask = compute_fundamental_matrix_ransac(
    pts1, pts2,
    max_iterations=1000,
    threshold=1.0
)
```

**8-Point Algorithm**:
1. Normalize point coordinates for numerical stability
2. Build design matrix A from normalized correspondences
3. SVD decomposition: A = U Σ V^T
4. F = reshape(V^T[-1]) and enforce rank-2 constraint
5. Denormalize to original coordinates

**RANSAC for Robustness**:
1. Random sampling of 8 correspondences
2. Compute F from sample
3. Count inliers (epipolar error < threshold)
4. Repeat and keep F with most inliers
5. Typical iterations: 1000-2000 for 50% inlier ratio

### 6. Essential Matrix & Camera Pose
**Functions**:
- `compute_essential_matrix(F, K)` - Convert fundamental to essential matrix
- `decompose_essential_matrix(E)` - Get R,t pairs
- `recover_pose_from_essential_matrix(E, pts1, pts2, K)` - Select valid pose

```python
E = compute_essential_matrix(F, K)
R, t = recover_pose_from_essential_matrix(E, pts1[inliers], pts2[inliers], K)
```

**Essential Matrix Properties**:
- E = K^T F K (relates calibrated cameras)
- Has rank 2 with one singular value = 0
- Decomposes into 4 possible (R,t) pairs
- Valid pair: points triangulated in front of both cameras

### 7. Triangulation (DLT)
**Function**: `triangulate_point(P1, P2, pt1, pt2)`

Reconstructs 3D points from two 2D correspondences using Direct Linear Transform.

```python
points_3d = triangulate_points(P1, P2, pts1[inliers], pts2[inliers])
```

**DLT Method**:
1. Form 4×4 system: A·X = 0
   - Row 1: pt1[1]·P1[2,:] - P1[1,:]
   - Row 2: P1[0,:] - pt1[0]·P1[2,:]
   - Row 3: pt2[1]·P2[2,:] - P2[1,:]
   - Row 4: P2[0,:] - pt2[0]·P2[2,:]
2. SVD decomposition: A = U Σ V^T
3. X = V^T[-1] / V^T[-1][3]

### 8. Bundle Adjustment
**Function**: `simple_bundle_adjustment(points_3d, points_2d_list, K, camera_matrices)`

Refines 3D structure and camera parameters by minimizing reprojection error.

```python
points_3d_refined = simple_bundle_adjustment(
    points_3d, 
    [pts1, pts2], 
    K, 
    [P1, P2],
    max_iterations=10
)
```

**Optimization Objective**:
```
minimize: Σ ||projected_2d - observed_2d||²
```

---

## Algorithm Details

### Complete SfM Pipeline Flow

```
Input Video
    ↓
Frame Extraction → Feature Detection (SIFT)
    ↓
Feature Matching (Lowe's Ratio Test)
    ↓
Get Matched Points
    ↓
Fundamental Matrix (8-point + RANSAC)
    ↓
Essential Matrix & Pose Recovery
    ↓
Triangulation (DLT)
    ↓
Bundle Adjustment (Optional)
    ↓
Outlier Filtering
    ↓
Point Cloud Visualization & Export
```

### Mathematical Formulations

#### 1. Epipolar Constraint
For corresponding points x₁ and x₂ in two images:
```
x₂^T F x₁ = 0
```
where F is the fundamental matrix.

#### 2. Projection Matrix
For point X in 3D and image projection x:
```
λx = P·X = K[R|t]·X
```
where λ is arbitrary scale, K is intrinsic, R is rotation, t is translation.

#### 3. Triangulation (Least Squares)
Given P₁, P₂ and corresponding points x₁, x₂:
```
minimize ||[x₁×]P₁X|| + ||[x₂×]P₂X||
```
where [x×] is the skew-symmetric cross-product matrix.

---

## Usage Guide

### Basic Usage
```python
from sfm_pipeline import run_sfm_pipeline

# Run complete pipeline
results = run_sfm_pipeline(
    video_path="lee_wee_nam_library.mp4",
    output_dir="results"
)

# Access results
print(f"3D Points: {len(results['points_3d'])}")
print(f"Camera Matrix:\n{results['K']}")
print(f"Rotation:\n{results['R']}")
print(f"Translation: {results['t']}")
```

### Output Files
```
results/
├── frames/
│   ├── frame_00000.jpg
│   ├── frame_00001.jpg
│   └── ...
├── pointcloud.ply          # 3D point cloud
├── pointcloud.png          # Visualization
└── camera_params.json      # Camera parameters
```

### Visualization
```python
from sfm_pipeline import visualize_point_cloud_matplotlib

visualize_point_cloud_matplotlib(
    results['points_3d'],
    output_path="result.png"
)
```

### Advanced Usage - Multi-Frame SfM
```python
from sfm_advanced_guide import IncrementalSfM

sfm = IncrementalSfM(K=camera_matrix, max_frames=20)
camera_poses = sfm.run_full_reconstruction(frame_list)
```

---

## Advanced Techniques

### 1. Adaptive Keyframe Selection
Reduces redundant frames by selecting only keyframes with significant motion:

```python
from sfm_advanced_guide import adaptive_frame_selection

selected_frames = adaptive_frame_selection(
    frame_paths,
    motion_threshold=30,
    max_frames=15
)
```

### 2. Robust Outlier Filtering

**Statistical Filtering**:
```python
points_filtered = filter_outliers_statistical(
    points_3d,
    threshold=3.0  # Z-score threshold
)
```

**Spatial Filtering**:
```python
points_filtered = filter_outliers_spatial(
    points_3d,
    radius=10.0,
    min_neighbors=5
)
```

### 3. Multiple Export Formats
```python
# PLY Format
save_point_cloud_ply(points_3d, "output.ply")

# OBJ Format
export_point_cloud_obj(points_3d, "output.obj")

# NumPy Format
np.save("points.npy", points_3d)
```

### 4. Open3D Visualization
```python
from sfm_advanced_guide import visualize_with_open3d

visualize_with_open3d(points_3d, colors=None)
```

---

## Troubleshooting

### Issue 1: "Not enough matches found"
**Causes**:
- Insufficient camera motion in video
- Too few extracted frames
- Poor texture/features in scene

**Solutions**:
```python
# Reduce frame_skip to extract more frames
frame_list = extract_frames_from_video(video_path, skip=3)

# Increase SIFT features
sift = cv2.SIFT_create(nfeatures=10000)

# Use longer video with more camera movement
```

### Issue 2: "Poor reconstruction quality"
**Causes**:
- Not enough inliers in RANSAC
- Weak camera poses
- Outliers in point cloud

**Solutions**:
```python
# Increase RANSAC iterations
F, mask = compute_fundamental_matrix_ransac(
    pts1, pts2,
    max_iterations=5000,
    threshold=0.5
)

# Apply outlier filtering
points_filtered = filter_outliers_statistical(points_3d, threshold=2.0)

# Use bundle adjustment
points_refined = simple_bundle_adjustment(points_3d, [pts1, pts2], K, [P1, P2])
```

### Issue 3: "Degenerate scenes"
**Causes**:
- Camera purely rotating (no translation)
- All points on a plane
- Repeated textures

**Solutions**:
```python
# Ensure sufficient baseline between frames
# Use adaptive keyframe selection for better coverage
selected = adaptive_frame_selection(frame_list, motion_threshold=50)

# Capture from varied distances
```

### Issue 4: "Out of memory"
**Causes**:
- Too many frames processed simultaneously
- High resolution frames
- Large feature descriptor matrices

**Solutions**:
```python
# Resize frames
def resize_frame(frame, max_width=1280):
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_h = int(h * scale)
        return cv2.resize(frame, (max_width, new_h))
    return frame

# Reduce number of frames
frame_list = frame_list[::10]  # Every 10th frame
```

---

## Parameter Configuration

### Recommended Settings for Lee Wee Nam Library

```python
# Video Processing
FRAME_SKIP = 4              # ~7.5 fps from 30 fps video
IMAGE_MAX_WIDTH = 1280
IMAGE_MAX_HEIGHT = 720

# Feature Detection
SIFT_FEATURES = 5000        # SIFT keypoints per frame
MIN_FEATURES = 20           # Minimum keypoints for valid frame

# Feature Matching
MATCH_RATIO_TEST = 0.7      # Lowe's ratio
MIN_MATCHES = 20            # Minimum good matches

# Robust Estimation
RANSAC_ITERATIONS = 2000
RANSAC_THRESHOLD = 1.0      # pixels
MIN_INLIERS = 15

# Filtering
ZSCORE_THRESHOLD = 3.0      # Statistical outlier removal
SPATIAL_RADIUS = 5.0        # Local density filtering
MIN_NEIGHBORS = 5
```

---

## Report Structure

Your project report should include:

### 1. Introduction
- Problem formulation
- Related work (reference papers)
- Contribution of your implementation

### 2. Methodology
- Detailed explanation of SfM algorithm
- Mathematical formulations
- Implementation details for each component
- Diagrams of pipeline

### 3. Experimental Setup
- Video capture details (location, camera, resolution, duration)
- Parameter settings used
- Computational resources

### 4. Results
- Number of frames extracted
- Feature matching statistics
- Point cloud statistics (number of points, density)
- Camera pose estimates
- Visualizations (point clouds, camera trajectories)

### 5. Analysis & Discussion
- Quality of reconstruction
- Challenges encountered
- Limitations of approach
- Comparison with ground truth (if available)

### 6. Conclusion
- Summary of achievements
- Future improvements
- Lessons learned

---

## References

### Primary References
[1] Schönberger, J. L., & Frahm, J. M. (2016). Structure-from-motion revisited. CVPR.
[2] Schönberger, J. L., Zheng, E., Frahm, J. M., et al. (2016). Pixelwise view selection for unstructured multi-view stereo. ECCV.

### Algorithm References
- SIFT: Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. IJCV.
- Fundamental Matrix: Hartley, R., & Zisserman, A. (2003). Multiple view geometry in computer vision.
- Triangulation: Lindstrom, P. (2010). Point-to-line distance triangulation. 3DV.
- Bundle Adjustment: Triggs, B., et al. (1999). Bundle adjustment—a modern synthesis. ICCV.

### Useful Resources
- OpenCV Documentation: https://docs.opencv.org/
- Multiple View Geometry Book: http://www.robots.ox.ac.uk/~vgg/hzbook/
- Open3D Documentation: http://www.open3d.org/

---

## Final Notes

- **Important**: You must declare all AI/LLM usage in your report. Any undeclared AI assistance will be treated as plagiarism.
- Ensure your 3D reconstruction includes detailed analysis specific to Lee Wee Nam Library architecture.
- The point cloud should show recognizable architectural features of the building.
- Test your code thoroughly before submission.
- Include proper error handling and logging in your final submission.
- Keep computational efficiency in mind (runtime should be reasonable for 40s video).

**Submission Deadline**: November 21, 2025