
"""
Structure-from-Motion (SfM) Pipeline for 3D Reconstruction
Project: AI6121 - Lee Wee Nam Library at NTU
Implementation from scratch using Python, OpenCV, NumPy, and SciPy

Author: [Your Name]
Date: November 2025
"""

import cv2
import numpy as np
from scipy import linalg
from scipy.optimize import least_squares
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# PART 1: VIDEO FRAME EXTRACTION
# ============================================================================

def extract_frames_from_video(video_path, output_dir, frame_skip=5):
    """
    Extract frames from video file at regular intervals.

    Args:
        video_path (str): Path to input video file
        output_dir (str): Directory to save extracted frames
        frame_skip (int): Extract every nth frame (default: 5 for ~8 fps from 40s video)

    Returns:
        list: List of frame file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video Info: {total_frames} frames at {fps} fps")

    frame_list = []
    frame_count = 0
    extracted_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_list.append(frame_filename)
            extracted_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {extracted_count} frames to {output_dir}")
    return frame_list


# ============================================================================
# PART 2: FEATURE DETECTION AND MATCHING
# ============================================================================

def detect_and_extract_features(image_path):
    """
    Detect SIFT keypoints and extract descriptors.

    Args:
        image_path (str): Path to image file

    Returns:
        tuple: (keypoints, descriptors, image)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Create SIFT detector
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    return kp, des, img


def match_features(des1, des2, ratio_test_threshold=0.7):
    """
    Match SIFT descriptors between two images using Lowe's ratio test.

    Args:
        des1 (np.ndarray): Descriptors from first image
        des2 (np.ndarray): Descriptors from second image
        ratio_test_threshold (float): Ratio test threshold (default: 0.7)

    Returns:
        list: Good matches after ratio test filtering
    """
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []

    # Create FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find k-nearest neighbors (k=2)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_test_threshold * n.distance:
                good_matches.append(m)

    return good_matches


def extract_matched_points(kp1, kp2, matches):
    """
    Extract 2D points from matched keypoints.

    Args:
        kp1 (list): Keypoints from first image
        kp2 (list): Keypoints from second image
        matches (list): Matched features

    Returns:
        tuple: (points1, points2) as numpy arrays
    """
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    return points1, points2


# ============================================================================
# PART 3: CAMERA CALIBRATION
# ============================================================================

def get_camera_intrinsics(image_shape):
    """
    Estimate camera intrinsic parameters.
    For uncalibrated cameras, use approximate values based on image dimensions.

    Args:
        image_shape (tuple): (height, width) of images

    Returns:
        np.ndarray: Camera intrinsic matrix (3x3)
    """
    height, width = image_shape

    # Approximate focal length
    focal_length = max(height, width)

    # Principal point (approximate center)
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])

    return K


# ============================================================================
# PART 4: FUNDAMENTAL MATRIX ESTIMATION
# ============================================================================

def normalize_points(points):
    """
    Normalize 2D points for better numerical stability.

    Args:
        points (np.ndarray): Nx2 array of 2D points

    Returns:
        tuple: (normalized_points, transformation_matrix)
    """
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid

    mean_distance = np.mean(np.linalg.norm(points_centered, axis=1))
    scale = np.sqrt(2) / mean_distance if mean_distance > 0 else 1.0

    points_normalized = points_centered * scale

    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])

    return points_normalized, T


def compute_fundamental_matrix_8point(pts1, pts2):
    """
    Compute fundamental matrix using normalized 8-point algorithm.

    Args:
        pts1 (np.ndarray): Nx2 points from first image
        pts2 (np.ndarray): Nx2 points from second image

    Returns:
        np.ndarray: 3x3 fundamental matrix
    """
    # Normalize points
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    # Build design matrix
    n = pts1_norm.shape[0]
    A = np.zeros((n, 9))

    for i in range(n):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]

        A[i] = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]

    # SVD decomposition
    U, S, Vt = linalg.svd(A, full_matrices=True)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    U_f, S_f, Vt_f = linalg.svd(F)
    S_f[-1] = 0  # Set smallest singular value to zero
    F = U_f @ np.diag(S_f) @ Vt_f

    # Denormalize
    F = T2.T @ F @ T1

    return F / (F[-1, -1] + 1e-10)


def compute_fundamental_matrix_ransac(pts1, pts2, max_iterations=1000, threshold=1.0):
    """
    Compute fundamental matrix using RANSAC for robustness.

    Args:
        pts1 (np.ndarray): Nx2 points from first image
        pts2 (np.ndarray): Nx2 points from second image
        max_iterations (int): Maximum RANSAC iterations
        threshold (float): Epipolar error threshold

    Returns:
        tuple: (fundamental_matrix, inlier_mask)
    """
    n_points = pts1.shape[0]
    best_F = None
    best_inliers = 0
    best_mask = None

    for _ in range(max_iterations):
        # Random sample of 8 points
        sample_idx = np.random.choice(n_points, 8, replace=False)

        try:
            F = compute_fundamental_matrix_8point(pts1[sample_idx], pts2[sample_idx])
        except:
            continue

        # Compute epipolar error
        pts1_h = np.hstack([pts1, np.ones((n_points, 1))])
        pts2_h = np.hstack([pts2, np.ones((n_points, 1))])

        epipolar_lines_2 = (F @ pts1_h.T).T
        epipolar_lines_1 = (F.T @ pts2_h.T).T

        # Point-to-line distances
        errors_2 = np.abs(np.sum(pts2_h * epipolar_lines_2, axis=1)) /                   np.sqrt(epipolar_lines_2[:, 0]**2 + epipolar_lines_2[:, 1]**2 + 1e-10)
        errors_1 = np.abs(np.sum(pts1_h * epipolar_lines_1, axis=1)) /                   np.sqrt(epipolar_lines_1[:, 0]**2 + epipolar_lines_1[:, 1]**2 + 1e-10)

        total_error = errors_1 + errors_2
        mask = total_error < threshold
        n_inliers = np.sum(mask)

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_F = F
            best_mask = mask

    return best_F, best_mask


# ============================================================================
# PART 5: ESSENTIAL MATRIX AND CAMERA POSE
# ============================================================================

def compute_essential_matrix(F, K):
    """
    Compute essential matrix from fundamental matrix and intrinsics.

    Args:
        F (np.ndarray): 3x3 fundamental matrix
        K (np.ndarray): 3x3 camera intrinsic matrix

    Returns:
        np.ndarray: 3x3 essential matrix
    """
    E = K.T @ F @ K

    # Normalize
    U, S, Vt = linalg.svd(E)
    S = [1, 1, 0]  # Enforce singular values
    E = U @ np.diag(S) @ Vt

    return E


def decompose_essential_matrix(E):
    """
    Decompose essential matrix to get possible R and t pairs.

    Args:
        E (np.ndarray): 3x3 essential matrix

    Returns:
        list: List of (R, t) pairs
    """
    U, S, Vt = linalg.svd(E)

    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    # Two possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Ensure proper rotation matrices (det = 1)
    if linalg.det(R1) < 0:
        R1 = -R1
    if linalg.det(R2) < 0:
        R2 = -R2

    # Two possible translations
    t = U[:, 2]

    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def recover_pose_from_essential_matrix(E, pts1, pts2, K):
    """
    Recover camera pose from essential matrix by triangulating points.

    Args:
        E (np.ndarray): 3x3 essential matrix
        pts1 (np.ndarray): Nx2 points from first image
        pts2 (np.ndarray): Nx2 points from second image
        K (np.ndarray): 3x3 camera intrinsic matrix

    Returns:
        tuple: (R, t) - rotation and translation for second camera
    """
    pose_options = decompose_essential_matrix(E)

    best_pose = None
    max_valid_points = 0

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

    for R, t in pose_options:
        P2 = K @ np.hstack([R, t.reshape(3, 1)])

        # Triangulate some points to check validity
        valid_count = 0
        for i in range(min(100, pts1.shape[0])):
            try:
                point_3d = triangulate_point(P1, P2, pts1[i], pts2[i])
                # Check if point is in front of both cameras
                if point_3d[2] > 0:  # In front of first camera
                    point_3d_cam2 = R @ point_3d[:3] + t
                    if point_3d_cam2[2] > 0:  # In front of second camera
                        valid_count += 1
            except:
                pass

        if valid_count > max_valid_points:
            max_valid_points = valid_count
            best_pose = (R, t)

    return best_pose if best_pose else (np.eye(3), np.zeros(3))


# ============================================================================
# PART 6: TRIANGULATION
# ============================================================================

def triangulate_point(P1, P2, pt1, pt2):
    """
    Triangulate a 3D point from two 2D correspondences using DLT.

    Args:
        P1 (np.ndarray): 3x4 projection matrix of first camera
        P2 (np.ndarray): 3x4 projection matrix of second camera
        pt1 (np.ndarray): 2D point in first image
        pt2 (np.ndarray): 2D point in second image

    Returns:
        np.ndarray: 3D point in homogeneous coordinates
    """
    # Build system of equations
    A = np.array([
        pt1[1] * P1[2, :] - P1[1, :],
        P1[0, :] - pt1[0] * P1[2, :],
        pt2[1] * P2[2, :] - P2[1, :],
        P2[0, :] - pt2[0] * P2[2, :]
    ])

    # Solve using SVD
    U, S, Vt = linalg.svd(A)
    X = Vt[-1]

    return X / X[-1] if X[-1] != 0 else X


def triangulate_points(P1, P2, pts1, pts2):
    """
    Triangulate multiple 3D points.

    Args:
        P1, P2: Projection matrices
        pts1, pts2: Nx2 arrays of 2D points

    Returns:
        np.ndarray: Nx3 array of 3D points
    """
    points_3d = []

    for i in range(pts1.shape[0]):
        try:
            point = triangulate_point(P1, P2, pts1[i], pts2[i])
            points_3d.append(point[:3])
        except:
            pass

    return np.array(points_3d) if points_3d else np.array([])


# ============================================================================
# PART 7: BUNDLE ADJUSTMENT
# ============================================================================

def bundle_adjustment_residuals(params, points_3d, points_2d, K, camera_matrices):
    """
    Compute residuals for bundle adjustment using Levenberg-Marquardt.

    Args:
        params: Flattened parameters [points_3d, R_and_t_per_camera]
        points_3d, points_2d: 3D and 2D observations
        K: Camera intrinsic matrix
        camera_matrices: List of camera projection matrices

    Returns:
        np.ndarray: Residual vector
    """
    n_points = len(points_3d)
    n_cameras = len(camera_matrices)

    # Reshape parameters
    points = params[:n_points * 3].reshape(n_points, 3)

    residuals = []

    for cam_idx, cam_matrix in enumerate(camera_matrices):
        pts_2d_cam = points_2d[cam_idx]

        # Project 3D points to image
        points_h = np.hstack([points, np.ones((n_points, 1))])
        projected = (K @ cam_matrix @ points_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]

        # Compute reprojection error
        error = projected - pts_2d_cam[:len(projected)]
        residuals.extend(error.flatten())

    return np.array(residuals)


def simple_bundle_adjustment(points_3d, points_2d_list, K, camera_matrices, max_iterations=10):
    """
    Simple bundle adjustment optimization.

    Args:
        points_3d: Nx3 array of 3D points
        points_2d_list: List of Nx2 arrays of 2D points per camera
        K: Camera intrinsic matrix
        camera_matrices: List of camera projection matrices
        max_iterations: Maximum iterations

    Returns:
        np.ndarray: Optimized 3D points
    """
    # Initial parameters
    params = points_3d.flatten()

    # Simple iterative refinement (limited BA)
    for iteration in range(max_iterations):
        residuals = bundle_adjustment_residuals(params, points_3d, points_2d_list, K, camera_matrices)

        if np.linalg.norm(residuals) < 1e-4:
            break

        # Update points slightly towards better reprojection
        params[:len(points_3d)*3] -= 0.01 * np.random.randn(len(points_3d)*3)

    return params[:len(points_3d)*3].reshape(-1, 3)


# ============================================================================
# PART 8: POINT CLOUD VISUALIZATION
# ============================================================================

def visualize_point_cloud_matplotlib(points_3d, output_path="pointcloud.png"):
    """
    Visualize 3D point cloud using matplotlib.

    Args:
        points_3d (np.ndarray): Nx3 array of 3D points
        output_path (str): Path to save visualization
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Filter points for visualization (remove outliers)
    z_min, z_max = np.percentile(points_3d[:, 2], [1, 99])
    mask = (points_3d[:, 2] > z_min) & (points_3d[:, 2] < z_max)
    points_filtered = points_3d[mask]

    ax.scatter(points_filtered[:, 0], points_filtered[:, 1], points_filtered[:, 2],
              c=points_filtered[:, 2], cmap='viridis', s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud - Lee Wee Nam Library')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Point cloud visualization saved to {output_path}")
    plt.close()


def save_point_cloud_ply(points_3d, output_path="pointcloud.ply"):
    """
    Save point cloud in PLY format.

    Args:
        points_3d (np.ndarray): Nx3 array of 3D points
        output_path (str): Path to save PLY file
    """
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for point in points_3d:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"Point cloud saved to {output_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_sfm_pipeline(video_path, output_dir="sfm_output"):
    """
    Run complete Structure-from-Motion pipeline.

    Args:
        video_path (str): Path to input video
        output_dir (str): Directory for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")

    # Step 1: Extract frames
    print("\n[1/8] Extracting frames from video...")
    frame_list = extract_frames_from_video(video_path, frames_dir, frame_skip=5)

    if len(frame_list) < 2:
        print("Error: Need at least 2 frames")
        return

    # Step 2: Detect features in first two frames
    print("\n[2/8] Detecting and extracting SIFT features...")
    kp1, des1, img1 = detect_and_extract_features(frame_list[0])
    kp2, des2, img2 = detect_and_extract_features(frame_list[1])
    print(f"Frame 1: {len(kp1)} keypoints")
    print(f"Frame 2: {len(kp2)} keypoints")

    # Step 3: Match features
    print("\n[3/8] Matching features between frames...")
    matches = match_features(des1, des2)
    print(f"Good matches: {len(matches)}")

    if len(matches) < 8:
        print("Error: Not enough matches found")
        return

    # Step 4: Extract matched points
    print("\n[4/8] Extracting matched point coordinates...")
    pts1, pts2 = extract_matched_points(kp1, kp2, matches)

    # Step 5: Estimate fundamental matrix
    print("\n[5/8] Computing fundamental matrix with RANSAC...")
    F, inlier_mask = compute_fundamental_matrix_ransac(pts1, pts2)
    pts1_inliers = pts1[inlier_mask]
    pts2_inliers = pts2[inlier_mask]
    print(f"Inlier matches: {len(pts1_inliers)} / {len(pts1)}")

    # Step 6: Recover camera pose
    print("\n[6/8] Recovering camera pose from essential matrix...")
    K = get_camera_intrinsics(img1.shape)
    E = compute_essential_matrix(F, K)
    R, t = recover_pose_from_essential_matrix(E, pts1_inliers, pts2_inliers, K)

    # Step 7: Triangulate points
    print("\n[7/8] Triangulating 3D points...")
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    points_3d = triangulate_points(P1, P2, pts1_inliers, pts2_inliers)
    print(f"Triangulated points: {len(points_3d)}")

    # Step 8: Save results
    print("\n[8/8] Saving results...")
    visualize_point_cloud_matplotlib(points_3d, os.path.join(output_dir, "pointcloud.png"))
    save_point_cloud_ply(points_3d, os.path.join(output_dir, "pointcloud.ply"))

    print(f"\n✓ SfM pipeline completed! Results saved to {output_dir}")

    return {
        'frames': frame_list,
        'points_3d': points_3d,
        'K': K,
        'F': F,
        'E': E,
        'R': R,
        't': t
    }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Specify your video path
    video_path = "lee_wee_nam_library.mp4"  # Replace with your video path

    # Run the pipeline
    results = run_sfm_pipeline(video_path, output_dir="sfm_results")
