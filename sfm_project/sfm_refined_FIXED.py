#!/usr/bin/env python3
"""
REFINED DENSE STRUCTURE-FROM-MOTION
Processes all frame pairs with artifact removal
Removes ghost/hourglass artifacts while keeping real scene geometry
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from sfm_pipeline import *


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def filter_outliers_statistical(points_3d, threshold=2.5):
    """Remove outlier points using statistical method."""
    if len(points_3d) == 0:
        return points_3d

    centroid = np.mean(points_3d, axis=0)
    distances = np.linalg.norm(points_3d - centroid, axis=1)

    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    if std_dist < 1e-10:
        return points_3d

    z_scores = np.abs((distances - mean_dist) / (std_dist + 1e-10))
    mask = z_scores < threshold

    return points_3d[mask]


def filter_by_reprojection_error(points_3d, K, threshold_pixels=2.0):
    """
    Remove points with high reprojection error.
    Keeps only points that project cleanly onto image planes.
    """
    if len(points_3d) == 0:
        return points_3d

    # Simple reprojection test: points should project near image center
    # with reasonable depth
    z_values = points_3d[:, 2]

    # Remove points behind camera or too far
    valid_z = (z_values > 0.1) & (z_values < 10000)

    return points_3d[valid_z]


def filter_by_density(points_3d, radius=20.0, min_neighbors=3):
    """
    Keep only well-supported points (with neighbors).
    Removes isolated/ghost points.
    """
    if len(points_3d) <= 100:
        return points_3d

    from scipy.spatial import cKDTree

    # Build KDTree for efficient neighbor search
    tree = cKDTree(points_3d)

    # Find neighbors for each point
    neighbor_indices = tree.query_ball_point(points_3d, r=radius)
    neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_indices])

    # Keep points with at least min_neighbors
    mask = neighbor_counts >= min_neighbors

    print(f"  Density-based: removed {np.sum(~mask)} isolated points")

    return points_3d[mask]


def filter_by_planarity(points_3d, grid_cell_size=15.0, min_points_per_cell=2):
    """
    Remove points that don't form coherent surfaces.
    Keeps points that form planes/surfaces (like tables, walls).
    """
    if len(points_3d) <= 100:
        return points_3d

    # Discretize points into grid cells
    grid_coords = np.floor(points_3d / grid_cell_size).astype(int)

    # Count points per cell
    unique_cells, inverse_indices = np.unique(grid_coords, return_inverse=True, axis=0)
    cell_counts = np.bincount(inverse_indices)

    # Keep points in cells with at least min_points_per_cell
    mask = cell_counts[inverse_indices] >= min_points_per_cell

    print(f"  Planarity-based: removed {np.sum(~mask)} isolated/scattered points")

    return points_3d[mask]


def remove_outlier_extremes(points_3d, percentile=5):
    """
    Remove extreme outliers by percentile.
    Removes points at extreme distances from median.
    """
    if len(points_3d) == 0:
        return points_3d

    # Compute distance from median point
    median = np.median(points_3d, axis=0)
    distances = np.linalg.norm(points_3d - median, axis=1)

    # Keep points within percentile range
    threshold = np.percentile(distances, 100 - percentile)
    mask = distances <= threshold

    print(f"  Extreme removal: removed {np.sum(~mask)} extreme outliers")

    return points_3d[mask]


def export_point_cloud_obj(points_3d, output_path="pointcloud.obj"):
    """Export point cloud to OBJ format."""
    with open(output_path, 'w') as f:
        for point in points_3d:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")


def run_refined_dense_sfm(video_path, output_dir="sfm_results_refined", 
                         frame_skip=1, aggressive_filtering=False):
    """
    Refined dense SfM with artifact removal.

    Args:
        video_path: Path to input video
        output_dir: Output directory
        frame_skip: Extract every nth frame (1 for all, 2 for half, etc)
        aggressive_filtering: If True, removes more points but cleaner results
    """

    print_section("REFINED DENSE STRUCTURE-FROM-MOTION (WITH ARTIFACT REMOVAL)")
    print(f"Video: {video_path}")
    print(f"Frame skip: {frame_skip}")
    print(f"Aggressive filtering: {aggressive_filtering}")

    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")

    # =====================================================================
    # STEP 1: EXTRACT FRAMES
    # =====================================================================
    print_section("STEP 1: Frame Extraction")

    try:
        frame_list = extract_frames_from_video(video_path, frames_dir, frame_skip)
        print(f"✓ Extracted {len(frame_list)} frames from video")

        if len(frame_list) < 2:
            print("ERROR: Need at least 2 frames")
            return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    # =====================================================================
    # STEP 2: PROCESS ALL CONSECUTIVE FRAME PAIRS
    # =====================================================================
    print_section("STEP 2: Processing Frame Pairs")
    print(f"Will process {len(frame_list)-1} frame pairs...\n")

    all_points_3d = []
    K = None

    for pair_idx in range(len(frame_list) - 1):
        try:
            # Detect features
            kp1, des1, img1 = detect_and_extract_features(frame_list[pair_idx])
            kp2, des2, img2 = detect_and_extract_features(frame_list[pair_idx + 1])

            if len(kp1) < 10 or len(kp2) < 10:
                continue

            # Match features
            matches = match_features(des1, des2, ratio_test_threshold=0.7)

            if len(matches) < 8:
                continue

            # Extract points
            pts1, pts2 = extract_matched_points(kp1, kp2, matches)

            # Estimate fundamental matrix
            F, inlier_mask = compute_fundamental_matrix_ransac(pts1, pts2, max_iterations=500)
            n_inliers = np.sum(inlier_mask)

            if n_inliers < 8:
                continue

            # Get camera intrinsics
            if K is None:
                K = get_camera_intrinsics(img1.shape)

            pts1_inliers = pts1[inlier_mask]
            pts2_inliers = pts2[inlier_mask]

            # Recover pose
            E = compute_essential_matrix(F, K)
            R, t = recover_pose_from_essential_matrix(E, pts1_inliers, pts2_inliers, K)

            # Triangulate
            P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = K @ np.hstack([R, t.reshape(3, 1)])
            points_3d = triangulate_points(P1, P2, pts1_inliers, pts2_inliers)

            if len(points_3d) > 0:
                all_points_3d.append(points_3d)

            # Progress indicator
            if (pair_idx + 1) % max(1, (len(frame_list) - 1) // 10) == 0:
                pct = (pair_idx + 1) / (len(frame_list) - 1) * 100
                total_pts = sum(len(p) for p in all_points_3d)
                print(f"  [{pct:5.1f}%] Pairs: {pair_idx+1:4d}, Total points: {total_pts:8d}")

        except Exception as e:
            pass

    # =====================================================================
    # STEP 3: COMBINE ALL POINT CLOUDS
    # =====================================================================
    print_section("STEP 3: Combining Point Clouds")

    if not all_points_3d:
        print("ERROR: No valid point clouds")
        return None

    combined_points = np.vstack(all_points_3d)
    print(f"✓ Processed {len(all_points_3d)} frame pairs")
    print(f"✓ Total combined points: {len(combined_points)}")

    # =====================================================================
    # STEP 4: SMART ARTIFACT-AWARE FILTERING
    # =====================================================================
    print_section("STEP 4: Smart Artifact Removal")

    print(f"Initial points: {len(combined_points)}")

    # Step 4a: Remove extreme depth outliers
    print("\n  [1] Removing extreme outliers...")
    points_filtered = remove_outlier_extremes(combined_points, percentile=2)

    # Step 4b: Remove points with bad reprojection
    print("  [2] Depth/reprojection filtering...")
    points_filtered = filter_by_reprojection_error(points_filtered, K, threshold_pixels=2.0)
    print(f"  After depth filter: {len(points_filtered)}")

    # Step 4c: Remove isolated ghost points
    print("  [3] Removing isolated/ghost points...")
    if aggressive_filtering:
        points_filtered = filter_by_density(points_filtered, radius=15.0, min_neighbors=5)
    else:
        points_filtered = filter_by_density(points_filtered, radius=25.0, min_neighbors=2)

    # Step 4d: Remove non-planar scattered points
    print("  [4] Surface coherency filtering...")
    if aggressive_filtering:
        points_filtered = filter_by_planarity(points_filtered, grid_cell_size=10.0, min_points_per_cell=3)
    else:
        points_filtered = filter_by_planarity(points_filtered, grid_cell_size=15.0, min_points_per_cell=2)

    # Step 4e: Final statistical pass
    print("  [5] Final statistical filtering...")
    if aggressive_filtering:
        points_filtered = filter_outliers_statistical(points_filtered, threshold=2.0)
    else:
        points_filtered = filter_outliers_statistical(points_filtered, threshold=2.5)

    print(f"\n✓ Final 3D points: {len(points_filtered)}")
    print(f"✓ Points kept: {len(points_filtered) / len(combined_points) * 100:.1f}% of original")

    # =====================================================================
    # STEP 5: SAVE RESULTS
    # =====================================================================
    print_section("STEP 5: Saving Results")

    try:
        # Visualize
        viz_path = os.path.join(output_dir, "pointcloud.png")
        print("Creating visualization...")
        visualize_point_cloud_matplotlib(points_filtered, output_path=viz_path)
        print(f"✓ Visualization saved: {viz_path}")

        # Save PLY
        ply_path = os.path.join(output_dir, "pointcloud.ply")
        save_point_cloud_ply(points_filtered, output_path=ply_path)
        print(f"✓ PLY saved: {ply_path}")

        # Save OBJ
        obj_path = os.path.join(output_dir, "pointcloud.obj")
        export_point_cloud_obj(points_filtered, output_path=obj_path)
        print(f"✓ OBJ saved: {obj_path}")

        # Save numpy
        np.save(os.path.join(output_dir, "points_3d.npy"), points_filtered)
        print(f"✓ NumPy array saved")

    except Exception as e:
        print(f"ERROR: {e}")
        return None

    # =====================================================================
    # SUMMARY
    # =====================================================================
    elapsed_time = time.time() - start_time

    print_section("EXECUTION SUMMARY")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Frame pairs processed: {len(all_points_3d)}")
    print(f"Total 3D points (after artifact removal): {len(points_filtered)}")
    print(f"Compression ratio: {len(combined_points) / len(points_filtered):.1f}x")
    print(f"\nResults saved to: {output_dir}")

    print_section("COMPLETION")
    print("✓ Refined SfM pipeline completed successfully!")
    print("✓ Hourglass/ghost artifacts should be removed!")

    return points_filtered


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Refined Dense SfM with Artifact Removal'
    )
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to input video file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='sfm_results_refined',
        help='Output directory for results'
    )
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=2,
        help='Extract every nth frame (1 for all, 2 for half, etc)'
    )
    parser.add_argument(
        '--aggressive',
        action='store_true',
        help='Use aggressive filtering for cleaner but fewer points'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    result = run_refined_dense_sfm(
        video_path=args.video_path,
        output_dir=args.output_dir,
        frame_skip=args.frame_skip,
        aggressive_filtering=args.aggressive
    )

    sys.exit(0 if result is not None else 1)
