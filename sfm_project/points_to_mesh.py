#!/usr/bin/env python3
"""
POINT CLOUD TO MESH CONVERSION
Converts sparse 3D point cloud to mesh surface
Provides much better visualization than point clouds
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path

# Try to import optional libraries
try:
    from scipy.spatial import Delaunay, ConvexHull
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available. Install with: pip install scipy")


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def load_point_cloud(npy_file):
    """Load point cloud from numpy file."""
    if not os.path.exists(npy_file):
        print(f"ERROR: File not found: {npy_file}")
        return None
    
    points = np.load(npy_file)
    print(f"✓ Loaded {len(points)} points from {npy_file}")
    
    return points


def filter_point_cloud_by_density(points, grid_size=10.0, min_points=3):
    """
    Downsample point cloud to improve mesh quality.
    Keeps one representative point per grid cell.
    """
    print(f"Downsampling by density (grid_size={grid_size})...")
    
    # Create grid
    grid_coords = np.floor(points / grid_size).astype(int)
    
    # Keep one point per grid cell
    unique_cells, indices = np.unique(grid_coords, return_index=True, axis=0)
    
    filtered_points = points[indices]
    
    print(f"✓ Downsampled from {len(points)} to {len(filtered_points)} points")
    
    return filtered_points


def create_mesh_delaunay(points, max_points=50000):
    """
    Create mesh using Delaunay triangulation.
    Best for dense, well-distributed point clouds.
    """
    if len(points) > max_points:
        print(f"WARNING: Point cloud too large ({len(points)} > {max_points})")
        print(f"Downsampling to {max_points} points for Delaunay...")
        
        # Random sampling
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    
    print(f"Computing Delaunay triangulation for {len(points)} points...")
    print("(This may take a minute...)")
    
    try:
        delaunay = Delaunay(points)
        print(f"✓ Created {len(delaunay.simplices)} tetrahedra")
        
        # Extract surface triangles (faces on convex hull)
        hull = ConvexHull(points)
        print(f"✓ Extracted {len(hull.simplices)} surface triangles")
        
        return points, hull.simplices
    
    except Exception as e:
        print(f"ERROR: {e}")
        return None, None


def create_mesh_surface_reconstruction(points, max_points=100000):
    """
    Create mesh using simple surface reconstruction.
    Works better for sparse point clouds.
    """
    if len(points) > max_points:
        print(f"Downsampling to {max_points} points...")
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    
    print(f"Creating surface mesh from {len(points)} points...")
    
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        print(f"✓ Created {len(hull.simplices)} triangles")
        return points, hull.simplices
    
    except Exception as e:
        print(f"ERROR: {e}")
        return None, None


def export_mesh_ply(points, triangles, output_path):
    """Export mesh to PLY format."""
    print(f"Writing PLY file with {len(points)} vertices and {len(triangles)} faces...")
    
    with open(output_path, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(triangles)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Vertices
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
        
        # Faces
        for triangle in triangles:
            f.write(f"3 {triangle[0]} {triangle[1]} {triangle[2]}\n")
    
    print(f"✓ PLY saved: {output_path}")


def export_mesh_obj(points, triangles, output_path):
    """Export mesh to OBJ format."""
    print(f"Writing OBJ file...")
    
    with open(output_path, 'w') as f:
        f.write("# Mesh from point cloud\n")
        f.write(f"# {len(points)} vertices, {len(triangles)} faces\n\n")
        
        # Vertices
        for point in points:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")
        
        # Faces (OBJ is 1-indexed)
        for triangle in triangles:
            v1, v2, v3 = triangle + 1
            f.write(f"f {v1} {v2} {v3}\n")
    
    print(f"✓ OBJ saved: {output_path}")


def export_mesh_stl(points, triangles, output_path):
    """Export mesh to STL format (binary)."""
    print(f"Writing STL file...")
    
    import struct
    
    with open(output_path, 'wb') as f:
        # STL header
        header = b'Mesh from point cloud' + b' ' * (80 - 20)
        f.write(header)
        
        # Number of triangles
        f.write(struct.pack('<I', len(triangles)))
        
        # Triangles
        for triangle in triangles:
            v1 = points[triangle[0]]
            v2 = points[triangle[1]]
            v3 = points[triangle[2]]
            
            # Calculate normal
            normal = np.cross(v2 - v1, v3 - v1)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal = normal / norm_len
            else:
                normal = np.array([0, 0, 1])
            
            # Write normal
            f.write(struct.pack('<fff', *normal))
            
            # Write vertices
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            f.write(struct.pack('<fff', *v3))
            
            # Attribute byte count
            f.write(struct.pack('<H', 0))
    
    print(f"✓ STL saved: {output_path}")


def run_mesh_conversion(input_file, output_dir="mesh_results", 
                       method="convex_hull", downsample=True, grid_size=10.0):
    """
    Convert point cloud to mesh.
    
    Args:
        input_file: Path to .npy point cloud file
        output_dir: Output directory
        method: "convex_hull" (fast, simple) or "delaunay" (complex, slower)
        downsample: Whether to downsample point cloud
        grid_size: Grid size for downsampling
    """
    
    if not SCIPY_AVAILABLE:
        print("ERROR: scipy required. Install with: pip install scipy")
        return None
    
    print_section("POINT CLOUD TO MESH CONVERSION")
    print(f"Input: {input_file}")
    print(f"Method: {method}")
    print(f"Downsample: {downsample}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # =====================================================================
    # STEP 1: LOAD POINT CLOUD
    # =====================================================================
    print_section("STEP 1: Load Point Cloud")
    
    points = load_point_cloud(input_file)
    
    if points is None:
        return None
    
    # =====================================================================
    # STEP 2: OPTIONAL DOWNSAMPLING
    # =====================================================================
    if downsample:
        print_section("STEP 2: Downsample Point Cloud")
        points = filter_point_cloud_by_density(points, grid_size=grid_size)
    
    # =====================================================================
    # STEP 3: CREATE MESH
    # =====================================================================
    print_section("STEP 3: Create Mesh")
    
    if method == "convex_hull":
        print("Creating convex hull mesh...")
        points_mesh, triangles = create_mesh_surface_reconstruction(points)
    elif method == "delaunay":
        print("Creating Delaunay mesh...")
        points_mesh, triangles = create_mesh_delaunay(points)
    else:
        print(f"ERROR: Unknown method {method}")
        return None
    
    if triangles is None:
        print("ERROR: Mesh creation failed")
        return None
    
    # =====================================================================
    # STEP 4: SAVE RESULTS
    # =====================================================================
    print_section("STEP 4: Save Mesh")
    
    # Save PLY (best for MeshLab)
    ply_path = os.path.join(output_dir, "mesh.ply")
    export_mesh_ply(points_mesh, triangles, ply_path)
    
    # Save OBJ
    obj_path = os.path.join(output_dir, "mesh.obj")
    export_mesh_obj(points_mesh, triangles, obj_path)
    
    # Save STL
    stl_path = os.path.join(output_dir, "mesh.stl")
    export_mesh_stl(points_mesh, triangles, stl_path)
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    print_section("COMPLETION")
    print(f"✓ Mesh created successfully!")
    print(f"✓ Vertices: {len(points_mesh)}")
    print(f"✓ Faces: {len(triangles)}")
    print(f"\nOutput files:")
    print(f"  - {ply_path} (recommended for MeshLab)")
    print(f"  - {obj_path} (for other viewers)")
    print(f"  - {stl_path} (for 3D printing)")
    print(f"\nOpen {ply_path} in MeshLab for best results!")
    
    return points_mesh, triangles


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert Point Cloud to Mesh'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to .npy point cloud file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='mesh_results',
        help='Output directory for mesh files'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['convex_hull', 'delaunay'],
        default='convex_hull',
        help='Mesh creation method (default: convex_hull)'
    )
    parser.add_argument(
        '--no-downsample',
        action='store_true',
        help='Skip downsampling step'
    )
    parser.add_argument(
        '--grid-size',
        type=float,
        default=10.0,
        help='Grid size for downsampling (default: 10.0)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    result = run_mesh_conversion(
        input_file=args.input_file,
        output_dir=args.output_dir,
        method=args.method,
        downsample=not args.no_downsample,
        grid_size=args.grid_size
    )
    
    sys.exit(0 if result is not None else 1)
