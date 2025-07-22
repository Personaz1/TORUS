"""
ΔΣ::TorusQ - Phase 1: Triangular Torus Mesh
Discrete torus mesh generation with UV parametrization
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from typing import Dict, List, Tuple, Optional
import pandas as pd


class TorusMesh:
    """
    Triangular torus mesh with UV parametrization
    T² = S¹ × S¹ with discrete triangulation
    """
    
    def __init__(self, major_radius: float = 1.0, minor_radius: float = 0.3, 
                 resolution: int = 50):
        """
        Initialize torus mesh
        
        Args:
            major_radius: Major radius of torus (R)
            minor_radius: Minor radius of torus (r)
            resolution: Number of points per dimension for triangulation
        """
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.resolution = resolution
        
        # Mesh data
        self.vertices = None
        self.triangles = None
        self.edges = None
        self.uv_coords = None
        
        # Geometric properties
        self.edge_lengths = None
        self.triangle_areas = None
        self.vertex_angles = None
        
    def create_torus_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create triangulated torus mesh using UV parametrization
        
        Returns:
            vertices: 3D coordinates of mesh vertices
            triangles: Triangle indices
        """
        # Generate UV grid
        theta = np.linspace(0, 2*np.pi, self.resolution, endpoint=False)
        phi = np.linspace(0, 2*np.pi, self.resolution, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        
        # Torus parametrization: (R + r*cos(φ))*cos(θ), (R + r*cos(φ))*sin(θ), r*sin(φ)
        x = (self.major_radius + self.minor_radius * np.cos(phi_grid)) * np.cos(theta_grid)
        y = (self.major_radius + self.minor_radius * np.cos(phi_grid)) * np.sin(theta_grid)
        z = self.minor_radius * np.sin(phi_grid)
        
        # Flatten coordinates for triangulation
        points_2d = np.column_stack([theta_grid.flatten(), phi_grid.flatten()])
        points_3d = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
        
        # Create triangulation (using 2D projection)
        tri = Delaunay(points_2d)
        
        # Store mesh data
        self.vertices = points_3d
        self.triangles = tri.simplices
        self.uv_coords = points_2d
        
        # Compute mesh properties
        self._compute_mesh_properties()
        
        print(f"✅ Torus mesh created: {len(self.vertices)} vertices, {len(self.triangles)} triangles")
        return self.vertices, self.triangles
    
    def _compute_mesh_properties(self):
        """Compute edge lengths, triangle areas, and vertex angles"""
        # Compute edge lengths
        self.edge_lengths = {}
        for triangle in self.triangles:
            for i in range(3):
                v1, v2 = triangle[i], triangle[(i+1)%3]
                edge = tuple(sorted([v1, v2]))
                if edge not in self.edge_lengths:
                    length = np.linalg.norm(self.vertices[v1] - self.vertices[v2])
                    self.edge_lengths[edge] = length
        
        # Compute triangle areas using Heron's formula
        self.triangle_areas = []
        for triangle in self.triangles:
            a = np.linalg.norm(self.vertices[triangle[1]] - self.vertices[triangle[0]])
            b = np.linalg.norm(self.vertices[triangle[2]] - self.vertices[triangle[1]])
            c = np.linalg.norm(self.vertices[triangle[0]] - self.vertices[triangle[2]])
            s = (a + b + c) / 2
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            self.triangle_areas.append(area)
        
        # Compute vertex angles
        self.vertex_angles = self._compute_vertex_angles()
        
        # Extract edges list
        self.edges = list(self.edge_lengths.keys())
    
    def _compute_vertex_angles(self) -> Dict[int, List[float]]:
        """Compute angles at each vertex"""
        vertex_angles = {i: [] for i in range(len(self.vertices))}
        
        for triangle_idx, triangle in enumerate(self.triangles):
            for i in range(3):
                v1, v2, v3 = triangle[i], triangle[(i+1)%3], triangle[(i+2)%3]
                
                # Compute angle at vertex v1
                vec1 = self.vertices[v2] - self.vertices[v1]
                vec2 = self.vertices[v3] - self.vertices[v1]
                
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
                angle = np.arccos(cos_angle)
                
                vertex_angles[v1].append(angle)
        
        return vertex_angles
    
    def compute_discrete_curvature(self) -> np.ndarray:
        """
        Compute discrete curvature at each vertex
        
        K_i = 2π - Σ(angles at vertex i)
        
        Returns:
            curvatures: Array of curvature values at each vertex
        """
        curvatures = np.zeros(len(self.vertices))
        
        for vertex_idx in range(len(self.vertices)):
            angle_sum = sum(self.vertex_angles[vertex_idx])
            curvatures[vertex_idx] = 2 * np.pi - angle_sum
        
        return curvatures
    
    def save_mesh_data(self, filename: str = "mesh.json"):
        """Save mesh data to JSON file"""
        mesh_data = {
            'major_radius': self.major_radius,
            'minor_radius': self.minor_radius,
            'resolution': self.resolution,
            'vertices': self.vertices.tolist(),
            'triangles': self.triangles.tolist(),
            'edges': self.edges,
            'edge_lengths': {str(k): v for k, v in self.edge_lengths.items()},
            'triangle_areas': self.triangle_areas,
            'uv_coords': self.uv_coords.tolist()
        }
        
        with open(filename, 'w') as f:
            json.dump(mesh_data, f, indent=2)
        
        print(f"✅ Mesh data saved to {filename}")
    
    def load_mesh_data(self, filename: str = "mesh.json"):
        """Load mesh data from JSON file"""
        with open(filename, 'r') as f:
            mesh_data = json.load(f)
        
        self.major_radius = mesh_data['major_radius']
        self.minor_radius = mesh_data['minor_radius']
        self.resolution = mesh_data['resolution']
        self.vertices = np.array(mesh_data['vertices'])
        self.triangles = np.array(mesh_data['triangles'])
        self.edges = mesh_data['edges']
        self.edge_lengths = {eval(k): v for k, v in mesh_data['edge_lengths'].items()}
        self.triangle_areas = mesh_data['triangle_areas']
        self.uv_coords = np.array(mesh_data['uv_coords'])
        
        # Recompute derived properties
        self.vertex_angles = self._compute_vertex_angles()
        
        print(f"✅ Mesh data loaded from {filename}")
    
    def visualize_mesh(self, curvatures: Optional[np.ndarray] = None, 
                      title: str = "Torus Mesh"):
        """Visualize torus mesh with optional curvature coloring"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot triangles
        for triangle in self.triangles:
            vertices = self.vertices[triangle]
            ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                           alpha=0.3, color='lightblue')
        
        # Plot vertices with curvature coloring
        if curvatures is not None:
            scatter = ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2],
                               c=curvatures, cmap='viridis', s=20)
            plt.colorbar(scatter, ax=ax, label='Curvature')
        else:
            ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2],
                      c='red', s=20)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        plt.tight_layout()
        plt.show()
    
    def get_mesh_statistics(self) -> Dict:
        """Get mesh statistics"""
        curvatures = self.compute_discrete_curvature()
        
        stats = {
            'num_vertices': len(self.vertices),
            'num_triangles': len(self.triangles),
            'num_edges': len(self.edges),
            'total_area': sum(self.triangle_areas),
            'mean_curvature': np.mean(curvatures),
            'std_curvature': np.std(curvatures),
            'min_curvature': np.min(curvatures),
            'max_curvature': np.max(curvatures),
            'mean_edge_length': np.mean(list(self.edge_lengths.values())),
            'std_edge_length': np.std(list(self.edge_lengths.values()))
        }
        
        return stats


def create_torus_mesh_demo():
    """Demo function for torus mesh creation"""
    print("🚀 ΔΣ::TorusQ - Phase 1: Torus Mesh Demo")
    print("=" * 50)
    
    # Create torus mesh
    mesh = TorusMesh(major_radius=1.0, minor_radius=0.3, resolution=30)
    vertices, triangles = mesh.create_torus_mesh()
    
    # Compute curvature
    curvatures = mesh.compute_discrete_curvature()
    
    # Get statistics
    stats = mesh.get_mesh_statistics()
    print("\n📊 Mesh Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")
    
    # Save mesh data
    mesh.save_mesh_data("phase1/mesh.json")
    
    # Visualize mesh
    mesh.visualize_mesh(curvatures, "Initial Torus Mesh with Curvature")
    
    return mesh, curvatures


if __name__ == "__main__":
    create_torus_mesh_demo() 