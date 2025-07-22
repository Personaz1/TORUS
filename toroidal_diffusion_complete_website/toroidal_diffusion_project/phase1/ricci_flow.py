"""
ΔΣ::TorusQ - Phase 1: Discrete Ricci Flow
Discrete Ricci flow implementation for torus mesh
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import pandas as pd
from torus_mesh import TorusMesh


class DiscreteRicciFlow:
    """
    Discrete Ricci flow implementation for triangular torus mesh
    
    Implements: dl_ij/dt = -(K_i - K_j)
    With volume normalization to preserve total area
    """
    
    def __init__(self, mesh: TorusMesh, time_step: float = 0.01, 
                 convergence_threshold: float = 1e-5, max_iterations: int = 1000):
        """
        Initialize discrete Ricci flow
        
        Args:
            mesh: TorusMesh object
            time_step: Time step for evolution
            convergence_threshold: Convergence criterion
            max_iterations: Maximum number of iterations
        """
        self.mesh = mesh
        self.time_step = time_step
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        
        # Evolution history
        self.curvature_history = []
        self.edge_length_history = []
        self.area_history = []
        
        # Initial state
        self.initial_total_area = sum(self.mesh.triangle_areas)
        
    def find_vertex_neighbors(self, vertex_idx: int) -> List[int]:
        """Find neighboring vertices for a given vertex"""
        neighbors = set()
        
        for triangle in self.mesh.triangles:
            if vertex_idx in triangle:
                for v in triangle:
                    if v != vertex_idx:
                        neighbors.add(v)
        
        return list(neighbors)
    
    def find_connected_edges(self, vertex_idx: int) -> List[Tuple[int, int]]:
        """Find edges connected to a vertex"""
        connected_edges = []
        
        for edge in self.mesh.edges:
            if vertex_idx in edge:
                connected_edges.append(edge)
        
        return connected_edges
    
    def compute_ricci_flow_step(self) -> Tuple[np.ndarray, Dict, float]:
        """
        Single step of discrete Ricci flow
        
        Returns:
            curvatures: Current curvature values
            new_edge_lengths: Updated edge lengths
            total_area: Current total area
        """
        # Compute current curvatures
        curvatures = self.mesh.compute_discrete_curvature()
        
        # Update edge lengths according to Ricci flow
        new_edge_lengths = {}
        for edge in self.mesh.edges:
            v1, v2 = edge
            current_length = self.mesh.edge_lengths[edge]
            
            # Ricci flow equation: dl_ij/dt = -(K_i - K_j)
            curvature_diff = curvatures[v1] - curvatures[v2]
            length_change = -self.time_step * curvature_diff
            
            new_length = current_length + length_change
            new_length = max(new_length, 1e-6)  # Prevent negative lengths
            new_edge_lengths[edge] = new_length
        
        # Update mesh with new edge lengths
        self._update_mesh_geometry(new_edge_lengths)
        
        # Compute new total area
        total_area = sum(self.mesh.triangle_areas)
        
        return curvatures, new_edge_lengths, total_area
    
    def _update_mesh_geometry(self, new_edge_lengths: Dict):
        """Update mesh geometry based on new edge lengths"""
        # Update edge lengths
        self.mesh.edge_lengths = new_edge_lengths
        
        # Recompute triangle areas using new edge lengths
        self.mesh.triangle_areas = []
        for triangle in self.mesh.triangles:
            a = new_edge_lengths[tuple(sorted([triangle[0], triangle[1]]))]
            b = new_edge_lengths[tuple(sorted([triangle[1], triangle[2]]))]
            c = new_edge_lengths[tuple(sorted([triangle[2], triangle[0]]))]
            
            s = (a + b + c) / 2
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            self.mesh.triangle_areas.append(area)
        
        # Recompute vertex angles
        self.mesh.vertex_angles = self.mesh._compute_vertex_angles()
    
    def normalize_volume(self, target_area: Optional[float] = None):
        """
        Normalize volume (total area) to preserve geometric properties
        
        Args:
            target_area: Target total area (default: initial area)
        """
        if target_area is None:
            target_area = self.initial_total_area
        
        current_area = sum(self.mesh.triangle_areas)
        scale_factor = np.sqrt(target_area / current_area)
        
        # Scale all edge lengths
        new_edge_lengths = {}
        for edge, length in self.mesh.edge_lengths.items():
            new_edge_lengths[edge] = length * scale_factor
        
        # Update mesh geometry
        self._update_mesh_geometry(new_edge_lengths)
        
        print(f"✅ Volume normalized: {current_area:.6f} → {sum(self.mesh.triangle_areas):.6f}")
    
    def evolve_metric(self, normalize_volume: bool = True) -> Dict:
        """
        Evolve metric under discrete Ricci flow until convergence
        
        Args:
            normalize_volume: Whether to normalize volume during evolution
            
        Returns:
            evolution_data: Dictionary containing evolution history
        """
        print("🔄 Starting discrete Ricci flow evolution...")
        
        iteration = 0
        max_curvature_change = float('inf')
        
        while iteration < self.max_iterations and max_curvature_change > self.convergence_threshold:
            # Store current state
            current_curvatures = self.mesh.compute_discrete_curvature()
            current_edge_lengths = self.mesh.edge_lengths.copy()
            current_area = sum(self.mesh.triangle_areas)
            
            # Perform Ricci flow step
            curvatures, new_edge_lengths, total_area = self.compute_ricci_flow_step()
            
            # Normalize volume if requested
            if normalize_volume:
                self.normalize_volume()
            
            # Compute convergence metric
            curvature_change = np.abs(curvatures - current_curvatures)
            max_curvature_change = np.max(curvature_change)
            
            # Store evolution history
            self.curvature_history.append(curvatures.copy())
            self.edge_length_history.append(current_edge_lengths.copy())
            self.area_history.append(current_area)
            
            # Progress report
            if iteration % 50 == 0:
                print(f"  Iteration {iteration}: max ΔK = {max_curvature_change:.8f}, "
                      f"area = {total_area:.6f}")
            
            iteration += 1
        
        # Final state
        final_curvatures = self.mesh.compute_discrete_curvature()
        self.curvature_history.append(final_curvatures.copy())
        self.edge_length_history.append(self.mesh.edge_lengths.copy())
        self.area_history.append(sum(self.mesh.triangle_areas))
        
        print(f"✅ Ricci flow converged after {iteration} iterations")
        print(f"   Final max curvature change: {max_curvature_change:.8f}")
        print(f"   Final total area: {sum(self.mesh.triangle_areas):.6f}")
        
        return {
            'iterations': iteration,
            'converged': max_curvature_change <= self.convergence_threshold,
            'final_max_curvature_change': max_curvature_change,
            'curvature_history': self.curvature_history,
            'edge_length_history': self.edge_length_history,
            'area_history': self.area_history
        }
    
    def detect_solitons(self, curvature_threshold: float = 0.01) -> List[Dict]:
        """
        Detect Ricci solitons (stable curvature patterns)
        
        Args:
            curvature_threshold: Threshold for considering curvature as stable
            
        Returns:
            solitons: List of detected soliton regions
        """
        if not self.curvature_history:
            print("⚠️ No evolution history available. Run evolve_metric() first.")
            return []
        
        final_curvatures = self.curvature_history[-1]
        solitons = []
        
        # Find vertices with low curvature (stable regions)
        stable_vertices = np.where(np.abs(final_curvatures) < curvature_threshold)[0]
        
        if len(stable_vertices) > 0:
            # Group stable vertices into connected components
            soliton_regions = self._find_connected_components(stable_vertices)
            
            for region in soliton_regions:
                if len(region) >= 3:  # Minimum size for soliton
                    soliton = {
                        'vertices': region,
                        'size': len(region),
                        'mean_curvature': np.mean(final_curvatures[region]),
                        'std_curvature': np.std(final_curvatures[region]),
                        'stability_score': self._compute_stability_score(region, final_curvatures)
                    }
                    solitons.append(soliton)
        
        print(f"✅ Detected {len(solitons)} soliton regions")
        return solitons
    
    def _find_connected_components(self, vertices: np.ndarray) -> List[List[int]]:
        """Find connected components among vertices"""
        components = []
        visited = set()
        
        for vertex in vertices:
            if vertex not in visited:
                component = self._dfs_component(vertex, vertices, visited)
                components.append(component)
        
        return components
    
    def _dfs_component(self, start_vertex: int, vertices: np.ndarray, 
                      visited: set) -> List[int]:
        """Depth-first search to find connected component"""
        component = []
        stack = [start_vertex]
        
        while stack:
            vertex = stack.pop()
            if vertex not in visited and vertex in vertices:
                visited.add(vertex)
                component.append(vertex)
                
                # Add neighbors to stack
                neighbors = self.find_vertex_neighbors(vertex)
                for neighbor in neighbors:
                    if neighbor not in visited and neighbor in vertices:
                        stack.append(neighbor)
        
        return component
    
    def _compute_stability_score(self, region: List[int], curvatures: np.ndarray) -> float:
        """Compute stability score for a region"""
        region_curvatures = curvatures[region]
        return 1.0 / (1.0 + np.std(region_curvatures))
    
    def visualize_evolution(self, save_plots: bool = True):
        """Visualize Ricci flow evolution"""
        if not self.curvature_history:
            print("⚠️ No evolution history available. Run evolve_metric() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Curvature evolution
        curvatures_array = np.array(self.curvature_history)
        axes[0, 0].plot(curvatures_array)
        axes[0, 0].set_title('Curvature Evolution')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Curvature')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Max curvature change
        max_changes = []
        for i in range(1, len(self.curvature_history)):
            change = np.max(np.abs(self.curvature_history[i] - self.curvature_history[i-1]))
            max_changes.append(change)
        
        axes[0, 1].semilogy(max_changes)
        axes[0, 1].set_title('Max Curvature Change (log scale)')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Max |ΔK|')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Area evolution
        axes[1, 0].plot(self.area_history)
        axes[1, 0].set_title('Total Area Evolution')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Total Area')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Final curvature distribution
        final_curvatures = self.curvature_history[-1]
        axes[1, 1].hist(final_curvatures, bins=30, alpha=0.7, color='skyblue')
        axes[1, 1].set_title('Final Curvature Distribution')
        axes[1, 1].set_xlabel('Curvature')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('phase1/ricci_flow_evolution.png', dpi=300, bbox_inches='tight')
            print("✅ Evolution plots saved to phase1/ricci_flow_evolution.png")
        
        plt.show()
    
    def save_evolution_data(self, filename: str = "ricci_flow_evolution.json"):
        """Save evolution data to JSON file"""
        evolution_data = {
            'time_step': self.time_step,
            'convergence_threshold': self.convergence_threshold,
            'max_iterations': self.max_iterations,
            'initial_total_area': self.initial_total_area,
            'curvature_history': [curv.tolist() for curv in self.curvature_history],
            'area_history': self.area_history,
            'final_curvatures': self.curvature_history[-1].tolist() if self.curvature_history else []
        }
        
        with open(filename, 'w') as f:
            json.dump(evolution_data, f, indent=2)
        
        print(f"✅ Evolution data saved to {filename}")
    
    def generate_report(self) -> Dict:
        """Generate comprehensive evolution report"""
        if not self.curvature_history:
            return {"error": "No evolution data available"}
        
        initial_curvatures = self.curvature_history[0]
        final_curvatures = self.curvature_history[-1]
        
        # Detect solitons
        solitons = self.detect_solitons()
        
        report = {
            'evolution_summary': {
                'iterations': len(self.curvature_history) - 1,
                'initial_area': self.area_history[0],
                'final_area': self.area_history[-1],
                'area_change_percent': ((self.area_history[-1] - self.area_history[0]) / self.area_history[0]) * 100
            },
            'curvature_analysis': {
                'initial_mean_curvature': np.mean(initial_curvatures),
                'initial_std_curvature': np.std(initial_curvatures),
                'final_mean_curvature': np.mean(final_curvatures),
                'final_std_curvature': np.std(final_curvatures),
                'curvature_improvement': np.std(initial_curvatures) - np.std(final_curvatures)
            },
            'soliton_analysis': {
                'num_solitons': len(solitons),
                'solitons': solitons
            },
            'convergence_metrics': {
                'final_max_curvature_change': np.max(np.abs(final_curvatures - self.curvature_history[-2])) if len(self.curvature_history) > 1 else 0,
                'converged': len(self.curvature_history) - 1 < self.max_iterations
            }
        }
        
        return report


def run_ricci_flow_demo():
    """Demo function for discrete Ricci flow"""
    print("🚀 ΔΣ::TorusQ - Phase 1: Discrete Ricci Flow Demo")
    print("=" * 60)
    
    # Create torus mesh
    mesh = TorusMesh(major_radius=1.0, minor_radius=0.3, resolution=25)
    mesh.create_torus_mesh()
    
    # Save initial mesh
    mesh.save_mesh_data("phase1/initial_mesh.json")
    
    # Initialize Ricci flow
    ricci_flow = DiscreteRicciFlow(
        mesh=mesh,
        time_step=0.01,
        convergence_threshold=1e-5,
        max_iterations=500
    )
    
    # Visualize initial state
    initial_curvatures = mesh.compute_discrete_curvature()
    mesh.visualize_mesh(initial_curvatures, "Initial Torus Mesh")
    
    # Evolve metric
    evolution_data = ricci_flow.evolve_metric(normalize_volume=True)
    
    # Visualize final state
    final_curvatures = mesh.compute_discrete_curvature()
    mesh.visualize_mesh(final_curvatures, "Final Torus Mesh (Ricci Flow)")
    
    # Generate evolution plots
    ricci_flow.visualize_evolution()
    
    # Save evolution data
    ricci_flow.save_evolution_data("phase1/ricci_flow_evolution.json")
    
    # Generate and print report
    report = ricci_flow.generate_report()
    print("\n📊 Ricci Flow Evolution Report:")
    print("=" * 40)
    
    for section, data in report.items():
        print(f"\n{section.upper()}:")
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {data}")
    
    # Save final mesh
    mesh.save_mesh_data("phase1/final_mesh.json")
    
    return mesh, ricci_flow, report


if __name__ == "__main__":
    run_ricci_flow_demo() 