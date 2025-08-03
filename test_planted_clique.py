import numpy as np
import sys
import os

# Add the theoretical_analysis directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'theoretical_analysis'))

from planted_submatrix_analysis import generate_planted_clique

def test_planted_clique():
    """Test the generate_planted_clique function with a small example."""
    
    # Small example parameters
    n = 10  # matrix size
    beta = 0.5  # parameter controlling clique size
    
    print(f"Testing generate_planted_clique with n={n}, beta={beta}")
    print("=" * 50)
    
    # Generate the planted clique
    A_p, y = generate_planted_clique(n, beta)
    
    # Calculate clique size
    k = int(np.sqrt(n) * beta)
    print(f"Matrix size (n): {n}")
    print(f"Beta parameter: {beta}")
    print(f"Clique size (k = sqrt(n) * beta): {k}")
    print(f"Matrix shape: {A_p.shape}")
    print(f"Vector y shape: {y.shape}")
    
    print("\nMatrix A_p:")
    print(A_p)
    
    print(f"\nVector y (normalized):")
    print(y)
    
    print(f"\nClique vertices (where y[i] > 0):")
    clique_vertices = np.where(y > 0)[0]
    print(clique_vertices)
    
    print(f"\nSubmatrix of clique vertices:")
    clique_submatrix = A_p[np.ix_(clique_vertices, clique_vertices)]
    print(clique_submatrix)
    
    print(f"\nMatrix properties:")
    print(f"  - Symmetric: {np.allclose(A_p, A_p.T)}")
    print(f"  - Diagonal elements: {np.diag(A_p)}")
    print(f"  - Matrix norm: {np.linalg.norm(A_p)}")
    print(f"  - Vector y norm: {np.linalg.norm(y)}")
    
    # Check that the clique submatrix has all 1s (except diagonal)
    expected_clique_values = np.ones((k, k)) / np.sqrt(n)
    np.fill_diagonal(expected_clique_values, 0)  # Diagonal should be different
    print(f"\nClique submatrix should have values ≈ 1/sqrt(n) = {1/np.sqrt(n):.4f}")
    print(f"  - Clique values close to expected: {np.allclose(clique_submatrix, expected_clique_values, atol=1e-10)}")

if __name__ == "__main__":
    test_planted_clique() 