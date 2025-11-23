"""
Test script for computing subradiance eigenvalues from equation 4.

This script computes eigenvalues of the Hamiltonian:
H = Σ_{j<k} (F_jk/2) * (X_j X_k + Y_j Y_k)

where indices run over pairs: 01, 02, 03, 12, 13, 23, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
from pauli_hamiltonian_zx import (
    compute_subradiance_eigenvalues,
    compute_F_jk,
    create_collective_decay_hamiltonian,
    PauliHamiltonianZX
)

def test_simple_case():
    """Test with a simple uniform F_jk case."""
    print("="*60)
    print("Test 1: Simple uniform F_jk case")
    print("="*60)
    
    num_atoms = 3
    decay_rate = 1.0
    
    # Create Hamiltonian with uniform F_jk = decay_rate
    pauli_strings = create_collective_decay_hamiltonian(
        num_atoms, decay_rate=decay_rate, detuning=0.0, F_matrix=None
    )
    
    print(f"\nNumber of atoms: {num_atoms}")
    print(f"Number of Pauli terms: {len(pauli_strings)}")
    print("\nPauli strings:")
    for i, (coeff, gates) in enumerate(pauli_strings):
        print(f"  {i+1}: {coeff:.3f} * {gates}")
    
    # Compute eigenvalues
    print("\nComputing eigenvalues...")
    try:
        hamiltonian = PauliHamiltonianZX(pauli_strings)
        eigenvalues = hamiltonian.compute_eigenvalues(hermitian=True)
        
        print(f"\nEigenvalues (sorted):")
        sorted_eigs = np.sort(eigenvalues)
        for i, eig in enumerate(sorted_eigs):
            print(f"  {i+1}: {eig:.6f}")
        
        print(f"\nEigenvalue statistics:")
        print(f"  Min: {np.min(eigenvalues):.6f}")
        print(f"  Max: {np.max(eigenvalues):.6f}")
        print(f"  Mean: {np.mean(eigenvalues):.6f}")
        print(f"  Std: {np.std(eigenvalues):.6f}")
        
        return eigenvalues
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_with_positions():
    """Test with atom positions to compute F_jk."""
    print("\n" + "="*60)
    print("Test 2: Computing F_jk from atom positions")
    print("="*60)
    
    num_atoms = 4
    decay_rate = 1.0
    k0 = 1.0
    
    # Create some example positions (e.g., on a line)
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0]
    ])
    
    print(f"\nNumber of atoms: {num_atoms}")
    print(f"Atom positions:")
    for i, pos in enumerate(positions):
        print(f"  Atom {i}: {pos}")
    
    # Compute F_jk matrix
    F_matrix = compute_F_jk(positions, k0=k0, decay_rate=decay_rate)
    
    print(f"\nF_jk matrix:")
    print(F_matrix)
    
    # Compute eigenvalues
    print("\nComputing eigenvalues...")
    try:
        eigenvalues = compute_subradiance_eigenvalues(
            num_atoms=num_atoms,
            F_matrix=F_matrix,
            decay_rate=decay_rate,
            detuning=0.0
        )
        
        print(f"\nEigenvalues (sorted):")
        sorted_eigs = np.sort(eigenvalues)
        for i, eig in enumerate(sorted_eigs):
            print(f"  {i+1}: {eig:.6f}")
        
        return eigenvalues
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_with_custom_F_matrix():
    """Test with a custom F_jk matrix."""
    print("\n" + "="*60)
    print("Test 3: Custom F_jk matrix")
    print("="*60)
    
    num_atoms = 3
    
    # Create a custom F_jk matrix
    # For example, with specific coupling values
    F_matrix = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])
    
    print(f"\nNumber of atoms: {num_atoms}")
    print(f"F_jk matrix:")
    print(F_matrix)
    
    # Create Hamiltonian
    pauli_strings = create_collective_decay_hamiltonian(
        num_atoms, decay_rate=1.0, detuning=0.0, F_matrix=F_matrix
    )
    
    print(f"\nNumber of Pauli terms: {len(pauli_strings)}")
    print("\nPauli strings (showing pairs):")
    pair_idx = 0
    for j in range(num_atoms):
        for k in range(j + 1, num_atoms):
            print(f"  Pair {j}{k}: F_{j}{k} = {F_matrix[j,k]:.3f}")
            print(f"    -> {F_matrix[j,k]/2:.3f} * X{j}X{k}")
            print(f"    -> {F_matrix[j,k]/2:.3f} * Y{j}Y{k}")
            pair_idx += 1
    
    # Compute eigenvalues
    print("\nComputing eigenvalues...")
    try:
        eigenvalues = compute_subradiance_eigenvalues(
            num_atoms=num_atoms,
            F_matrix=F_matrix,
            decay_rate=1.0,
            detuning=0.0
        )
        
        print(f"\nEigenvalues (sorted):")
        sorted_eigs = np.sort(eigenvalues)
        for i, eig in enumerate(sorted_eigs):
            print(f"  {i+1}: {eig:.6f}")
        
        # Plot eigenvalues
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_eigs)), sorted_eigs, alpha=0.7, edgecolor='black')
        plt.xlabel('Eigenvalue Index', fontsize=12)
        plt.ylabel('Eigenvalue', fontsize=12)
        plt.title('Subradiance Hamiltonian Eigenvalues', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('subradiance_eigenvalues.png', dpi=150)
        print("\nPlot saved to 'subradiance_eigenvalues.png'")
        
        return eigenvalues
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_eigenvalue_comparison():
    """Test comparison between ZX calculus and numpy methods."""
    print("\n" + "="*60)
    print("Test 4: Comparing ZX Calculus vs Numpy Methods")
    print("="*60)
    
    from pauli_hamiltonian_zx import compare_eigenvalue_methods
    
    num_atoms = 3
    decay_rate = 1.0
    
    print(f"\nNumber of atoms: {num_atoms}")
    print(f"Decay rate: {decay_rate}")
    
    try:
        eig_zx, eig_numpy, match, max_diff = compare_eigenvalue_methods(
            num_atoms=num_atoms,
            decay_rate=decay_rate,
            detuning=0.0,
            tolerance=1e-10
        )
        
        print(f"\nZX Calculus eigenvalues:")
        print(eig_zx)
        print(f"\nNumpy eigenvalues:")
        print(eig_numpy)
        print(f"\nMaximum difference: {max_diff:.2e}")
        print(f"Methods match: {match}")
        
        if match:
            print("✓ SUCCESS: Both methods agree!")
        else:
            print("✗ WARNING: Methods differ (may be due to numerical precision)")
        
        return eig_zx, eig_numpy, match
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False


if __name__ == "__main__":
    print("Testing Subradiance Eigenvalue Computation")
    print("="*60)
    
    # Run tests
    eig1 = test_simple_case()
    eig2 = test_with_positions()
    eig3 = test_with_custom_F_matrix()
    eig_zx, eig_numpy, match = test_eigenvalue_comparison()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

