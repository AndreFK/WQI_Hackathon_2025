"""
Quick test to verify the code structure works.
This tests the basic functionality without requiring all dependencies.
"""

import sys
import numpy as np

# Test if we can import the module
try:
    from pauli_hamiltonian_zx import (
        PauliHamiltonianZX,
        create_collective_decay_hamiltonian,
        compute_F_jk,
        compute_subradiance_eigenvalues
    )
    print("✓ Successfully imported all functions")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease install required packages:")
    print("  pip install pyzx numpy scipy quimb cotengra matplotlib")
    sys.exit(1)

# Test creating Hamiltonian
print("\nTesting Hamiltonian creation...")
try:
    num_atoms = 3
    pauli_strings = create_collective_decay_hamiltonian(
        num_atoms, decay_rate=1.0, detuning=0.0, F_matrix=None
    )
    print(f"✓ Created Hamiltonian with {len(pauli_strings)} Pauli terms")
    print(f"  Number of atoms: {num_atoms}")
    print(f"  Pairs: 01, 02, 12")
    
    # Show first few terms
    print("\n  First few Pauli strings:")
    for i, (coeff, gates) in enumerate(pauli_strings[:4]):
        print(f"    {i+1}: {coeff:.3f} * {gates}")
    
except Exception as e:
    print(f"✗ Error creating Hamiltonian: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test F_jk computation
print("\nTesting F_jk computation...")
try:
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0]
    ])
    F_matrix = compute_F_jk(positions, k0=1.0, decay_rate=1.0)
    print(f"✓ Computed F_jk matrix:")
    print(f"  Shape: {F_matrix.shape}")
    print(f"  F_jk values:")
    for j in range(len(positions)):
        for k in range(j + 1, len(positions)):
            print(f"    F_{j}{k} = {F_matrix[j,k]:.6f}")
    
except Exception as e:
    print(f"✗ Error computing F_jk: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("All basic tests passed!")
print("="*60)
print("\nTo compute eigenvalues, you need to run the full code with:")
print("  python test_subradiance.py")
print("\nOr use the notebook:")
print("  example_usage.ipynb")

