# Usage Guide for Subradiance Eigenvalue Computation

This guide explains how to use the code to compute eigenvalues from equation 4 of the subradiance paper.

## Equation 4

The Hamiltonian from equation 4 is:
```
H = Σ_{j<k} (F_jk/2) * (X_j X_k + Y_j Y_k)
```

where:
- Indices `j` and `k` run over pairs: 01, 02, 03, 12, 13, 23, etc.
- `F_jk` are coupling coefficients (from PhysRevA.110.023709.pdf)
- The other part of the Hamiltonian is already diagonal, so we only need eigenvalues of this part

## Getting F_jk Values

F_jk values can be obtained from:
1. **PhysRevA.110.023709.pdf** - Theoretical values
2. **Atom positions** - Using dipole-dipole interaction formula
3. **Custom values** - If you have specific coupling values

## Usage Examples

### Example 1: Simple Case (Uniform F_jk)

```python
from pauli_hamiltonian_zx import compute_subradiance_eigenvalues
import numpy as np

num_atoms = 3
eigenvalues = compute_subradiance_eigenvalues(
    num_atoms=num_atoms,
    decay_rate=1.0,  # Uniform F_jk = 1.0 for all pairs
    detuning=0.0
)

print("Eigenvalues:", eigenvalues)
```

This creates pairs: 01, 02, 12 (for 3 atoms)

### Example 2: Custom F_jk Matrix

```python
from pauli_hamiltonian_zx import compute_subradiance_eigenvalues
import numpy as np

num_atoms = 4
# Provide F_jk values from the paper
F_matrix = np.array([
    [1.0, 0.5, 0.3, 0.2],
    [0.5, 1.0, 0.4, 0.25],
    [0.3, 0.4, 1.0, 0.3],
    [0.2, 0.25, 0.3, 1.0]
])

eigenvalues = compute_subradiance_eigenvalues(
    num_atoms=num_atoms,
    F_matrix=F_matrix,
    decay_rate=1.0,
    detuning=0.0
)

print("Eigenvalues:", eigenvalues)
```

This creates pairs: 01, 02, 03, 12, 13, 23 (for 4 atoms)

### Example 3: Compute F_jk from Atom Positions

```python
from pauli_hamiltonian_zx import (
    compute_subradiance_eigenvalues,
    compute_F_jk
)
import numpy as np

num_atoms = 4
decay_rate = 1.0
k0 = 1.0  # Wavevector magnitude

# Atom positions (e.g., on a line)
positions = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [3.0, 0.0, 0.0]
])

# Compute F_jk from positions
F_matrix = compute_F_jk(positions, k0=k0, decay_rate=decay_rate)
print("F_jk matrix:", F_matrix)

# Compute eigenvalues
eigenvalues = compute_subradiance_eigenvalues(
    num_atoms=num_atoms,
    positions=positions,
    k0=k0,
    decay_rate=decay_rate,
    detuning=0.0
)

print("Eigenvalues:", eigenvalues)
```

## Understanding the Pairs

For `num_atoms = N`, the pairs are:
- 01, 02, 03, ..., 0(N-1)
- 12, 13, ..., 1(N-1)
- ...
- (N-2)(N-1)

Total number of pairs: N*(N-1)/2

Each pair contributes two Pauli terms:
- (F_jk/2) * X_j X_k
- (F_jk/2) * Y_j Y_k

## Running the Code

1. **Install dependencies:**
   ```bash
   pip install pyzx numpy scipy quimb cotengra matplotlib
   ```

2. **Test the code:**
   ```bash
   python quick_test.py
   ```

3. **Run full test:**
   ```bash
   python test_subradiance.py
   ```

4. **Use the notebook:**
   Open `example_usage.ipynb` and run the cells

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, install missing packages:
```bash
pip install pyzx numpy scipy quimb cotengra matplotlib
```

### Memory Issues

For large systems (>10 qubits), the tensor network contraction can be memory-intensive. Try:
- Reducing the number of atoms
- Using smaller F_jk values
- Simplifying the Hamiltonian

### Slow Computation

The tensor network contraction can be slow for large systems. The code uses cotengra optimization, but for very large systems, consider:
- Using approximate methods
- Reducing system size
- Using sparse matrix methods

## Next Steps

1. Extract F_jk values from PhysRevA.110.023709.pdf
2. Use those values in `compute_subradiance_eigenvalues()`
3. Compare eigenvalues with plots from the paper
4. Reproduce the subradiance plots

