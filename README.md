# Pauli Hamiltonian ZX Calculus

This package implements the method from **"How to Sum and Exponentiate Hamiltonians in ZX Calculus"** by Razin A. Shaikh, Quanlong Wang, and Richie Yeung for summing Pauli string Hamiltonians and computing time evolution using ZX calculus with W-states.

## Features

- **Sum Pauli strings** using ZX calculus with W inputs/outputs
- **Compute time evolution** operators exp(-iHt)
- **Eigenvalue computation** for Hamiltonians
- **State evolution** under Hamiltonians
- **Helper functions** for subradiance collective decay applications

## Installation

Required packages:
```bash
pip install pyzx numpy scipy quimb cotengra matplotlib
```

## Quick Start

### Basic Usage

```python
from pauli_hamiltonian_zx import PauliHamiltonianZX

# Define a Pauli string Hamiltonian
pauli_strings = [
    (3.0, ["X0", "X1"]),
    (1.0, ["X1", "X2"]),
    (-1.0, ["Z0"]),
    (-1.0, ["Z1"]),
    (-1.0, ["Z2"])
]

# Initialize and compute
hamiltonian = PauliHamiltonianZX(pauli_strings)
eigenvalues = hamiltonian.compute_eigenvalues()
print(f"Eigenvalues: {eigenvalues}")
```

### Time Evolution

```python
# Compute time evolution operator
U = hamiltonian.time_evolution(time=1.0)

# Evolve an initial state
initial_state = np.zeros(2**hamiltonian.total_qubits)
initial_state[0] = 1.0  # |00...0⟩
evolved_state = hamiltonian.evolve_state(initial_state, time=1.0)
```

### Subradiance Collective Decay (Equation 4)

The Hamiltonian from equation 4 is:
**H = Σ_{j<k} (F_jk/2) * (X_j X_k + Y_j Y_k)**

where indices run over pairs: 01, 02, 03, 12, 13, 23, etc.

```python
from pauli_hamiltonian_zx import (
    compute_subradiance_eigenvalues,
    compute_F_jk,
    create_collective_decay_hamiltonian
)

# Method 1: Simple case with uniform F_jk
num_atoms = 3
eigenvalues = compute_subradiance_eigenvalues(
    num_atoms=num_atoms,
    decay_rate=1.0,
    detuning=0.0
)

# Method 2: With custom F_jk matrix (from paper)
F_matrix = np.array([
    [1.0, 0.5, 0.3],
    [0.5, 1.0, 0.4],
    [0.3, 0.4, 1.0]
])
eigenvalues = compute_subradiance_eigenvalues(
    num_atoms=num_atoms,
    F_matrix=F_matrix,
    decay_rate=1.0
)

# Method 3: Compute F_jk from atom positions
positions = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [2.0, 0.0, 0.0]
])
eigenvalues = compute_subradiance_eigenvalues(
    num_atoms=num_atoms,
    positions=positions,
    k0=1.0,
    decay_rate=1.0
)

# Method 4: Compare ZX calculus vs numpy approach
from pauli_hamiltonian_zx import compare_eigenvalue_methods

eig_zx, eig_numpy, match, max_diff = compare_eigenvalue_methods(
    num_atoms=3,
    decay_rate=1.0,
    detuning=0.0
)
print(f"Methods match: {match}, Max difference: {max_diff:.2e}")
```

## API Reference

### `PauliHamiltonianZX`

Main class for working with Pauli string Hamiltonians.

#### Methods

- `build_graph()`: Build the ZX graph representation
- `simplify_graph()`: Simplify using ZX calculus rules
- `compute_matrix(optimize=True)`: Get full matrix representation
- `compute_eigenvalues(hermitian=True)`: Compute eigenvalues
- `time_evolution(time, optimize=True)`: Compute exp(-iHt)
- `evolve_state(initial_state, time)`: Evolve a quantum state
- `expectation_value(state)`: Compute <ψ|H|ψ>

### Helper Functions

- `generate_random_pauli_string(num_terms, num_qubits, ...)`: Generate random Pauli strings
- `create_collective_decay_hamiltonian(num_atoms, decay_rate, detuning, F_matrix)`: Create subradiance Hamiltonian from equation 4
- `compute_F_jk(positions, k0, decay_rate)`: Compute F_jk coupling coefficients from atom positions
- `compute_subradiance_eigenvalues(...)`: Compute eigenvalues of the subradiance Hamiltonian (ZX calculus or numpy)
- `compute_subradiance_eigenvalues_numpy(...)`: Compute eigenvalues using standard numpy approach
- `compute_subradiance_hamiltonian_numpy(...)`: Construct Hamiltonian matrix directly using numpy
- `compare_eigenvalue_methods(...)`: Compare ZX calculus vs numpy eigenvalue results
- `compute_subradiance_decay(...)`: Compute decay dynamics

## Examples

See `example_usage.ipynb` for detailed examples.

## References

1. **"How to Sum and Exponentiate Hamiltonians in ZX Calculus"**
   - Razin A. Shaikh, Quanlong Wang, Richie Yeung
   - arXiv:2212.04462

2. **"Analytical and numerical study of subradiance-only collective decay from atomic ensembles"**
   - Anirudh Yadav and D. D. Yavuz

## Notes

- For large systems (>10 qubits), tensor network contraction can be memory-intensive
- The code automatically optimizes tensor network contraction using cotengra
- Hamiltonians are automatically symmetrized to be Hermitian when computing eigenvalues

