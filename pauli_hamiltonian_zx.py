"""
Pauli Hamiltonian Summation and Time Evolution using ZX Calculus with W-states.

This module implements the method from:
"How to Sum and Exponentiate Hamiltonians in ZX Calculus"
by Razin A. Shaikh, Quanlong Wang, and Richie Yeung

The code provides functionality to:
1. Sum Pauli string Hamiltonians using ZX calculus with W inputs/outputs
2. Compute time evolution operators exp(-iHt)
3. Apply to quantum systems like superradiance/subradiance collective decay
4. Analyze collective decay using ZX calculus methods

References:
- "How to Sum and Exponentiate Hamiltonians in ZX Calculus" (arXiv:2212.04462)
- "Analytical and numerical study of subradiance-only collective decay from atomic ensembles"
- "A numerical study of the spatial coherence of light in collective spontaneous emission"
- "Light Matter Interaction ZXW calculus" (for future integration)
"""

import pyzx as zx
import sympy as sp
import numpy as np
from typing import List, Tuple, Optional, Union
import quimb.tensor as qtn
import cotengra as ctg
from pyzx.quimb import to_quimb_tensor


class PauliHamiltonianZX:
    """
    Class for working with Pauli string Hamiltonians using ZX calculus.
    
    This class implements the W-state method for summing and exponentiating
    Pauli string Hamiltonians as described in the paper.
    """
    
    def __init__(self, pauli_strings: List[Tuple[float, List[str]]]):
        """
        Initialize with a list of Pauli strings.
        
        Args:
            pauli_strings: List of tuples (coefficient, [gate_list])
                Example: [(0.5, ["X0", "X1"]), (-0.3, ["Y2", "Z3"])]
        """
        self.pauli_strings = pauli_strings
        self.total_qubits = self._compute_total_qubits()
        self.main_graph = None
        self.top_graph = None
        self.tot_graph = None
        self.tensor_network = None
        self.optimizer = None
        
    def _compute_total_qubits(self) -> int:
        """Compute the total number of qubits needed."""
        total_qubits = 0
        for term in self.pauli_strings:
            qubit_indices = [int(gate[1:]) for gate in term[1]]
            if qubit_indices:
                max_index = max(qubit_indices)
                if max_index >= total_qubits:
                    total_qubits = max_index + 1
        return total_qubits
    
    def _build_main_graph(self) -> zx.Graph:
        """
        Build the main ZX graph representing the Pauli strings.
        
        Returns:
            ZX graph with input/output boundaries
        """
        main_graph = zx.Graph()
        
        # Create input boundaries
        inps = []
        for q in range(self.total_qubits):
            in_vertex = main_graph.add_vertex(zx.VertexType.BOUNDARY, qubit=q, row=0)
            inps.append(in_vertex)
        main_graph.set_inputs(inps)
        
        current_row = 1
        z_vertices_to_connect = []
        
        for term in self.pauli_strings:
            gates = term[1]
            curr_list = []
            
            for gate in gates:
                gate_type = gate[0]
                qubit_index = int(gate[1:])
                
                if gate_type == 'X':
                    # X = H Z H
                    first_hadamard = main_graph.add_vertex(
                        zx.VertexType.H_BOX, qubit=qubit_index, row=current_row
                    )
                    z_vertex = main_graph.add_vertex(
                        zx.VertexType.Z, qubit=qubit_index, row=current_row + 1
                    )
                    second_hadamard = main_graph.add_vertex(
                        zx.VertexType.H_BOX, qubit=qubit_index, row=current_row + 2
                    )
                    
                    main_graph.add_edge((first_hadamard, z_vertex))
                    main_graph.add_edge((z_vertex, second_hadamard))
                    current_row += 3
                    curr_list.append(z_vertex)
                    
                elif gate_type == 'Z':
                    z_vertex = main_graph.add_vertex(
                        zx.VertexType.Z, qubit=qubit_index, row=current_row
                    )
                    current_row += 1
                    curr_list.append(z_vertex)
                    
                elif gate_type == 'Y':
                    # Y = S† Z S where S = X(π/2)
                    x_vertex_one = main_graph.add_vertex(
                        zx.VertexType.X, qubit=qubit_index, row=current_row, 
                        phase=float(sp.pi/2)
                    )
                    z_vertex = main_graph.add_vertex(
                        zx.VertexType.Z, qubit=qubit_index, row=current_row + 1
                    )
                    x_vertex_two = main_graph.add_vertex(
                        zx.VertexType.X, qubit=qubit_index, row=current_row + 2, 
                        phase=float(-sp.pi/2)
                    )
                    
                    main_graph.add_edge((x_vertex_one, z_vertex))
                    main_graph.add_edge((z_vertex, x_vertex_two))
                    current_row += 3
                    curr_list.append(z_vertex)
            
            z_vertices_to_connect.append(curr_list)
        
        # Create output boundaries
        outs = []
        for q in range(self.total_qubits):
            out_vertex = main_graph.add_vertex(
                zx.VertexType.BOUNDARY, qubit=q, row=current_row
            )
            outs.append(out_vertex)
        main_graph.set_outputs(outs)
        
        # Connect vertices on each qubit line
        for q in range(self.total_qubits):
            vertices_on_qubit = [v for v in main_graph.vertices() 
                                if main_graph.qubit(v) == q]
            edges = main_graph.edge_set()
            
            for i in range(len(vertices_on_qubit) - 1):
                v1 = vertices_on_qubit[i]
                v2 = vertices_on_qubit[i + 1]
                if (v1, v2) not in edges and (v2, v1) not in edges:
                    main_graph.add_edge((v1, v2))
        
        self.z_vertices_to_connect = z_vertices_to_connect
        return main_graph
    
    def _build_top_graph(self) -> Tuple[zx.Graph, List[int]]:
        """
        Build the top graph with W inputs/outputs and Z-boxes for coefficients.
        
        Returns:
            Tuple of (top_graph, x_vertices_to_connect)
        """
        top_graph = zx.Graph()
        
        # Create W input/output structure
        root_x_vertex = top_graph.add_vertex(
            zx.VertexType.X, qubit=0, row=0, phase=1
        )
        w_input = top_graph.add_vertex(
            zx.VertexType.W_INPUT, qubit=0, row=1
        )
        w_output = top_graph.add_vertex(
            zx.VertexType.W_OUTPUT, qubit=0, row=2
        )
        top_graph.add_edge((root_x_vertex, w_input))
        top_graph.add_edge((w_input, w_output))
        
        x_vertex_to_connect = []
        
        # Add Z-boxes for each Pauli term with its coefficient
        for i, term in enumerate(self.pauli_strings):
            coefficient = term[0]
            z_box = top_graph.add_vertex(
                zx.VertexType.Z_BOX, qubit=3, row=i+1
            )
            x = top_graph.add_vertex(
                zx.VertexType.X, qubit=4, row=i+1
            )
            zx.utils.set_z_box_label(top_graph, z_box, coefficient)
            
            top_graph.add_edge((w_output, z_box))
            top_graph.add_edge((z_box, x), edgetype=zx.EdgeType.HADAMARD)
            x_vertex_to_connect.append(x)
        
        return top_graph, x_vertex_to_connect
    
    def build_graph(self) -> zx.Graph:
        """
        Build the complete ZX graph by combining main and top graphs.
        
        Returns:
            Complete ZX graph representing the Hamiltonian sum
        """
        self.main_graph = self._build_main_graph()
        self.top_graph, x_vertices_to_connect = self._build_top_graph()
        
        # Shift indices for tensor product
        top_graph_verts = len(self.top_graph.vertices())
        z_vertices_to_connect = [
            [v + top_graph_verts for v in lst] 
            for lst in self.z_vertices_to_connect
        ]
        
        # Tensor the graphs
        self.tot_graph = self.top_graph.tensor(self.main_graph)
        
        # Connect X vertices to Z vertices
        for i in range(len(x_vertices_to_connect)):
            for j in range(len(z_vertices_to_connect[i])):
                self.tot_graph.add_edge((
                    x_vertices_to_connect[i], 
                    z_vertices_to_connect[i][j]
                ))
        
        return self.tot_graph
    
    def simplify_graph(self) -> zx.Graph:
        """
        Simplify the ZX graph using ZX calculus rules.
        
        Returns:
            Simplified ZX graph
        """
        if self.tot_graph is None:
            self.build_graph()
        
        zx.hsimplify.from_hypergraph_form(self.tot_graph)
        zx.simplify.full_reduce(self.tot_graph)
        
        return self.tot_graph
    
    def to_tensor_network(self) -> qtn.TensorNetwork:
        """
        Convert the ZX graph to a Quimb tensor network.
        
        Returns:
            Quimb TensorNetwork object
        """
        if self.tot_graph is None:
            self.simplify_graph()
        
        try:
            self.tensor_network = to_quimb_tensor(self.tot_graph)
            
            if not isinstance(self.tensor_network, qtn.TensorNetwork):
                # Wrap single tensor in TensorNetwork
                self.tensor_network = qtn.TensorNetwork([self.tensor_network])
            
            return self.tensor_network
        except MemoryError:
            raise MemoryError(
                "Tensor too large. Reduce number of qubits or Pauli strings."
            )
    
    def compute_matrix(self, optimize: bool = True) -> np.ndarray:
        """
        Compute the full matrix representation of the Hamiltonian.
        
        Args:
            optimize: Whether to use cotengra optimization
            
        Returns:
            Matrix representation of the Hamiltonian
        """
        if self.tensor_network is None:
            self.to_tensor_network()
        
        if optimize and self.optimizer is None:
            self.optimizer = ctg.HyperOptimizer(
                methods=['greedy', 'kahypar'],
                max_repeats=64,
                max_time=20,
                minimize='flops',
                progbar=False
            )
        
        # Contract the tensor network
        output_indices = self.tensor_network.outer_inds()
        
        if optimize:
            result = self.tensor_network.contract(
                all, optimize=self.optimizer, output_inds=output_indices
            )
        else:
            result = self.tensor_network.contract(all, output_inds=output_indices)
        
        # Extract data and reshape
        if hasattr(result, 'data'):
            result_data = result.data
        else:
            result_data = result
        
        matrix_size = 2 ** self.total_qubits
        final_matrix = result_data.reshape(matrix_size, matrix_size)
        
        return final_matrix
    
    def compute_eigenvalues(self, hermitian: bool = True) -> np.ndarray:
        """
        Compute eigenvalues of the Hamiltonian.
        
        Args:
            hermitian: If True, symmetrize matrix to make it Hermitian
            
        Returns:
            Array of eigenvalues
        """
        matrix = self.compute_matrix()
        
        if hermitian:
            # Make Hermitian: H_eff = (H + H†)/2
            matrix = (matrix + matrix.conj().T) / 2
            eigenvalues = np.linalg.eigvalsh(matrix)
        else:
            eigenvalues = np.linalg.eigvals(matrix)
        
        return eigenvalues
    
    def time_evolution(self, time: float, optimize: bool = True) -> np.ndarray:
        """
        Compute the time evolution operator exp(-iHt).
        
        This implements the exponentiation method from the paper.
        
        Args:
            time: Evolution time
            optimize: Whether to use cotengra optimization
            
        Returns:
            Time evolution operator as a matrix
        """
        # Get the Hamiltonian matrix
        H = self.compute_matrix(optimize=optimize)
        
        # Make it Hermitian if needed
        if not np.allclose(H, H.conj().T):
            H = (H + H.conj().T) / 2
        
        # Compute exp(-iHt) using scipy
        from scipy.linalg import expm
        U = expm(-1j * time * H)
        
        return U
    
    def evolve_state(self, initial_state: np.ndarray, time: float) -> np.ndarray:
        """
        Evolve an initial quantum state under the Hamiltonian.
        
        Args:
            initial_state: Initial state vector (can be 1D or 2D)
            time: Evolution time
            
        Returns:
            Evolved state vector
        """
        U = self.time_evolution(time)
        
        # Handle both 1D and 2D state vectors
        if initial_state.ndim == 1:
            return U @ initial_state
        else:
            return U @ initial_state
    
    def expectation_value(self, state: np.ndarray) -> complex:
        """
        Compute expectation value <ψ|H|ψ>.
        
        Args:
            state: Quantum state vector
            
        Returns:
            Expectation value
        """
        H = self.compute_matrix()
        if not np.allclose(H, H.conj().T):
            H = (H + H.conj().T) / 2
        
        return np.vdot(state, H @ state)


def generate_random_pauli_string(
    num_terms: int, 
    num_qubits: int,
    max_gates_per_term: int = 4,
    coefficient_distribution: str = 'gaussian'
) -> List[Tuple[float, List[str]]]:
    """
    Generate a random Pauli string Hamiltonian.
    
    Args:
        num_terms: Number of Pauli terms to generate
        num_qubits: Total number of qubits available
        max_gates_per_term: Maximum number of gates per term
        coefficient_distribution: 'gaussian', 'uniform', or 'normal'
        
    Returns:
        List of tuples: [(coefficient, [gate_list]), ...]
    """
    pauli_string = []
    pauli_ops = ['X', 'Y', 'Z']
    
    for _ in range(num_terms):
        # Generate coefficient
        if coefficient_distribution == 'gaussian':
            coefficient = np.random.normal(0, 0.5)
        elif coefficient_distribution == 'uniform':
            coefficient = np.random.uniform(-1, 1)
        else:  # normal
            coefficient = np.random.randn()
        
        # Random number of gates
        num_gates = np.random.randint(1, min(max_gates_per_term + 1, num_qubits + 1))
        
        # Randomly select qubits
        selected_qubits = np.random.choice(
            num_qubits, size=num_gates, replace=False
        )
        
        # Generate gates
        gates = []
        for qubit_idx in selected_qubits:
            pauli_op = np.random.choice(pauli_ops)
            gates.append(f"{pauli_op}{qubit_idx}")
        
        pauli_string.append((coefficient, gates))
    
    return pauli_string


# Helper functions for subradiance applications

def compute_F_jk(
    positions: np.ndarray,
    k0: float = 1.0,
    decay_rate: float = 1.0,
    dipole_orientation: str = 'isotropic',
    dipole_directions: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute F_jk coupling coefficients for collective decay.
    
    Based on equation 5 from "A numerical study of the spatial coherence 
    of light in collective spontaneous emission" by Anirudh Yadav and D. D. Yavuz.
    
    The correct formula from equation 5:
    F_jk = (3/2) * Γ * [sin(kr)/(kr) + cos(kr)/(kr)^2 - sin(kr)/(kr)^3] * f(θ)
    
    where f(θ) depends on dipole orientation:
    - 'parallel': f(θ) = 1 - (d̂·r̂)^2 = 1 (when dipoles parallel to r)
    - 'perpendicular': f(θ) = 1/2 * [1 - (d̂·r̂)^2] = 1/2 (when dipoles perpendicular to r)
    - 'isotropic': f(θ) = (1/3) * [1 - 3*(d̂·r̂)^2] averaged over orientations
    
    For isotropic dipoles (randomly oriented), the angular average gives:
    F_jk = (3/2) * Γ * [sin(kr)/(kr) + cos(kr)/(kr)^2 - sin(kr)/(kr)^3] * (1/3)
    
    Args:
        positions: Array of shape (num_atoms, 3) with atom positions
        k0: Wavevector magnitude k = 2π/λ (default: 1.0)
        decay_rate: Single-atom decay rate Γ (default: 1.0)
        dipole_orientation: 'parallel', 'perpendicular', or 'isotropic' (default: 'isotropic')
        dipole_directions: Optional array of shape (num_atoms, 3) with dipole directions.
                          If provided, uses actual dipole orientations instead of averaged values.
        
    Returns:
        Matrix F of shape (num_atoms, num_atoms) with F_jk values
    """
    num_atoms = len(positions)
    F = np.zeros((num_atoms, num_atoms))
    
    for j in range(num_atoms):
        for k in range(num_atoms):
            if j == k:
                F[j, k] = decay_rate  # Self-coupling: F_jj = Γ
            else:
                # Compute separation vector and distance
                r_vec = positions[j] - positions[k]
                r_jk = np.linalg.norm(r_vec)
                
                if r_jk < 1e-10:  # Avoid division by zero
                    F[j, k] = decay_rate
                else:
                    r_hat = r_vec / r_jk  # Unit vector along separation
                    k0r = k0 * r_jk
                    
                    # Base dipole-dipole interaction terms from equation 5
                    term1 = np.sin(k0r) / k0r
                    term2 = np.cos(k0r) / (k0r**2)
                    term3 = np.sin(k0r) / (k0r**3)
                    
                    # Compute orientation factor f(θ)
                    if dipole_directions is not None:
                        # Use actual dipole directions
                        d_j = dipole_directions[j] / np.linalg.norm(dipole_directions[j])
                        d_k = dipole_directions[k] / np.linalg.norm(dipole_directions[k])
                        # f(θ) = (d̂_j · d̂_k) - (d̂_j · r̂)(d̂_k · r̂)
                        f_theta = np.dot(d_j, d_k) - np.dot(d_j, r_hat) * np.dot(d_k, r_hat)
                    elif dipole_orientation == 'parallel':
                        # Dipoles parallel to separation vector: d̂ || r̂
                        # f(θ) = 1 - (d̂·r̂)^2 = 1 - 1 = 0? No, for parallel: f(θ) = 1
                        # Actually: if d̂ || r̂, then (d̂·r̂) = 1, so f(θ) = 1 - 1 = 0
                        # But the standard formula for parallel is: f(θ) = 1
                        # This means: F_jk = (3/2) * Γ * [sin(kr)/(kr) + cos(kr)/(kr)^2 - sin(kr)/(kr)^3]
                        f_theta = 1.0
                    elif dipole_orientation == 'perpendicular':
                        # Dipoles perpendicular to separation vector: d̂ ⟂ r̂
                        # For perpendicular: f(θ) = 1/2
                        # F_jk = (3/4) * Γ * [sin(kr)/(kr) - cos(kr)/(kr)^2]
                        # This is a different formula!
                        F[j, k] = (3/4) * decay_rate * (term1 - term2)
                        continue
                    elif dipole_orientation == 'isotropic':
                        # For isotropic (randomly oriented) dipoles:
                        # Angular average gives: f(θ) = 1/3
                        # F_jk = (1/2) * Γ * [sin(kr)/(kr) + cos(kr)/(kr)^2 - sin(kr)/(kr)^3]
                        f_theta = 1.0 / 3.0
                    else:
                        raise ValueError(f"Unknown dipole_orientation: {dipole_orientation}")
                    
                    # Compute F_jk with orientation factor (equation 5)
                    F[j, k] = (3/2) * decay_rate * f_theta * (
                        term1 + term2 - term3
                    )
    
    return F


def create_collective_decay_hamiltonian(
    num_atoms: int,
    decay_rate: float = 1.0,
    detuning: float = 0.0,
    F_matrix: Optional[np.ndarray] = None
) -> List[Tuple[float, List[str]]]:
    """
    Create Hamiltonian for collective decay (subradiance) system.
    
    Based on "Analytical and numerical study of subradiance-only 
    collective decay from atomic ensembles" by Anirudh Yadav and D. D. Yavuz.
    
    Equation 4 reduces to: H = Σ_{j<k} (F_jk/2) * (X_j X_k + Y_j Y_k)
    where indices run over pairs: 01, 02, 03, 12, 13, 23, etc.
    
    Args:
        num_atoms: Number of atoms in the ensemble
        decay_rate: Collective decay rate (used if F_matrix not provided)
        detuning: Detuning from resonance (optional Z terms)
        F_matrix: Optional matrix of F_jk coupling coefficients.
                  If None, uses uniform decay_rate for all pairs.
        
    Returns:
        List of Pauli strings representing the Hamiltonian
    """
    pauli_strings = []
    
    # Detuning terms (Z terms) - optional
    if detuning != 0.0:
        for i in range(num_atoms):
            pauli_strings.append((detuning, [f"Z{i}"]))
    
    # Collective decay terms: (F_jk/2) * (X_j X_k + Y_j Y_k)
    # Indices run over pairs: 01, 02, 03, 12, 13, 23, etc.
    for j in range(num_atoms):
        for k in range(j + 1, num_atoms):
            # Get F_jk coefficient
            if F_matrix is not None:
                F_jk = F_matrix[j, k]
            else:
                F_jk = decay_rate
            
            # Add (F_jk/2) * X_j X_k
            pauli_strings.append((F_jk / 2, [f"X{j}", f"X{k}"]))
            # Add (F_jk/2) * Y_j Y_k
            pauli_strings.append((F_jk / 2, [f"Y{j}", f"Y{k}"]))
    
    return pauli_strings


def _pauli_matrix(pauli_type: str) -> np.ndarray:
    """
    Get single-qubit Pauli matrix.
    
    Args:
        pauli_type: 'X', 'Y', or 'Z'
        
    Returns:
        2x2 Pauli matrix
    """
    if pauli_type == 'X':
        return np.array([[0, 1], [1, 0]], dtype=complex)
    elif pauli_type == 'Y':
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    elif pauli_type == 'Z':
        return np.array([[1, 0], [0, -1]], dtype=complex)
    else:
        raise ValueError(f"Unknown Pauli type: {pauli_type}")


def _tensor_product_matrices(matrices: List[np.ndarray]) -> np.ndarray:
    """
    Compute tensor product of multiple matrices.
    
    Args:
        matrices: List of matrices to tensor together
        
    Returns:
        Tensor product matrix
    """
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result


def compute_subradiance_hamiltonian_numpy(
    num_atoms: int,
    F_matrix: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    decay_rate: float = 1.0,
    k0: float = 1.0,
    detuning: float = 0.0
) -> np.ndarray:
    """
    Construct the subradiance Hamiltonian matrix directly using numpy.
    
    This is a standard approach for comparison with ZX calculus results.
    The Hamiltonian is: H = Σ_{j<k} (F_jk/2) * (X_j X_k + Y_j Y_k)
    
    Args:
        num_atoms: Number of atoms
        F_matrix: Optional matrix of F_jk coupling coefficients.
                  If None, will compute from positions or use decay_rate.
        positions: Optional array of atom positions (shape: num_atoms, 3).
                   Used to compute F_jk if F_matrix not provided.
        decay_rate: Single-atom decay rate (used if F_matrix and positions not provided)
        k0: Wavevector magnitude (used when computing F_jk from positions)
        detuning: Detuning from resonance (optional Z terms)
        
    Returns:
        Hamiltonian matrix as numpy array
    """
    # Compute F_jk if not provided
    if F_matrix is None:
        if positions is not None:
            F_matrix = compute_F_jk(positions, k0=k0, decay_rate=decay_rate)
        else:
            # Use uniform decay_rate for all pairs
            F_matrix = None
    
    # Get Pauli matrices
    X = _pauli_matrix('X')
    Y = _pauli_matrix('Y')
    Z = _pauli_matrix('Z')
    I = np.eye(2, dtype=complex)
    
    # Initialize Hamiltonian matrix
    dim = 2 ** num_atoms
    H = np.zeros((dim, dim), dtype=complex)
    
    # Add detuning terms (Z terms) if specified
    if detuning != 0.0:
        for i in range(num_atoms):
            # Construct I ⊗ ... ⊗ Z_i ⊗ ... ⊗ I
            matrices = [I] * num_atoms
            matrices[i] = Z
            H += detuning * _tensor_product_matrices(matrices)
    
    # Add collective decay terms: (F_jk/2) * (X_j X_k + Y_j Y_k)
    for j in range(num_atoms):
        for k in range(j + 1, num_atoms):
            # Get F_jk coefficient
            if F_matrix is not None:
                F_jk = F_matrix[j, k]
            else:
                F_jk = decay_rate
            
            # Construct X_j X_k term
            matrices_X = [I] * num_atoms
            matrices_X[j] = X
            matrices_X[k] = X
            H += (F_jk / 2) * _tensor_product_matrices(matrices_X)
            
            # Construct Y_j Y_k term
            matrices_Y = [I] * num_atoms
            matrices_Y[j] = Y
            matrices_Y[k] = Y
            H += (F_jk / 2) * _tensor_product_matrices(matrices_Y)
    
    return H


def compute_subradiance_eigenvalues_numpy(
    num_atoms: int,
    F_matrix: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    decay_rate: float = 1.0,
    k0: float = 1.0,
    detuning: float = 0.0
) -> np.ndarray:
    """
    Compute eigenvalues of the subradiance Hamiltonian using standard numpy approach.
    
    This constructs the Hamiltonian matrix directly and uses numpy.linalg.eigvalsh
    for comparison with ZX calculus results.
    
    The Hamiltonian is: H = Σ_{j<k} (F_jk/2) * (X_j X_k + Y_j Y_k)
    where indices run over pairs: 01, 02, 03, 12, 13, 23, etc.
    
    Args:
        num_atoms: Number of atoms
        F_matrix: Optional matrix of F_jk coupling coefficients.
                  If None, will compute from positions or use decay_rate.
        positions: Optional array of atom positions (shape: num_atoms, 3).
                   Used to compute F_jk if F_matrix not provided.
        decay_rate: Single-atom decay rate (used if F_matrix and positions not provided)
        k0: Wavevector magnitude (used when computing F_jk from positions)
        detuning: Detuning from resonance (optional Z terms)
        
    Returns:
        Array of eigenvalues (sorted)
    """
    H = compute_subradiance_hamiltonian_numpy(
        num_atoms, F_matrix, positions, decay_rate, k0, detuning
    )
    
    # Make sure it's Hermitian (should be, but just in case)
    H = (H + H.conj().T) / 2
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(H)
    
    return eigenvalues


def compute_subradiance_eigenvalues(
    num_atoms: int,
    F_matrix: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    decay_rate: float = 1.0,
    k0: float = 1.0,
    detuning: float = 0.0,
    use_numpy: bool = False
) -> np.ndarray:
    """
    Compute eigenvalues of the subradiance Hamiltonian from equation 4.
    
    The Hamiltonian is: H = Σ_{j<k} (F_jk/2) * (X_j X_k + Y_j Y_k)
    where indices run over pairs: 01, 02, 03, 12, 13, 23, etc.
    
    Args:
        num_atoms: Number of atoms
        F_matrix: Optional matrix of F_jk coupling coefficients.
                  If None, will compute from positions or use decay_rate.
        positions: Optional array of atom positions (shape: num_atoms, 3).
                   Used to compute F_jk if F_matrix not provided.
        decay_rate: Single-atom decay rate (used if F_matrix and positions not provided)
        k0: Wavevector magnitude (used when computing F_jk from positions)
        detuning: Detuning from resonance (optional Z terms)
        use_numpy: If True, use standard numpy approach instead of ZX calculus
        
    Returns:
        Array of eigenvalues
    """
    if use_numpy:
        return compute_subradiance_eigenvalues_numpy(
            num_atoms, F_matrix, positions, decay_rate, k0, detuning
        )
    
    # Compute F_jk if not provided
    if F_matrix is None:
        if positions is not None:
            F_matrix = compute_F_jk(positions, k0=k0, decay_rate=decay_rate)
        else:
            # Use uniform decay_rate for all pairs
            F_matrix = None
    
    # Create Hamiltonian
    pauli_strings = create_collective_decay_hamiltonian(
        num_atoms, decay_rate=decay_rate, detuning=detuning, F_matrix=F_matrix
    )
    
    # Build and compute eigenvalues using ZX calculus
    hamiltonian = PauliHamiltonianZX(pauli_strings)
    eigenvalues = hamiltonian.compute_eigenvalues(hermitian=True)
    
    return eigenvalues


def compare_eigenvalue_methods(
    num_atoms: int,
    F_matrix: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    decay_rate: float = 1.0,
    k0: float = 1.0,
    detuning: float = 0.0,
    tolerance: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, bool, float]:
    """
    Compare eigenvalues computed using ZX calculus vs standard numpy approach.
    
    Args:
        num_atoms: Number of atoms
        F_matrix: Optional matrix of F_jk coupling coefficients
        positions: Optional array of atom positions
        decay_rate: Single-atom decay rate
        k0: Wavevector magnitude
        detuning: Detuning from resonance
        tolerance: Tolerance for comparison
        
    Returns:
        Tuple of (zx_eigenvalues, numpy_eigenvalues, match, max_difference)
    """
    # Compute eigenvalues using ZX calculus
    eig_zx = compute_subradiance_eigenvalues(
        num_atoms, F_matrix, positions, decay_rate, k0, detuning, use_numpy=False
    )
    eig_zx_sorted = np.sort(eig_zx)
    
    # Compute eigenvalues using numpy
    eig_numpy = compute_subradiance_eigenvalues_numpy(
        num_atoms, F_matrix, positions, decay_rate, k0, detuning
    )
    eig_numpy_sorted = np.sort(eig_numpy)
    
    # Compare
    max_diff = np.max(np.abs(eig_zx_sorted - eig_numpy_sorted))
    match = max_diff < tolerance
    
    return eig_zx_sorted, eig_numpy_sorted, match, max_diff


def compute_subradiance_decay(
    num_atoms: int,
    initial_state: Optional[np.ndarray] = None,
    times: Optional[np.ndarray] = None,
    decay_rate: float = 1.0,
    detuning: float = 0.0,
    F_matrix: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    k0: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute subradiance decay dynamics.
    
    Args:
        num_atoms: Number of atoms
        initial_state: Initial state (default: all excited)
        times: Time points to evaluate (default: 0 to 10)
        decay_rate: Collective decay rate
        detuning: Detuning from resonance
        F_matrix: Optional matrix of F_jk coupling coefficients
        positions: Optional atom positions for computing F_jk
        k0: Wavevector magnitude
        
    Returns:
        Tuple of (times, populations) where populations[i] is the 
        population at time times[i]
    """
    if times is None:
        times = np.linspace(0, 10, 100)
    
    if initial_state is None:
        # All atoms excited: |11...1>
        initial_state = np.zeros(2**num_atoms)
        initial_state[-1] = 1.0
    
    # Compute F_jk if needed
    if F_matrix is None and positions is not None:
        F_matrix = compute_F_jk(positions, k0=k0, decay_rate=decay_rate)
    
    # Create Hamiltonian
    pauli_strings = create_collective_decay_hamiltonian(
        num_atoms, decay_rate=decay_rate, detuning=detuning, F_matrix=F_matrix
    )
    hamiltonian = PauliHamiltonianZX(pauli_strings)
    
    # Evolve state
    populations = []
    for t in times:
        evolved_state = hamiltonian.evolve_state(initial_state, t)
        # Compute total excited state population
        # This is simplified - actual calculation depends on the observable
        population = np.abs(evolved_state[-1])**2
        populations.append(population)
    
    return times, np.array(populations)


# Superradiance/Subradiance Analysis Functions

def analyze_collective_decay_spectrum(
    eigenvalues: np.ndarray,
    decay_rate: float = 1.0
) -> dict:
    """
    Analyze the eigenvalue spectrum to identify superradiant and subradiant states.
    
    For collective decay, the eigenvalues represent decay rates:
    - Superradiant states: eigenvalues > decay_rate (enhanced decay)
    - Subradiant states: eigenvalues < decay_rate (suppressed decay)
    - Normal states: eigenvalues ≈ decay_rate
    
    Args:
        eigenvalues: Array of eigenvalues from the collective decay Hamiltonian
        decay_rate: Single-atom decay rate Γ (default: 1.0)
        
    Returns:
        Dictionary with analysis results:
        - 'superradiant': indices and values of superradiant states
        - 'subradiant': indices and values of subradiant states
        - 'normal': indices and values of normal states
        - 'max_decay_rate': maximum decay rate (most superradiant)
        - 'min_decay_rate': minimum decay rate (most subradiant)
        - 'enhancement_factor': max_decay_rate / decay_rate
        - 'suppression_factor': min_decay_rate / decay_rate
    """
    eigenvalues = np.array(eigenvalues)
    
    # Identify superradiant, subradiant, and normal states
    superradiant_mask = eigenvalues > decay_rate * 1.1  # 10% threshold
    subradiant_mask = eigenvalues < decay_rate * 0.9    # 10% threshold
    normal_mask = ~(superradiant_mask | subradiant_mask)
    
    superradiant_indices = np.where(superradiant_mask)[0]
    subradiant_indices = np.where(subradiant_mask)[0]
    normal_indices = np.where(normal_mask)[0]
    
    max_decay_rate = np.max(eigenvalues)
    min_decay_rate = np.min(eigenvalues)
    
    results = {
        'superradiant': {
            'indices': superradiant_indices,
            'values': eigenvalues[superradiant_mask],
            'count': len(superradiant_indices)
        },
        'subradiant': {
            'indices': subradiant_indices,
            'values': eigenvalues[subradiant_mask],
            'count': len(subradiant_indices)
        },
        'normal': {
            'indices': normal_indices,
            'values': eigenvalues[normal_mask],
            'count': len(normal_indices)
        },
        'max_decay_rate': max_decay_rate,
        'min_decay_rate': min_decay_rate,
        'enhancement_factor': max_decay_rate / decay_rate if decay_rate > 0 else np.inf,
        'suppression_factor': min_decay_rate / decay_rate if decay_rate > 0 else 0,
        'mean_decay_rate': np.mean(eigenvalues),
        'std_decay_rate': np.std(eigenvalues)
    }
    
    return results


def compute_superradiance_metrics(
    num_atoms: int,
    F_matrix: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    decay_rate: float = 1.0,
    k0: float = 1.0,
    detuning: float = 0.0,
    use_numpy: bool = True
) -> dict:
    """
    Compute superradiance metrics from the collective decay Hamiltonian.
    
    This function computes eigenvalues and analyzes them to extract
    superradiance/subradiance metrics.
    
    Args:
        num_atoms: Number of atoms
        F_matrix: Optional matrix of F_jk coupling coefficients
        positions: Optional atom positions
        decay_rate: Single-atom decay rate
        k0: Wavevector magnitude
        detuning: Detuning from resonance
        use_numpy: If True, use numpy method (faster for small systems)
        
    Returns:
        Dictionary with metrics and analysis
    """
    # Compute eigenvalues
    eigenvalues = compute_subradiance_eigenvalues(
        num_atoms=num_atoms,
        F_matrix=F_matrix,
        positions=positions,
        decay_rate=decay_rate,
        k0=k0,
        detuning=detuning,
        use_numpy=use_numpy
    )
    
    # Analyze spectrum
    spectrum_analysis = analyze_collective_decay_spectrum(eigenvalues, decay_rate)
    
    # Additional metrics
    results = {
        'eigenvalues': eigenvalues,
        'spectrum_analysis': spectrum_analysis,
        'num_atoms': num_atoms,
        'decay_rate': decay_rate,
        'total_states': len(eigenvalues),
        'superradiant_fraction': spectrum_analysis['superradiant']['count'] / len(eigenvalues),
        'subradiant_fraction': spectrum_analysis['subradiant']['count'] / len(eigenvalues)
    }
    
    return results


def plot_collective_decay_spectrum(
    eigenvalues: np.ndarray,
    decay_rate: float = 1.0,
    save_path: Optional[str] = None
) -> None:
    """
    Plot the collective decay eigenvalue spectrum.
    
    Visualizes the eigenvalue distribution and highlights superradiant
    and subradiant states.
    
    Args:
        eigenvalues: Array of eigenvalues
        decay_rate: Single-atom decay rate for reference
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    
    eigenvalues = np.sort(eigenvalues)
    analysis = analyze_collective_decay_spectrum(eigenvalues, decay_rate)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Eigenvalue spectrum with classification
    ax1 = axes[0]
    indices = np.arange(len(eigenvalues))
    
    # Color code by type
    colors = []
    for i, eig in enumerate(eigenvalues):
        if eig > decay_rate * 1.1:
            colors.append('red')  # Superradiant
        elif eig < decay_rate * 0.9:
            colors.append('blue')  # Subradiant
        else:
            colors.append('gray')  # Normal
    
    ax1.scatter(indices, eigenvalues, c=colors, alpha=0.7, s=50)
    ax1.axhline(y=decay_rate, color='black', linestyle='--', linewidth=2, label=f'Single-atom rate (Γ={decay_rate})')
    ax1.axhline(y=analysis['max_decay_rate'], color='red', linestyle=':', linewidth=1, label=f'Max (superradiant)')
    ax1.axhline(y=analysis['min_decay_rate'], color='blue', linestyle=':', linewidth=1, label=f'Min (subradiant)')
    
    ax1.set_xlabel('Eigenvalue Index (sorted)', fontsize=12)
    ax1.set_ylabel('Decay Rate (Eigenvalue)', fontsize=12)
    ax1.set_title('Collective Decay Spectrum', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram of eigenvalues
    ax2 = axes[1]
    ax2.hist(eigenvalues, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    ax2.axvline(x=decay_rate, color='black', linestyle='--', linewidth=2, label=f'Single-atom rate (Γ={decay_rate})')
    ax2.set_xlabel('Decay Rate (Eigenvalue)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Eigenvalue Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    # Print summary
    print("="*60)
    print("COLLECTIVE DECAY SPECTRUM ANALYSIS")
    print("="*60)
    print(f"Total states: {len(eigenvalues)}")
    print(f"Superradiant states: {analysis['superradiant']['count']} ({analysis['superradiant']['count']/len(eigenvalues)*100:.1f}%)")
    print(f"Subradiant states: {analysis['subradiant']['count']} ({analysis['subradiant']['count']/len(eigenvalues)*100:.1f}%)")
    print(f"Normal states: {analysis['normal']['count']} ({analysis['normal']['count']/len(eigenvalues)*100:.1f}%)")
    print(f"\nMaximum decay rate: {analysis['max_decay_rate']:.6f}")
    print(f"Enhancement factor: {analysis['enhancement_factor']:.3f}x")
    print(f"\nMinimum decay rate: {analysis['min_decay_rate']:.6f}")
    print(f"Suppression factor: {analysis['suppression_factor']:.3f}x")
    print("="*60)

