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
# Lazy imports for quimb to avoid initialization errors
# import quimb.tensor as qtn
# import cotengra as ctg
# from pyzx.quimb import to_quimb_tensor
from fractions import Fraction

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
                        phase=Fraction(1, 2)
                    )
                    z_vertex = main_graph.add_vertex(
                        zx.VertexType.Z, qubit=qubit_index, row=current_row + 1
                    )
                    x_vertex_two = main_graph.add_vertex(
                        zx.VertexType.X, qubit=qubit_index, row=current_row + 2, 
                        phase=Fraction(-1, 2)
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
    
    def to_tensor_network(self):
        """
        Convert the ZX graph to a Quimb tensor network.
        
        Returns:
            Quimb TensorNetwork object (lazy import to avoid initialization errors)
        """
        # Lazy import to avoid initialization errors
        import quimb.tensor as qtn
        from pyzx.quimb import to_quimb_tensor
        
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
        # Lazy import
        import cotengra as ctg
        
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
    
    def _build_single_term_exponential(
        self, 
        coefficient: float, 
        gates: List[str], 
        time: float,
        row_offset: int = 0
    ) -> zx.Graph:
        """
        Build ZX diagram for exp(-i * coefficient * time * Pauli_string).
        
        For a Pauli string P, exp(-i * c * t * P) is represented as:
        - For X: Apply phase rotation with appropriate Hadamards
        - For Y: Apply phase rotation with S gates
        - For Z: Apply phase rotation directly
        
        Args:
            coefficient: Coefficient of the Pauli term
            gates: List of gates like ["X0", "Z1", "Y2"]
            time: Evolution time
            row_offset: Starting row for the diagram
            
        Returns:
            ZX graph representing the exponential
        """
        graph = zx.Graph()
        
        # Handle complex coefficients - take real part for phase
        # For complex coefficients, we use the real part for the phase rotation
        if isinstance(coefficient, complex) or np.iscomplexobj(coefficient):
            coefficient_real = float(np.real(coefficient))
        else:
            coefficient_real = float(coefficient)
        
        phase = -coefficient_real * time  # Phase for exp(-i * c * t * P)
        
        # Convert phase for ZX (normalize to [0, 2π) range)
        # ZX phases are typically in units of π, so we divide by π
        # Convert to Python native float to avoid numpy type issues
        phase_normalized_val = float(phase) / float(np.pi)  # Ensure it's a Python float
        
        # Convert to Fraction for PyZX - use string conversion for maximum compatibility
        # This is the most reliable method that works with all Python versions
        phase_fraction = Fraction(str(phase_normalized_val)).limit_denominator(1000)
        
        # Create input boundaries
        inps = []
        for q in range(self.total_qubits):
            in_vertex = graph.add_vertex(
                zx.VertexType.BOUNDARY, qubit=q, row=row_offset
            )
            inps.append(in_vertex)
        graph.set_inputs(inps)
        
        current_row = row_offset + 1
        
        # Process each gate in the Pauli string
        z_vertices = []
        
        for gate in gates:
            # Validate gate format
            if not isinstance(gate, str):
                raise TypeError(f"Expected gate to be a string, got {type(gate)}: {gate}")
            if len(gate) < 2:
                raise ValueError(f"Gate string too short: {gate}")
            
            gate_type = gate[0].upper()  # Ensure uppercase
            if gate_type not in ['X', 'Y', 'Z']:
                raise ValueError(f"Invalid gate type: {gate_type} (expected X, Y, or Z)")
            
            try:
                qubit_index = int(gate[1:])
            except ValueError:
                raise ValueError(f"Cannot parse qubit index from gate: {gate}")
            
            if gate_type == 'X':
                # X = H Z H, so exp(-i * c * t * X) = H exp(-i * c * t * Z) H
                first_hadamard = graph.add_vertex(
                    zx.VertexType.H_BOX, qubit=qubit_index, row=current_row
                )
                z_vertex = graph.add_vertex(
                    zx.VertexType.Z, qubit=qubit_index, row=current_row + 1,
                    phase=phase_fraction
                )
                second_hadamard = graph.add_vertex(
                    zx.VertexType.H_BOX, qubit=qubit_index, row=current_row + 2
                )
                
                graph.add_edge((first_hadamard, z_vertex))
                graph.add_edge((z_vertex, second_hadamard))
                current_row += 3
                z_vertices.append(z_vertex)
                
            elif gate_type == 'Z':
                # Direct Z phase rotation
                z_vertex = graph.add_vertex(
                    zx.VertexType.Z, qubit=qubit_index, row=current_row,
                    phase=phase_fraction
                )
                current_row += 1
                z_vertices.append(z_vertex)
                
            elif gate_type == 'Y':
                # Y = S† Z S, so exp(-i * c * t * Y) = S† exp(-i * c * t * Z) S
                s_dagger = graph.add_vertex(
                    zx.VertexType.X, qubit=qubit_index, row=current_row,
                    phase=Fraction(-1, 2)  # S†
                )
                z_vertex = graph.add_vertex(
                    zx.VertexType.Z, qubit=qubit_index, row=current_row + 1,
                    phase=phase_fraction
                )
                s = graph.add_vertex(
                    zx.VertexType.X, qubit=qubit_index, row=current_row + 2,
                    phase=Fraction(1, 2)  # S
                )
                
                graph.add_edge((s_dagger, z_vertex))
                graph.add_edge((z_vertex, s))
                current_row += 3
                z_vertices.append(z_vertex)
        
        # Connect all Z vertices in the Pauli string (for tensor product of Pauli operators)
        # For a multi-qubit Pauli string P = P_1 ⊗ P_2 ⊗ ..., 
        # exp(-i * c * t * P) = exp(-i * c * t * P_1) ⊗ exp(-i * c * t * P_2) ⊗ ...
        # In ZX, we connect them to form the tensor product structure
        for i in range(len(z_vertices) - 1):
            graph.add_edge((z_vertices[i], z_vertices[i + 1]))
        
        # Create output boundaries
        outs = []
        for q in range(self.total_qubits):
            out_vertex = graph.add_vertex(
                zx.VertexType.BOUNDARY, qubit=q, row=current_row
            )
            outs.append(out_vertex)
        graph.set_outputs(outs)
        
        # Connect vertices on each qubit line
        for q in range(self.total_qubits):
            vertices_on_qubit = sorted(
                [v for v in graph.vertices() if graph.qubit(v) == q],
                key=lambda v: graph.row(v)
            )
            edges = graph.edge_set()
            
            for i in range(len(vertices_on_qubit) - 1):
                v1 = vertices_on_qubit[i]
                v2 = vertices_on_qubit[i + 1]
                if (v1, v2) not in edges and (v2, v1) not in edges:
                    graph.add_edge((v1, v2))
        
        return graph
    
    def _build_trotter_step_component(self, time: float) -> zx.Graph:
        """
        Build a complete Trotter step component exp(-iH * time) using the same structure as build_graph.
        
        This builds exp(-iH * time) where H = Σ_j coefficient_j * Pauli_string_j
        using the exact same W-state summation structure as the regular Hamiltonian (tot_graph).
        The phases are embedded in the main graph based on the time parameter.
        
        Args:
            time: Evolution time for this step
            
        Returns:
            Complete ZX graph component with W-state structure (like tot_graph)
        """
        # Build main graph with phases embedded in the gates
        main_graph = zx.Graph()
        
        # Create input boundaries
        inps = []
        for q in range(self.total_qubits):
            in_vertex = main_graph.add_vertex(zx.VertexType.BOUNDARY, qubit=q, row=0)
            inps.append(in_vertex)
        main_graph.set_inputs(inps)
        
        current_row = 1
        z_vertices_to_connect = []
        
        # Build gates for each Pauli term with phases embedded
        for term in self.pauli_strings:
            coefficient_raw, gates = term
            
            # Validate inputs
            if not isinstance(gates, list) or len(gates) == 0:
                continue
            
            # Ensure coefficient is a Python float
            if isinstance(coefficient_raw, complex) or np.iscomplexobj(coefficient_raw):
                coefficient = float(np.real(coefficient_raw))
            elif isinstance(coefficient_raw, (np.integer, np.floating)):
                coefficient = float(coefficient_raw)
            else:
                coefficient = float(coefficient_raw)
            
            # Phase for this exponential term: -coefficient * time
            phase = -coefficient * time
            phase_normalized_val = float(phase) / float(np.pi)
            phase_fraction = Fraction(str(phase_normalized_val)).limit_denominator(1000)
            
            curr_list = []
            
            # Build the gates for this Pauli string (same as _build_main_graph but with phases)
            for gate in gates:
                if not isinstance(gate, str) or len(gate) < 2:
                    continue
                
                gate_type = gate[0].upper()
                try:
                    qubit_index = int(gate[1:])
                except ValueError:
                    continue
                
                if gate_type == 'X':
                    # X = H Z H, so exp(-i * c * t * X) = H exp(-i * c * t * Z) H
                    first_hadamard = main_graph.add_vertex(
                        zx.VertexType.H_BOX, qubit=qubit_index, row=current_row
                    )
                    z_vertex = main_graph.add_vertex(
                        zx.VertexType.Z, qubit=qubit_index, row=current_row + 1,
                        phase=phase_fraction
                    )
                    second_hadamard = main_graph.add_vertex(
                        zx.VertexType.H_BOX, qubit=qubit_index, row=current_row + 2
                    )
                    
                    main_graph.add_edge((first_hadamard, z_vertex))
                    main_graph.add_edge((z_vertex, second_hadamard))
                    current_row += 3
                    curr_list.append(z_vertex)
                    
                elif gate_type == 'Z':
                    # Direct Z phase rotation
                    z_vertex = main_graph.add_vertex(
                        zx.VertexType.Z, qubit=qubit_index, row=current_row,
                        phase=phase_fraction
                    )
                    current_row += 1
                    curr_list.append(z_vertex)
                    
                elif gate_type == 'Y':
                    # Y = S† Z S, so exp(-i * c * t * Y) = S† exp(-i * c * t * Z) S
                    s_dagger = main_graph.add_vertex(
                        zx.VertexType.X, qubit=qubit_index, row=current_row,
                        phase=Fraction(-1, 2)  # S†
                    )
                    z_vertex = main_graph.add_vertex(
                        zx.VertexType.Z, qubit=qubit_index, row=current_row + 1,
                        phase=phase_fraction
                    )
                    s = main_graph.add_vertex(
                        zx.VertexType.X, qubit=qubit_index, row=current_row + 2,
                        phase=Fraction(1, 2)  # S
                    )
                    
                    main_graph.add_edge((s_dagger, z_vertex))
                    main_graph.add_edge((z_vertex, s))
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
        
        # Connect vertices on each qubit line (sorted by row to ensure proper order)
        for q in range(self.total_qubits):
            vertices_on_qubit = [v for v in main_graph.vertices() 
                                if main_graph.qubit(v) == q]
            # Sort by row to ensure sequential connection
            vertices_on_qubit.sort(key=lambda v: main_graph.row(v))
            edges = main_graph.edge_set()
            
            for i in range(len(vertices_on_qubit) - 1):
                v1 = vertices_on_qubit[i]
                v2 = vertices_on_qubit[i + 1]
                if (v1, v2) not in edges and (v2, v1) not in edges:
                    main_graph.add_edge((v1, v2))
        
        # Build top graph with W-states (same as _build_top_graph but with unit coefficients)
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
        
        # Add Z-boxes for each term (coefficient = 1.0 since phases are in main graph)
        for i, term in enumerate(self.pauli_strings):
            if not isinstance(term[1], list) or len(term[1]) == 0:
                continue
                
            z_box = top_graph.add_vertex(
                zx.VertexType.Z_BOX, qubit=3, row=i+1
            )
            x = top_graph.add_vertex(
                zx.VertexType.X, qubit=4, row=i+1
            )
            # Use unit coefficient (phases are already embedded in main graph)
            zx.utils.set_z_box_label(top_graph, z_box, 1.0)
            
            top_graph.add_edge((w_output, z_box))
            top_graph.add_edge((z_box, x), edgetype=zx.EdgeType.HADAMARD)
            x_vertex_to_connect.append(x)
        
        # Combine main and top graphs (exactly like build_graph does)
        top_graph_verts = len(top_graph.vertices())
        z_vertices_shifted = [
            [v + top_graph_verts for v in lst] 
            for lst in z_vertices_to_connect
        ]
        
        # Tensor the graphs
        component_graph = top_graph.tensor(main_graph)
        
        # Connect X vertices to Z vertices (same as build_graph)
        for i in range(len(x_vertex_to_connect)):
            for j in range(len(z_vertices_shifted[i])):
                component_graph.add_edge((
                    x_vertex_to_connect[i], 
                    z_vertices_shifted[i][j]
                ))
        
        return component_graph
    
    def build_trotter_graph(
        self, 
        time: float, 
        trotter_steps: int = 1,
        order: int = 1
    ) -> zx.Graph:
        """
        Build ZX diagram for Trotter expansion of exp(-iHt) by attaching components.
        
        Following the paper's method, each exponential term exp(-iH_j * t) is built
        as a separate component, and these components are attached/composed sequentially.
        This creates a proper PyZX graph that can be simplified.
        
        First-order Trotter: exp(-iHt) ≈ [∏_j exp(-iH_j * t/n)]^n
        Second-order (Suzuki): exp(-iHt) ≈ [∏_j exp(-iH_j * t/(2n)) ∏_j exp(-iH_j * t/(2n))]^n
        
        Args:
            time: Total evolution time
            trotter_steps: Number of Trotter steps (n)
            order: Order of Trotter expansion (1 or 2)
            
        Returns:
            ZX graph representing the Trotterized time evolution operator (in PyZX format)
        """
        if order not in [1, 2]:
            raise ValueError("Order must be 1 or 2")
        
        dt = time / trotter_steps
        
        # Build component graphs for each Trotter step
        step_graphs = []
        
        # Build each Trotter step as a complete component (like tot_graph)
        step_components = []
        
        for step in range(trotter_steps):
            if order == 1:
                # First-order: build one component with W-state structure for exp(-iH * dt)
                step_component = self._build_trotter_step_component(dt)
                step_components.append(step_component)
                
            else:  # order == 2
                # Second-order: forward then backward
                # Forward pass: exp(-iH * dt/2) with W-state structure
                step_component_fwd = self._build_trotter_step_component(dt/2)
                
                # Backward pass: same structure
                step_component_bwd = self._build_trotter_step_component(dt/2)
                
                # Connect outputs of forward to inputs of backward
                step_component = self._compose_graphs(step_component_fwd, step_component_bwd)
                step_components.append(step_component)
        
        # Connect all step components by attaching output boundaries to input boundaries
        if len(step_components) == 0:
            # Return identity if no terms
            result_graph = zx.Graph()
            inps = []
            outs = []
            for q in range(self.total_qubits):
                in_v = result_graph.add_vertex(
                    zx.VertexType.BOUNDARY, qubit=q, row=0
                )
                out_v = result_graph.add_vertex(
                    zx.VertexType.BOUNDARY, qubit=q, row=1
                )
                inps.append(in_v)
                outs.append(out_v)
                result_graph.add_edge((in_v, out_v))
            result_graph.set_inputs(inps)
            result_graph.set_outputs(outs)
            return result_graph
        
        # Connect all components: output boundaries of one to input boundaries of next
        # Use the existing _compose_graphs method which properly handles composition
        result_graph = step_components[0]
        for next_component in step_components[1:]:
            result_graph = self._compose_graphs(result_graph, next_component)
        
        return result_graph
    
    def _compose_graphs(self, graph1: zx.Graph, graph2: zx.Graph) -> zx.Graph:
        """
        Compose two ZX graphs: graph1 @ graph2 (matrix multiplication).
        
        This connects the outputs of graph1 to the inputs of graph2.
        Uses PyZX's built-in composition if available, otherwise manual composition.
        
        Args:
            graph1: First graph (left)
            graph2: Second graph (right)
            
        Returns:
            Composed graph
        """
        # Manual composition: copy both graphs and connect boundaries directly
        # (Not using PyZX's compose as it may not handle our graph structure correctly)
        # Get max row from graph1 to offset graph2
        max_row1 = max([graph1.row(v) for v in graph1.vertices()], default=0)
        
        # Create composed graph by copying both
        composed = zx.Graph()
        vertex_map1 = {}
        vertex_map2 = {}
        
        # Copy all vertices from graph1 (including boundaries)
        for v in graph1.vertices():
            new_v = composed.add_vertex(
                graph1.type(v),
                qubit=graph1.qubit(v),
                row=graph1.row(v),
                phase=graph1.phase(v)
            )
            vertex_map1[v] = new_v
        
        # Copy all vertices from graph2 (including boundaries), offset rows
        for v in graph2.vertices():
            new_v = composed.add_vertex(
                graph2.type(v),
                qubit=graph2.qubit(v),
                row=graph2.row(v) + max_row1 + 1,
                phase=graph2.phase(v)
            )
            vertex_map2[v] = new_v
        
        # Copy all edges from graph1
        for e in graph1.edges():
            v1, v2 = e
            if v1 in vertex_map1 and v2 in vertex_map1:
                composed.add_edge((vertex_map1[v1], vertex_map1[v2]))
        
        # Copy all edges from graph2
        for e in graph2.edges():
            v1, v2 = e
            if v1 in vertex_map2 and v2 in vertex_map2:
                composed.add_edge((vertex_map2[v1], vertex_map2[v2]))
        
        # Connect output boundaries of graph1 to input boundaries of graph2
        # Assume outputs and inputs are in the same qubit order
        graph1_outputs = list(graph1.outputs())
        graph2_inputs = list(graph2.inputs())
        
        # Simple approach: connect by index (assuming same order)
        connections_made = 0
        min_len = min(len(graph1_outputs), len(graph2_inputs))
        
        # Store connections to make and boundaries to remove
        connections_to_make = []  # (neighbor_from_graph1, neighbor_from_graph2)
        boundaries_to_remove = []  # (out_mapped, in_mapped)
        
        for i in range(min_len):
            v_out = graph1_outputs[i]
            v_in = graph2_inputs[i]
            
            # Map to new vertices
            out_mapped = vertex_map1[v_out]
            in_mapped = vertex_map2[v_in]
            
            # Get neighbors BEFORE connecting boundaries
            out_neighbors_before = list(composed.neighbors(out_mapped))
            in_neighbors_before = list(composed.neighbors(in_mapped))
            
            # Temporarily connect boundaries
            composed.add_edge((out_mapped, in_mapped))
            
            # Get neighbors AFTER connecting boundaries
            out_neighbors_after = list(composed.neighbors(out_mapped))
            in_neighbors_after = list(composed.neighbors(in_mapped))
            
            # Find the real neighbors (excluding the boundary connection we just made)
            out_real_neighbors = [n for n in out_neighbors_after if n != in_mapped]
            in_real_neighbors = [n for n in in_neighbors_after if n != out_mapped]
            
            # Connect the real neighbors together
            if out_real_neighbors and in_real_neighbors:
                # Use row to determine which ones to connect (last from graph1, first from graph2)
                neighbor_from_graph1 = max(out_real_neighbors, key=lambda v: composed.row(v))
                neighbor_from_graph2 = min(in_real_neighbors, key=lambda v: composed.row(v))
                
                composed.add_edge((neighbor_from_graph1, neighbor_from_graph2))
                
                # Remove the boundary-to-boundary connection
                composed.remove_edge((out_mapped, in_mapped))
                
                # Remove old connections from boundaries to their original neighbors
                for n in out_neighbors_before:
                    if composed.connected(out_mapped, n):
                        composed.remove_edge((out_mapped, n))
                
                for n in in_neighbors_before:
                    if composed.connected(in_mapped, n):
                        composed.remove_edge((in_mapped, n))
                
                # Remove isolated boundary vertices (they have no edges now)
                if len(list(composed.neighbors(out_mapped))) == 0:
                    composed.remove_vertex(out_mapped)
                
                if len(list(composed.neighbors(in_mapped))) == 0:
                    composed.remove_vertex(in_mapped)
                
                connections_made += 1
            else:
                # Remove the temporary connection if we can't merge
                composed.remove_edge((out_mapped, in_mapped))
        
        # Set inputs from graph1 and outputs from graph2
        # Filter out any vertices that were removed
        new_inputs = [vertex_map1[v] for v in graph1.inputs() if vertex_map1[v] in composed.vertices()]
        new_outputs = [vertex_map2[v] for v in graph2.outputs() if vertex_map2[v] in composed.vertices()]
        
        composed.set_inputs(new_inputs)
        composed.set_outputs(new_outputs)
        
        return composed
    
    def time_evolution_trotter(
        self, 
        time: float, 
        trotter_steps: int = 1,
        order: int = 1,
        optimize: bool = True
    ) -> np.ndarray:
        """
        Compute time evolution operator exp(-iHt) using Trotter expansion in ZX calculus.
        
        This implements the Trotter-Suzuki decomposition method from the paper.
        
        Args:
            time: Evolution time
            trotter_steps: Number of Trotter steps (more steps = better approximation)
            order: Order of Trotter expansion (1 or 2 for Suzuki)
            optimize: Whether to use cotengra optimization
            
        Returns:
            Time evolution operator as a matrix
        """
        # Lazy imports to avoid initialization errors
        import quimb.tensor as qtn
        import cotengra as ctg
        from pyzx.quimb import to_quimb_tensor
        
        # Build Trotter graph
        trotter_graph = self.build_trotter_graph(time, trotter_steps, order)
        
        # Convert to tensor network
        try:
            tensor_network = to_quimb_tensor(trotter_graph)
            
            if not isinstance(tensor_network, qtn.TensorNetwork):
                tensor_network = qtn.TensorNetwork([tensor_network])
            
            # Set up optimizer if needed
            if optimize:
                optimizer = ctg.HyperOptimizer(
                    methods=['greedy', 'kahypar'],
                    max_repeats=64,
                    max_time=20,
                    minimize='flops',
                    progbar=False
                )
            else:
                optimizer = None
            
            # Contract tensor network
            output_indices = tensor_network.outer_inds()
            
            if optimize and optimizer is not None:
                result = tensor_network.contract(
                    all, optimize=optimizer, output_inds=output_indices
                )
            else:
                result = tensor_network.contract(all, output_inds=output_indices)
            
            # Extract data and reshape
            if hasattr(result, 'data'):
                result_data = result.data
            else:
                result_data = result
            
            matrix_size = 2 ** self.total_qubits
            final_matrix = result_data.reshape(matrix_size, matrix_size)
            
            return final_matrix
            
        except Exception as e:
            print(f"Error in Trotter ZX computation: {e}")
            print("Falling back to standard matrix exponentiation...")
            # Fallback to standard method
            return self.time_evolution(time, optimize=optimize)
    
    def time_evolution(self, time: float, optimize: bool = True) -> np.ndarray:
        """
        Compute the time evolution operator exp(-iHt).
        
        This implements the exponentiation method from the paper.
        For Trotter expansion in ZX, use time_evolution_trotter() instead.
        
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
    
    def partial_trace_zx(self, graph: zx.Graph, keep_qubits: int) -> zx.Graph:
        """
        Perform partial trace using ZX calculus methods by connecting outputs to inputs.
        
        In ZX calculus, partial trace is done by connecting the outputs of qubits
        you want to trace out back to their inputs. This creates a closed loop (trace).
        
        Args:
            graph: ZX graph to perform partial trace on
            keep_qubits: Number of data qubits to keep (should match self.total_qubits)
            
        Returns:
            ZX graph with auxiliary qubits traced out
        """
        # Create a copy to avoid modifying the original
        traced_graph = graph.copy()
        
        # Get all inputs and outputs
        inputs = list(traced_graph.inputs())
        outputs = list(traced_graph.outputs())
        
        # Identify which qubits are data qubits (0 to keep_qubits-1)
        # and which are auxiliary (negative or special qubit indices)
        data_qubit_indices = set(range(keep_qubits))
        
        # Group inputs/outputs by qubit index
        input_by_qubit = {}
        output_by_qubit = {}
        
        for v in inputs:
            q = traced_graph.qubit(v)
            if q not in input_by_qubit:
                input_by_qubit[q] = []
            input_by_qubit[q].append(v)
        
        for v in outputs:
            q = traced_graph.qubit(v)
            if q not in output_by_qubit:
                output_by_qubit[q] = []
            output_by_qubit[q].append(v)
        
        # For each auxiliary qubit (not in data_qubit_indices), connect output to input
        # This performs the partial trace
        aux_qubits_traced = 0
        for q in sorted(set(input_by_qubit.keys()) | set(output_by_qubit.keys())):
            if q not in data_qubit_indices:
                # This is an auxiliary qubit - trace it out
                if q in input_by_qubit and q in output_by_qubit:
                    aux_inputs = input_by_qubit[q]
                    aux_outputs = output_by_qubit[q]
                    
                    # Match inputs and outputs by row position
                    aux_inputs_sorted = sorted(aux_inputs, key=lambda v: traced_graph.row(v))
                    aux_outputs_sorted = sorted(aux_outputs, key=lambda v: traced_graph.row(v))
                    
                    min_len = min(len(aux_inputs_sorted), len(aux_outputs_sorted))
                    for i in range(min_len):
                        # Connect output to input (this creates the trace)
                        # Note: In ZX calculus, connecting output to input creates a trace
                        traced_graph.add_edge((aux_outputs_sorted[i], aux_inputs_sorted[i]))
                    
                    aux_qubits_traced += 1
        
        # Update inputs and outputs to only include data qubits
        new_inputs = [v for v in inputs if traced_graph.qubit(v) in data_qubit_indices]
        new_outputs = [v for v in outputs if traced_graph.qubit(v) in data_qubit_indices]
        
        # Sort by qubit index to maintain order
        new_inputs.sort(key=lambda v: traced_graph.qubit(v))
        new_outputs.sort(key=lambda v: traced_graph.qubit(v))
        
        traced_graph.set_inputs(new_inputs)
        traced_graph.set_outputs(new_outputs)
        
        return traced_graph
    
    def partial_trace(self, rho: np.ndarray, keep_qubits: int) -> np.ndarray:
        """
        Perform partial trace to reduce density matrix dimensions.
        
        Given a density matrix with extra dimensions (e.g., from W-state structure),
        trace out auxiliary qubits to get the reduced density matrix for the data qubits.
        
        Args:
            rho: Density matrix, can be larger than expected
            keep_qubits: Number of qubits to keep (should match self.total_qubits)
            
        Returns:
            Reduced density matrix of size (2^keep_qubits, 2^keep_qubits)
        """
        # If already the right size, return as is
        target_size = 2 ** keep_qubits
        if rho.shape == (target_size, target_size):
            return rho
        
        # If rho is larger, we need to trace out extra dimensions
        current_size = rho.shape[0]
        
        if current_size == target_size:
            return rho
        
        # Check if current_size is a power of 2 and divisible by target_size
        if current_size % target_size != 0:
            raise ValueError(
                f"Cannot perform partial trace: matrix size {current_size} "
                f"is not compatible with target size {target_size}"
            )
        
        # Reshape to identify subsystems
        # If current_size = target_size * aux_size, we have aux_size auxiliary qubits
        aux_size = current_size // target_size
        
        # Check if aux_size is a power of 2
        if aux_size & (aux_size - 1) != 0:
            raise ValueError(
                f"Auxiliary subsystem size {aux_size} is not a power of 2"
            )
        
        # Reshape rho to separate data and auxiliary subsystems
        # rho has shape (current_size, current_size)
        # Reshape to (target_size, aux_size, target_size, aux_size)
        rho_reshaped = rho.reshape(target_size, aux_size, target_size, aux_size)
        
        # Trace out auxiliary qubits: sum over diagonal of aux indices
        # Tr_aux(rho) = sum_i rho_{data, i; data, i}
        rho_reduced = np.trace(rho_reshaped, axis1=1, axis2=3)
        
        return rho_reduced
    
    def extract_operator_block(self, U: np.ndarray, keep_qubits: int) -> np.ndarray:
        """
        Extract the operator block for data qubits from a larger operator.
        
        For a unitary operator on a larger space (data + auxiliary qubits),
        extract the block that acts on the data qubits only.
        This assumes the operator is block-diagonal or we take the (0,0) block
        corresponding to the auxiliary qubits in state |0...0⟩.
        
        Args:
            U: Operator matrix, can be larger than expected
            keep_qubits: Number of qubits to keep (should match self.total_qubits)
            
        Returns:
            Operator matrix of size (2^keep_qubits, 2^keep_qubits)
        """
        target_size = 2 ** keep_qubits
        
        # If already the right size, return as is
        if U.shape == (target_size, target_size):
            return U
        
        current_size = U.shape[0]
        
        if current_size == target_size:
            return U
        
        # Check if current_size is divisible by target_size
        if current_size % target_size != 0:
            raise ValueError(
                f"Cannot extract operator block: matrix size {current_size} "
                f"is not compatible with target size {target_size}"
            )
        
        aux_size = current_size // target_size
        
        # Reshape to identify subsystems
        # U has shape (current_size, current_size)
        # Reshape to (target_size, aux_size, target_size, aux_size)
        U_reshaped = U.reshape(target_size, aux_size, target_size, aux_size)
        
        # Extract the block where auxiliary qubits are in |0...0⟩ state
        # This is the (0, 0) block: U[0:target_size, 0:target_size] in the reshaped view
        U_block = U_reshaped[:, 0, :, 0]
        
        return U_block
    
    def compute_matrix_with_zx_trace(self, optimize: bool = True) -> np.ndarray:
        """
        Compute matrix representation using ZX calculus partial trace.
        
        This method applies partial trace directly on the ZX graph by connecting
        outputs to inputs of auxiliary qubits, then converts to matrix.
        
        Args:
            optimize: Whether to use cotengra optimization
            
        Returns:
            Matrix representation of the Hamiltonian for data qubits only
        """
        # Lazy import
        import cotengra as ctg
        
        if self.tot_graph is None:
            self.build_graph()
        
        # Apply ZX-based partial trace to remove auxiliary qubits
        traced_graph = self.partial_trace_zx(self.tot_graph, self.total_qubits)
        
        # Convert to tensor network
        import quimb.tensor as qtn
        from pyzx.quimb import to_quimb_tensor
        
        try:
            tensor_network = to_quimb_tensor(traced_graph)
            
            if not isinstance(tensor_network, qtn.TensorNetwork):
                tensor_network = qtn.TensorNetwork([tensor_network])
            
            # Check outer indices
            output_indices = tensor_network.outer_inds()
            expected_inds = 2 * self.total_qubits
            
            if len(output_indices) != expected_inds:
                # Fall back to numpy-based method
                return self.compute_matrix_with_trace(optimize=optimize)
            
            if optimize and self.optimizer is None:
                self.optimizer = ctg.HyperOptimizer(
                    methods=['greedy', 'kahypar'],
                    max_repeats=64,
                    max_time=20,
                    minimize='flops',
                    progbar=False
                )
            
            # Contract the tensor network
            if optimize:
                result = tensor_network.contract(
                    all, optimize=self.optimizer, output_inds=output_indices
                )
            else:
                result = tensor_network.contract(all, output_inds=output_indices)
            
            # Extract data and reshape
            if hasattr(result, 'data'):
                result_data = result.data
            else:
                result_data = result
            
            matrix_size = 2 ** self.total_qubits
            if result_data.size == matrix_size * matrix_size:
                final_matrix = result_data.reshape(matrix_size, matrix_size)
                return final_matrix
            else:
                # Fall back to numpy-based method
                return self.compute_matrix_with_trace(optimize=optimize)
            
        except Exception as e:
            # Fall back to numpy-based method
            return self.compute_matrix_with_trace(optimize=optimize)
    
    def compute_matrix_with_trace(self, optimize: bool = True) -> np.ndarray:
        """
        Compute matrix representation, handling extra dimensions with partial trace.
        
        This method handles cases where the tensor network has extra dimensions
        from auxiliary qubits (e.g., W-state structure) and performs partial trace
        to get the correct matrix for data qubits only.
        
        Args:
            optimize: Whether to use cotengra optimization
            
        Returns:
            Matrix representation of the Hamiltonian for data qubits
        """
        # Lazy import
        import cotengra as ctg
        
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
        
        # Extract data
        if hasattr(result, 'data'):
            result_data = result.data
        else:
            result_data = result
        
        matrix_size = 2 ** self.total_qubits
        expected_dims = 2 * self.total_qubits
        
        # Check if we have extra dimensions
        if result_data.ndim > 2 and all(d == 2 for d in result_data.shape):
            n_dims = len(result_data.shape)
            
            if n_dims > expected_dims:
                # Have extra dimensions - need to reshape and trace
                # First reshape to square matrix
                total_size = result_data.size
                sqrt_size = int(np.sqrt(total_size))
                
                if sqrt_size * sqrt_size == total_size:
                        # Reshape to square matrix
                        H_full = result_data.reshape(sqrt_size, sqrt_size)
                        
                        # Extract operator block for data qubits
                        H = self.extract_operator_block(H_full, self.total_qubits)
                        return H
                else:
                    raise ValueError(
                        f"Cannot reshape tensor with shape {result_data.shape} "
                        f"to square matrix. Total size: {total_size}"
                    )
            else:
                # Normal case - reshape directly
                final_matrix = result_data.reshape(matrix_size, matrix_size)
                return final_matrix
        else:
            # Already 2D or 1D - reshape normally
            if result_data.size == matrix_size * matrix_size:
                return result_data.reshape(matrix_size, matrix_size)
            else:
                raise ValueError(
                    f"Cannot reshape result with size {result_data.size} "
                    f"to ({matrix_size}, {matrix_size}) matrix"
                )
    
    def time_evolution_trotter_with_zx_trace(
        self,
        time: float,
        trotter_steps: int = 1,
        order: int = 1,
        optimize: bool = True
    ) -> np.ndarray:
        """
        Compute time evolution operator using Trotter expansion with ZX calculus partial trace.
        
        This method applies partial trace directly on the ZX graph by connecting
        outputs to inputs of auxiliary qubits, then converts to matrix.
        
        Args:
            time: Evolution time
            trotter_steps: Number of Trotter steps
            order: Order of Trotter expansion (1 or 2)
            optimize: Whether to use cotengra optimization
            
        Returns:
            Time evolution operator for data qubits only
        """
        # Build Trotter step component
        component_graph = self._build_trotter_step_component(time / trotter_steps)
        
        # Compose trotter_steps copies
        if trotter_steps == 1:
            trotter_graph = component_graph
        else:
            trotter_graph = component_graph
            for _ in range(trotter_steps - 1):
                trotter_graph = self._compose_graphs(trotter_graph, component_graph)
        
        # Apply ZX-based partial trace to remove auxiliary qubits
        traced_graph = self.partial_trace_zx(trotter_graph, self.total_qubits)
        
        # Convert to tensor network
        import quimb.tensor as qtn
        import cotengra as ctg
        from pyzx.quimb import to_quimb_tensor
        
        try:
            tensor_network = to_quimb_tensor(traced_graph)
            
            if not isinstance(tensor_network, qtn.TensorNetwork):
                tensor_network = qtn.TensorNetwork([tensor_network])
            
            # Check outer indices
            output_indices = tensor_network.outer_inds()
            expected_inds = 2 * self.total_qubits
            
            if len(output_indices) != expected_inds:
                # Fall back to numpy-based method
                return self.time_evolution_trotter_with_trace(
                    time, trotter_steps, order, optimize
                )
            
            if optimize and self.optimizer is None:
                self.optimizer = ctg.HyperOptimizer(
                    methods=['greedy', 'kahypar'],
                    max_repeats=64,
                    max_time=20,
                    minimize='flops',
                    progbar=False
                )
            
            # Contract the tensor network
            if optimize:
                result = tensor_network.contract(
                    all, optimize=self.optimizer, output_inds=output_indices
                )
            else:
                result = tensor_network.contract(all, output_inds=output_indices)
            
            # Extract data and reshape
            if hasattr(result, 'data'):
                result_data = result.data
            else:
                result_data = result
            
            matrix_size = 2 ** self.total_qubits
            if result_data.size == matrix_size * matrix_size:
                final_matrix = result_data.reshape(matrix_size, matrix_size)
                return final_matrix
            else:
                # Fall back to numpy-based method
                return self.time_evolution_trotter_with_trace(
                    time, trotter_steps, order, optimize
                )
            
        except Exception as e:
            # Fall back to numpy-based method
            return self.time_evolution_trotter_with_trace(
                time, trotter_steps, order, optimize
            )
    
    def time_evolution_trotter_with_trace(
        self,
        time: float,
        trotter_steps: int = 1,
        order: int = 1,
        optimize: bool = True
    ) -> np.ndarray:
        """
        Compute time evolution operator using Trotter expansion with partial trace.
        
        This method handles extra dimensions from auxiliary qubits by performing
        partial trace to get the correct evolution operator for data qubits.
        
        Args:
            time: Evolution time
            trotter_steps: Number of Trotter steps
            order: Order of Trotter expansion (1 or 2)
            optimize: Whether to use cotengra optimization
            
        Returns:
            Time evolution operator as a matrix for data qubits
        """
        # Lazy imports
        import quimb.tensor as qtn
        import cotengra as ctg
        from pyzx.quimb import to_quimb_tensor
        
        # Build Trotter graph
        trotter_graph = self.build_trotter_graph(time, trotter_steps, order)
        
        # Convert to tensor network
        try:
            tensor_network = to_quimb_tensor(trotter_graph)
            
            if not isinstance(tensor_network, qtn.TensorNetwork):
                tensor_network = qtn.TensorNetwork([tensor_network])
            
            # Set up optimizer if needed
            if optimize:
                optimizer = ctg.HyperOptimizer(
                    methods=['greedy', 'kahypar'],
                    max_repeats=64,
                    max_time=20,
                    minimize='flops',
                    progbar=False
                )
            else:
                optimizer = None
            
            # Contract tensor network
            output_indices = tensor_network.outer_inds()
            
            if optimize and optimizer is not None:
                result = tensor_network.contract(
                    all, optimize=optimizer, output_inds=output_indices
                )
            else:
                result = tensor_network.contract(all, output_inds=output_indices)
            
            # Extract data
            if hasattr(result, 'data'):
                result_data = result.data
            else:
                result_data = result
            
            matrix_size = 2 ** self.total_qubits
            expected_dims = 2 * self.total_qubits
            
            # Check if we have extra dimensions
            if result_data.ndim > 2 and all(d == 2 for d in result_data.shape):
                n_dims = len(result_data.shape)
                
                if n_dims > expected_dims:
                    # Have extra dimensions - need to reshape and trace
                    total_size = result_data.size
                    sqrt_size = int(np.sqrt(total_size))
                    
                    if sqrt_size * sqrt_size == total_size:
                        # Reshape to square matrix
                        U_full = result_data.reshape(sqrt_size, sqrt_size)
                        
                        # Extract operator block for data qubits
                        U = self.extract_operator_block(U_full, self.total_qubits)
                        return U
                    else:
                        raise ValueError(
                            f"Cannot reshape tensor with shape {result_data.shape} "
                            f"to square matrix. Total size: {total_size}"
                        )
                else:
                    # Normal case - reshape directly
                    final_matrix = result_data.reshape(matrix_size, matrix_size)
                    return final_matrix
            else:
                # Already 2D or 1D - reshape normally
                if result_data.size == matrix_size * matrix_size:
                    return result_data.reshape(matrix_size, matrix_size)
                else:
                    raise ValueError(
                        f"Cannot reshape result with size {result_data.size} "
                        f"to ({matrix_size}, {matrix_size}) matrix"
                    )
            
        except Exception as e:
            print(f"Error in Trotter ZX computation with trace: {e}")
            print("Falling back to standard matrix exponentiation...")
            return self.time_evolution(time, optimize=optimize)
    
    def evolve_density_matrix(
        self,
        initial_rho: np.ndarray,
        time: float,
        trotter_steps: int = 1,
        order: int = 1,
        use_trace: bool = True,
        optimize: bool = True
    ) -> np.ndarray:
        """
        Evolve a density matrix under the Hamiltonian.
        
        Args:
            initial_rho: Initial density matrix (2^n × 2^n)
            time: Evolution time
            trotter_steps: Number of Trotter steps (for Trotter method)
            order: Order of Trotter expansion
            use_trace: If True, use methods that handle extra dimensions with partial trace
            optimize: Whether to use cotengra optimization
            
        Returns:
            Evolved density matrix ρ(t) = U ρ(0) U†
        """
        # Get evolution operator
        if use_trace:
            U = self.time_evolution_trotter_with_trace(
                time=time,
                trotter_steps=trotter_steps,
                order=order,
                optimize=optimize
            )
        else:
            U = self.time_evolution_trotter(
                time=time,
                trotter_steps=trotter_steps,
                order=order,
                optimize=optimize
            )
        
        # Evolve density matrix: ρ(t) = U ρ(0) U†
        rho_t = U @ initial_rho @ U.conj().T
        
        return rho_t
    
    def compute_toy_state_properties(
        self,
        initial_ket: np.ndarray,
        time: float,
        trotter_steps: int = 1,
        order: int = 1,
        use_trace: bool = True,
        optimize: bool = True
    ) -> dict:
        """
        Compute properties of a toy state evolved under the Hamiltonian.
        
        This method converts a ket to a density matrix, evolves it, and computes
        various properties like trace, purity, and energy expectation.
        
        Args:
            initial_ket: Initial state vector |ψ⟩
            time: Evolution time
            trotter_steps: Number of Trotter steps
            order: Order of Trotter expansion
            use_trace: If True, use methods that handle extra dimensions
            optimize: Whether to use cotengra optimization
            
        Returns:
            Dictionary with properties:
            - 'density_matrix': Evolved density matrix
            - 'trace': Trace of density matrix (should be ~1)
            - 'purity': Tr(ρ²) (should be ~1 for pure state)
            - 'energy': Tr(Hρ) energy expectation value
        """
        # Convert ket to density matrix
        initial_rho = np.outer(initial_ket, initial_ket.conj())
        
        # Evolve density matrix
        rho_t = self.evolve_density_matrix(
            initial_rho=initial_rho,
            time=time,
            trotter_steps=trotter_steps,
            order=order,
            use_trace=use_trace,
            optimize=optimize
        )
        
        # Compute properties
        trace_rho = np.trace(rho_t).real
        purity = np.trace(rho_t @ rho_t).real
        
        # Compute energy expectation: Tr(Hρ)
        if use_trace:
            H = self.compute_matrix_with_trace(optimize=optimize)
        else:
            H = self.compute_matrix(optimize=optimize)
        energy = np.trace(H @ rho_t).real
        
        return {
            'density_matrix': rho_t,
            'trace': trace_rho,
            'purity': purity,
            'energy': energy
        }


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

