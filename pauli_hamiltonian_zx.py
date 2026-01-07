"""
Pauli Hamiltonian Summation and Time Evolution using ZX Calculus with W-states.

This module implements the method from:
"How to Sum and Exponentiate Hamiltonians in ZX Calculus"
by Razin A. Shaikh, Quanlong Wang, and Richie Yeung

The code provides functionality to:
1. Sum Pauli string Hamiltonians using ZX calculus with W inputs/outputs
2. Compute time evolution operators exp(-iHt)

References:
- "How to Sum and Exponentiate Hamiltonians in ZX Calculus" (arXiv:2212.04462)
- "Analytical and numerical study of subradiance-only collective decay from atomic ensembles"
- "A numerical study of the spatial coherence of light in collective spontaneous emission"
- "Light Matter Interaction ZXW calculus" (for future integration)
"""

import pyzx as zx
from pyzx.symbolic import new_var, Poly
import sympy as sp
import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from pyzx.graph.graph import Graph
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
            pauli_strings: Ex: [(0.5, ["X0", "X1"]), (-0.3, ["Y2", "Z3"])]
        """
        self.pauli_strings = pauli_strings
        self.total_qubits = self._compute_total_qubits()
        self.main_graph = None
        self.top_graph = None
        self.tot_graph = None
        self.tensor_network = None
        self.optimizer = None
        self.x_input_vertex = None
        self.zs = None
        self.trotter = None
        self.trotter_steps = 0
        
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
        pi_over_2 = Fraction(1, 2)  # π/2 in ZX calculus
        # Create input boundaries
        inps = []
        for q in range(self.total_qubits):
            in_vertex = main_graph.add_vertex(zx.VertexType.BOUNDARY, qubit=q, row=0)
            inps.append(in_vertex)
        main_graph.set_inputs(inps)
        
        current_row = 1
        z_vertices_to_connect = []
        zs=[]
        current_row = 1  
        z_vertices_to_connect = []
        for term in self.pauli_strings:
            phase = term[0]
            gates = term[1]
            curr_list = []
            row_increment = 0
            first_qubit_index = 0
            for gate in gates:
                gate_type = gate[0]
                qubit_index = int(gate[1:])

                if qubit_index != first_qubit_index:
                    row_increment = 0
                

                if gate_type == 'X':
                    first_hadamard = main_graph.add_vertex(zx.VertexType.H_BOX, qubit=qubit_index, row=current_row+row_increment)
                    z_vertex = main_graph.add_vertex(zx.VertexType.Z, qubit=qubit_index, row=current_row + 1+row_increment)
                    second_hadamard = main_graph.add_vertex(zx.VertexType.H_BOX, qubit=qubit_index, row=current_row + 2)

                    main_graph.add_edge((first_hadamard, z_vertex))
                    main_graph.add_edge((z_vertex, second_hadamard))
                    
                    row_increment += 3
                    
                    curr_list.append(z_vertex)
                elif gate_type == 'Z':
                    z_vertex = main_graph.add_vertex(zx.VertexType.Z, qubit=qubit_index, row=current_row)
                    row_increment += 1
                    
                    curr_list.append(z_vertex)
                elif gate_type == 'Y':
                    x_vertex_one = main_graph.add_vertex(zx.VertexType.X, qubit=qubit_index, row=current_row+row_increment, phase=pi_over_2)
                    z_vertex = main_graph.add_vertex(zx.VertexType.Z, qubit=qubit_index, row=current_row + 1+row_increment)
                    x_vertex_two = main_graph.add_vertex(zx.VertexType.X, qubit=qubit_index, row=current_row + 2+row_increment, phase=-pi_over_2)
                
                    main_graph.add_edge((x_vertex_one, z_vertex))
                    main_graph.add_edge((z_vertex, x_vertex_two))
                    
                    row_increment += 3
                    
                    curr_list.append(z_vertex)
                
            corresponding_x = main_graph.add_vertex(zx.VertexType.X, qubit=-1, row=current_row)
            corresponding_z = main_graph.add_vertex(zx.VertexType.Z_BOX, qubit=-2, row=current_row, phase=phase)
            zx.utils.set_z_box_label(main_graph, corresponding_z, phase)
            for z_vertex in curr_list:
                main_graph.add_edge((corresponding_x, z_vertex))
                main_graph.add_edge((corresponding_x, corresponding_z),edgetype=zx.EdgeType.HADAMARD)
                
            z_vertices_to_connect.append(curr_list)
            zs.append(corresponding_z)

            current_row += row_increment
        
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
        self.zs = zs
        for q in range(self.total_qubits):
            #find all vertices for qubit q
            vertices_on_qubit = [v for v in main_graph.vertices() if main_graph.qubit(v) == q]
        edges = main_graph.edge_set()
    
        # run through vertices and check wich are not connected
        for i in range(len(vertices_on_qubit)):
            v1 = vertices_on_qubit[i]
            
            if i+1 >= len(vertices_on_qubit):
                break
            
            v2 = vertices_on_qubit[i+1]
            if (v1, v2) not in edges and (v2, v1) not in edges:
                main_graph.add_edge((v1, v2))

        row_range = range(len(z_vertices_to_connect))
        argsorted = np.argsort([z[0] for z in z_vertices_to_connect])
        
        row_range = [row_range[i] for i in argsorted]
        
        pauli_string_x_vertex = []
        

        h = list(main_graph.vertices())[-1]

        top_zs = []
        for x_v in pauli_string_x_vertex:
            top_zs.append(main_graph.add_vertex(zx.VertexType.Z, qubit=main_graph.qubit(x_v)-1, row=main_graph.row(x_v)))
        return main_graph
    def build_trotter_graph(self, time: float, steps: int) -> zx.Graph:
        """
        Build the Trotterized ZX graph for time evolution.
        Args:
            time: Evolution time
        Returns:
            ZX graph representing the Trotterized Hamiltonian
        """
        # Build the main graph first
        if self.main_graph is None:
            self.main_graph = self._build_main_graph()
        graphToAppend = self.main_graph.copy()
        for z in self.zs:
            
            weight = graphToAppend.phase(z)
            zx.utils.set_z_box_label(graphToAppend, z, 0) 
            graphToAppend.set_type(z, zx.VertexType.Z)
            fractional_phase = -weight * time / steps
            graphToAppend.set_phase(z, fractional_phase)
        
        appended_graph = graphToAppend.copy()
        for i in range(steps-1):
            appended_graph.compose(graphToAppend.copy())
        return appended_graph
    
    
    
    def build_single_trotter_symbolic(self, steps: int) -> zx.Graph:
        """
        Returns a symbolic ZX graph where 't' is a variable.
        """
        print("#"*10,"Function [PauliHamiltonianZX.build_single_trotter_symbolic()] is not implemented correctly yet.","#"*10)
        # 1. Create the symbolic variable 't'
        t = new_var('t', 'continuous')
        
        graphToAppend = self.main_graph.copy()
        self.trotter_steps = steps

        for i, z in enumerate(self.zs[:3]):
            raw_phase = graphToAppend.phase(z)
            node_type = graphToAppend.type(z)
            print(f"Node {z}:")
            print(f"  - Type: {node_type} (1=Z, 2=X, 3=H_BOX)")
            print(f"  - Phase (from .phase()): {raw_phase}")

        for z in self.zs:
            weight = graphToAppend.phase(z)
            # Step 2
            # TRANSFORM: Convert to Z-spider
            zx.utils.set_z_box_label(graphToAppend, z, 0) 
            graphToAppend.set_type(z, zx.VertexType.Z)

            # 3. Now that it is a Z-spider, it accepts a phase (and Poly objects)
            symbolic_phase = -weight * t / self.trotter_steps
            symbolic_phase = Poly([-weight / self.trotter_steps, 0], [t])
            graphToAppend.set_phase(z, symbolic_phase)
            print(f"DEBUG: After setting symbolic phase:{z}")
            
        
        return graphToAppend


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
        if self.main_graph is None:
            self.main_graph = self._build_main_graph()
        top_graph = zx.Graph()
        root_x_vertex = top_graph.add_vertex(
            zx.VertexType.X, qubit=-3, row=1, phase=1
        )
        w_input = top_graph.add_vertex(
            zx.VertexType.W_INPUT, qubit=-3, row=1
        )
        w_output = top_graph.add_vertex(
            zx.VertexType.W_OUTPUT, qubit=-3, row=1
        )
        
        top_graph.add_edge((root_x_vertex, w_input))
        top_graph.add_edge((w_output, w_input))
        # Tensor the graphs
        self.tot_graph =  top_graph @ self.main_graph
        w_output = w_output
        # zx.draw(self.tot_graph)
        # Connect X vertices to Z vertices
        for z in self.zs:
            z = z+top_graph.num_vertices()
            self.tot_graph.add_edge((w_output, z))
            
        self.tot_graph.set_qubit(w_output, -3)
        self.tot_graph.set_qubit(w_input, -4)
        self.tot_graph.set_row(root_x_vertex, -5)
        return self.tot_graph
    
    def simplify_graph(self, normalize_rows: bool = True) -> zx.Graph:
        """
        Simplify the ZX graph using ZX calculus rules.
        Args:
            normalize_rows: If True, normalize row positions to start from 0 and be compact.
                          This reduces scrolling in visualizations.
        Returns:
            Simplified ZX graph
        """
        if self.tot_graph is None:
            self.build_graph()
        
        zx.hsimplify.from_hypergraph_form(self.tot_graph)
        zx.simplify.full_reduce(self.tot_graph)
        
        if normalize_rows:
            self.tot_graph.normalize()
        return self.tot_graph
    
    
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

        #[TODO: ASSUMES OUTPUTS/INPUTS ARE IN SAME ORDER]
        graph1_outputs = list(graph1.outputs())
        graph2_inputs = list(graph2.inputs())
        
        # Simple approach: connect by index (assuming same order)
        connections_made = 0
        min_len = min(len(graph1_outputs), len(graph2_inputs))
        
        
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
        
        new_inputs = [vertex_map1[v] for v in graph1.inputs() if vertex_map1[v] in composed.vertices()]
        new_outputs = [vertex_map2[v] for v in graph2.outputs() if vertex_map2[v] in composed.vertices()]
        
        composed.set_inputs(new_inputs)
        composed.set_outputs(new_outputs)
        return composed