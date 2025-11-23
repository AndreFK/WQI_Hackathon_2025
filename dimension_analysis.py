"""
Diagnostic script to analyze dimension handling in tensor network contraction.
This will help identify where dimension mismatches occur.
"""

import numpy as np

def analyze_tensor_network_dimensions(tensor_network, total_qubits):
    """
    Analyze the dimensions of a tensor network and its contraction result.
    
    Args:
        tensor_network: Quimb TensorNetwork object
        total_qubits: Expected number of qubits
    """
    print("="*70)
    print("TENSOR NETWORK DIMENSION ANALYSIS")
    print("="*70)
    
    # Get outer indices
    outer_inds = tensor_network.outer_inds()
    n_outer = len(outer_inds)
    
    print(f"\nExpected qubits: {total_qubits}")
    print(f"Number of outer indices: {n_outer}")
    print(f"Expected outer indices (inputs + outputs): {2 * total_qubits}")
    
    if n_outer != 2 * total_qubits:
        print(f"\n⚠️  WARNING: Mismatch! Expected {2 * total_qubits} outer indices, got {n_outer}")
        print("This suggests extra dimensions from W-state structure or other sources.")
        print("We may need to trace out auxiliary qubits or handle dimensions differently.")
    else:
        print(f"\n✓ Outer indices match expected: {n_outer} = 2 × {total_qubits}")
    
    # Contract and check result shape
    print("\n" + "-"*70)
    print("CONTRACTING TENSOR NETWORK")
    print("-"*70)
    
    try:
        result = tensor_network.contract(all, output_inds=outer_inds)
        
        if hasattr(result, 'data'):
            result_data = result.data
        else:
            result_data = result
            
        print(f"\nResult type: {type(result)}")
        print(f"Result shape: {result_data.shape}")
        print(f"Result size: {result_data.size}")
        
        # Expected size for a (2^n, 2^n) matrix
        expected_size = (2 ** total_qubits) ** 2
        print(f"\nExpected size for ({2**total_qubits}, {2**total_qubits}) matrix: {expected_size}")
        
        if result_data.size != expected_size:
            print(f"\n⚠️  WARNING: Size mismatch!")
            print(f"Result has {result_data.size} elements, expected {expected_size}")
            
            # Try to understand the shape
            if result_data.ndim == 1:
                print(f"\nResult is 1D array. Can we reshape to matrix?")
                # Try to find square root
                sqrt_size = int(np.sqrt(result_data.size))
                if sqrt_size * sqrt_size == result_data.size:
                    print(f"  ✓ Can reshape to ({sqrt_size}, {sqrt_size})")
                    print(f"  But expected ({2**total_qubits}, {2**total_qubits})")
                else:
                    print(f"  ✗ Cannot reshape to square matrix")
            elif result_data.ndim > 1:
                print(f"\nResult has {result_data.ndim} dimensions")
                print(f"Shape: {result_data.shape}")
                
                # Check if it's a tensor with 2*n dimensions
                if all(d == 2 for d in result_data.shape):
                    n_dims = len(result_data.shape)
                    print(f"\nAll dimensions are size 2. Total dimensions: {n_dims}")
                    if n_dims == 2 * total_qubits:
                        print(f"✓ This matches expected: 2 × {total_qubits} = {n_dims} dimensions")
                        print(f"  Need to reshape: first {total_qubits} dims → rows, last {total_qubits} dims → columns")
                        print(f"  Reshape to: ({2**total_qubits}, {2**total_qubits})")
                    else:
                        print(f"⚠️  Dimension count mismatch: {n_dims} != 2 × {total_qubits}")
        else:
            print(f"\n✓ Size matches expected!")
            
    except Exception as e:
        print(f"\n✗ Error during contraction: {e}")
        import traceback
        traceback.print_exc()

def analyze_density_matrix_evolution():
    """
    Analyze how density matrices should be evolved.
    """
    print("\n" + "="*70)
    print("DENSITY MATRIX EVOLUTION ANALYSIS")
    print("="*70)
    
    print("\nFor a unitary operator U and density matrix ρ:")
    print("  ρ(t) = U @ ρ(0) @ U.conj().T")
    print("\nDimensions:")
    print("  U: (2^n, 2^n)")
    print("  ρ(0): (2^n, 2^n)")
    print("  ρ(t): (2^n, 2^n)")
    print("\nIf U comes from tensor network with shape (2, 2, ..., 2):")
    print("  - Need to reshape U from tensor to matrix first")
    print("  - Then apply: U @ ρ @ U.conj().T")
    
    print("\n" + "-"*70)
    print("PARTIAL TRACE CONSIDERATIONS")
    print("-"*70)
    print("\nIf tensor network has extra dimensions (e.g., from W-state structure):")
    print("  - May need to trace out auxiliary qubits")
    print("  - Or properly identify which indices correspond to data qubits")
    print("  - Input/output indices should match data qubits, not auxiliary qubits")

if __name__ == "__main__":
    print("This is a diagnostic module.")
    print("Import and use analyze_tensor_network_dimensions() to check your tensor network.")
    analyze_density_matrix_evolution()

