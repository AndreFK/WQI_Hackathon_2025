# Bugs and Issues in the Code

This document identifies bugs, errors, and potential issues found in the codebase.

## Critical Bugs (Code Won't Run)

### 1. Missing `compute_matrix()` Method
**Location**: `pauli_hamiltonian_zx.py`
**Severity**: CRITICAL - Code will crash

**Issue**: The method `compute_matrix()` is called in multiple places but never defined:
- Line 397: `matrix = self.compute_matrix()` in `compute_eigenvalues()`
- Line 660: `H = self.compute_matrix(optimize=optimize)` in `time_evolution_numpy()`
- Line 701: `H = self.compute_matrix()` in `expectation_value()`
- Line 1139: `H = self.compute_matrix(optimize=optimize)` in `compute_toy_state_properties()`

**Expected**: Should probably be an alias or wrapper for `compute_matrix_with_trace()`:
```python
def compute_matrix(self, optimize: bool = True) -> np.ndarray:
    return self.compute_matrix_with_trace(optimize=optimize)
```


### 2. Missing `time_evolution()` Method
**Location**: `pauli_hamiltonian_zx.py`
**Severity**: CRITICAL - Code will crash

**Issue**: Called in `evolve_state()` at line 683:
```python
U = self.time_evolution(time)
```

**Expected**: Should probably call `time_evolution_numpy()` or `time_evolution_trotter()`:
```python
def time_evolution(self, time: float, optimize: bool = True) -> np.ndarray:
    return self.time_evolution_numpy(time, optimize=optimize)
```

### 3. Missing Trotter Evolution Methods
**Location**: `pauli_hamiltonian_zx.py`
**Severity**: CRITICAL - Code will crash

**Issue**: Methods referenced but not defined:
- `time_evolution_trotter()` - called at line 1076
- `time_evolution_trotter_with_trace()` - called at line 1069

**Expected**: These methods should exist to compute Trotterized time evolution.

## Logic Errors

### 4. Incorrect Sign in Equation 4 Hamiltonian
**Location**: `cor_decay_zxw.py`, line 199-200
**Severity**: HIGH - Results will be incorrect

**Issue**: The documentation says equation 4 should be `(F_jk/2) * (X_j X_k - Y_j Y_k)` (MINUS Y_j Y_k), but the code has:
```python
pauli_strings.append((-f_jk/2, [f"X{j}", f"X{k}"])) 
pauli_strings.append((-f_jk/2, [f"Y{j}", f"Y{k}"]))  # Should be +f_jk/2 for minus
```

**Problem**: If the decomposition is `(F_jk/2) * (X_j X_k - Y_j Y_k)`, then:
- X_j X_k term should be: `(F_jk/2) * X_j X_k`
- Y_j Y_k term should be: `(-F_jk/2) * Y_j Y_k`

But the code has both as `-f_jk/2`, which would give `(-F_jk/2) * (X_j X_k + Y_j Y_k)`.

**Fix**: The Y term should be `(+f_jk/2, [f"Y{j}", f"Y{k}"])` to get the minus sign in the decomposition.

### 5. Phase Modification Before Reading in Trotter
**Location**: `pauli_hamiltonian_zx.py`, line 214
**Severity**: HIGH - Trotter evolution incorrect

**Issue**: In `build_trotter_graph()`:
```python
for z in self.zs:
    zx.utils.set_z_box_label(graphToAppend, z, 0)
    graphToAppend.set_type(z, zx.VertexType.Z)
    graphToAppend.set_phase(z, -graphToAppend.phase(z) * time / (steps))
```

**Problem**: The code reads `graphToAppend.phase(z)` AFTER setting the label to 0 and changing the type. The phase might be lost or incorrect.

**Expected**: Should read the original phase before modifying, or use the coefficient from `self.pauli_strings`.

### 6. Incorrect Trotter Implementation
**Location**: `pauli_hamiltonian_zx.py`, `build_trotter_graph()`
**Severity**: HIGH - Trotter method incorrect

**Issue**: The Trotter implementation composes the same graph multiple times, but:
1. It modifies phases in place on `graphToAppend`
2. Then composes it with itself multiple times
3. This doesn't properly implement Trotter: `exp(-iHt) ≈ [∏_j exp(-iH_j * t/n)]^n`

**Problem**: The current implementation doesn't properly separate the Hamiltonian terms. Each term should be exponentiated separately, then composed.

### 7. Variable Shadowing in `build_graph()`
**Location**: `pauli_hamiltonian_zx.py`, line 294
**Severity**: MEDIUM - Potential confusion

**Issue**:
```python
self.tot_graph = top_graph @ self.main_graph
w_output = w_output  # This line does nothing
```

**Problem**: `w_output` is reassigned to itself, which is a no-op. Then later at line 297, it tries to use `w_output` but it's from the original `top_graph`, not the tensored graph.

**Expected**: Should get the correct `w_output` vertex from `self.tot_graph` after tensoring.

### 8. Incorrect Vertex Indexing After Tensor
**Location**: `pauli_hamiltonian_zx.py`, line 298
**Severity**: HIGH - Graph connections incorrect

**Issue**:
```python
for z in self.zs:
    z = z+top_graph.num_vertices()  # Modifies loop variable!
    self.tot_graph.add_edge((w_output, z))
```

**Problems**:
1. Modifying loop variable `z` inside the loop is bad practice
2. The offset calculation might be incorrect - vertices are renumbered when tensoring
3. `w_output` is from the original `top_graph`, not the tensored graph

**Expected**: Should properly map vertices from both graphs after tensoring.

## Code Quality Issues

### 9. Duplicate Code in `_build_main_graph()`
**Location**: `pauli_hamiltonian_zx.py`, lines 82-86 and 161-175
**Severity**: LOW - Code duplication

**Issue**: The code for connecting vertices on qubit lines appears twice:
- Lines 146-156: First implementation
- Lines 161-175: Duplicate implementation (almost identical)

**Problem**: Redundant code that should be removed.

### 10. Unused Variables
**Location**: `pauli_hamiltonian_zx.py`
**Severity**: LOW - Code clutter

**Issues**:
- Line 84: `zs=[]` is defined but `self.zs` is set later
- Line 85: `current_row = 1` is set twice (lines 82 and 85)
- Line 182: `pauli_string_x_vertex = []` is created but never used
- Line 185: `h = list(main_graph.vertices())[-1]` is computed but never used
- Line 187-189: `top_zs` loop uses empty `pauli_string_x_vertex` list
- Line 207: `inps = []` is created but never used in `build_trotter_graph()`
- Line 208: `appended_graph = None` is set but immediately overwritten

### 11. Incomplete Implementation
**Location**: `pauli_hamiltonian_zx.py`, `build_trotter_graph()`
**Severity**: MEDIUM - Function incomplete

**Issue**: Comments suggest incomplete implementation:
- Line 224: `# Add each Pauli term as a single term exponential` - but nothing is added
- The function doesn't properly implement Trotter expansion for multiple terms

### 12. Dimension Mismatch in Tensor Network
**Location**: `w_stuff.ipynb` (and potentially in code)
**Severity**: HIGH - Runtime error

**Issue**: From notebook output:
```
ValueError: cannot reshape array of size 2147483648 into shape (1024,1024)
```

**Problem**: The tensor network contraction produces a result with wrong dimensions. The result has 31 dimensions of size 2 each (2^31 elements), but expected 2^10 × 2^10 = 1024 × 1024.

**Cause**: Likely due to extra dimensions from W-state structure not being properly traced out, or incorrect handling of tensor network indices.

### 13. Incorrect Graph Composition Order
**Location**: `pauli_hamiltonian_zx.py`, `build_trotter_graph()`, line 222
**Severity**: MEDIUM - May cause incorrect results

**Issue**:
```python
appended_graph = self._compose_graphs(graphToAppend, appended_graph)
```

**Problem**: The order might be wrong. If `graphToAppend` is the first step and `appended_graph` is the accumulated result, the composition order should be checked. Typically Trotter is: `U_n @ U_{n-1} @ ... @ U_1`.

### 14. Missing Error Handling
**Location**: Multiple locations
**Severity**: MEDIUM - Poor error messages

**Issues**:
- `compute_matrix_with_trace()` has try-except but falls back silently
- No validation that `self.zs` is not None before using in `build_trotter_graph()`
- No check that `self.main_graph` is built before accessing `self.zs`

### 15. Inconsistent Method Naming
**Location**: `pauli_hamiltonian_zx.py`
**Severity**: LOW - Confusion

**Issue**: 
- `time_evolution_numpy()` - uses numpy
- `time_evolution_trotter()` - should use Trotter (but missing)
- `time_evolution()` - generic name (but missing)

**Problem**: Inconsistent naming makes it unclear which method to use.

## Mathematical/Physical Issues

### 16. Equation 4 vs Subradiance Hamiltonian Confusion
**Location**: Multiple files
**Severity**: MEDIUM - Conceptual error

**Issue**: There are two different Hamiltonians:
1. **Subradiance** (from `pauli_hamiltonian_zx.py`): `H = Σ_{j<k} (F_jk/2) * (X_j X_k + Y_j Y_k)` (PLUS)
2. **Equation 4** (from `cor_decay_zxw.py`): `H = (F_jk/2) * (X_j X_k - Y_j Y_k)` (MINUS)

**Problem**: The code in `create_equation4_hamiltonian()` doesn't correctly implement the minus sign (see issue #4).

### 17. F_jk Matrix Not Symmetric
**Location**: `cor_decay_zxw.py`, `compute_F_jk_equation5()`
**Severity**: MEDIUM - May cause issues

**Issue**: The function only sets `F[j, k]` and `F[k, j] = F[j, k]` for `j < k`, but doesn't handle the diagonal properly in all cases. Also, the loop only goes `for k in range(j, N)`, so `F[k, j]` might not be set for all cases.

**Fix**: Should ensure `F[k, j] = F[j, k]` for all `j != k`.

### 18. Incorrect Diagonal Terms in Equation 4
**Location**: `cor_decay_zxw.py`, line 202
**Severity**: MEDIUM - May be incorrect

**Issue**: 
```python
elif j == k:
    pauli_strings.append((gamma_diag, [f"Z{j}"]))  # Diagonal terms are zero in off-diagonal part
```

**Problem**: The comment says "Diagonal terms are zero in off-diagonal part" but the code adds Z terms. This seems contradictory. If it's the "off-diagonal part", why add diagonal Z terms?

## Summary

### Critical (Must Fix):
1. Missing `compute_matrix()` method
2. Missing `time_evolution()` method  
3. Missing `time_evolution_trotter()` methods

### High Priority (Results Incorrect):
4. Incorrect sign in Equation 4 Hamiltonian
5. Phase modification bug in Trotter
6. Incorrect Trotter implementation
8. Incorrect vertex indexing after tensor
12. Dimension mismatch in tensor network

### Medium Priority:
7. Variable shadowing in build_graph
11. Incomplete Trotter implementation
13. Graph composition order
14. Missing error handling
16. Equation 4 vs subradiance confusion
17. F_jk matrix symmetry
18. Diagonal terms in equation 4

### Low Priority:
9. Duplicate code
10. Unused variables
15. Inconsistent naming

## Recommendations

1. **Immediate**: Add the missing methods (`compute_matrix()`, `time_evolution()`, etc.)
2. **High Priority**: Fix the Equation 4 sign issue and Trotter implementation
3. **Testing**: Add unit tests to catch these issues
4. **Refactoring**: Clean up duplicate code and unused variables
5. **Documentation**: Clarify the difference between Equation 4 and subradiance Hamiltonians

