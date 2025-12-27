# Missing Analysis: Plots and Visualizations

This document identifies what might be missing in terms of plots, visualizations, and analysis tools based on a review of the repository.

## Existing Plots/Visualizations

### Currently Implemented:
1. **`plot_collective_decay_spectrum()`** - Eigenvalue spectrum with superradiant/subradiant classification
2. **Basic eigenvalue plots** in notebooks:
   - Complex plane scatter plots
   - Magnitude bar charts
   - Log-scale magnitude plots
3. **Simple decay evolution plot** in `example_usage.ipynb` (log of absolute values over time)
4. **Basic bar chart** in `test_subradiance.py` for eigenvalues

## Potentially Missing Visualizations

### 1. Time Evolution & Decay Dynamics
**Status**: Partially implemented but could be enhanced

**Missing:**
- **Population decay plots**: Plot of excited state population vs time for different initial states
- **Multi-state evolution**: Compare decay of different initial states (e.g., |11...1⟩ vs |10...0⟩)
- **Decay rate extraction**: Fit exponential decay and extract effective decay rates
- **Comparison plots**: ZX method vs numpy method for time evolution
- **Phase evolution**: Plot of phase dynamics during decay
- **Density matrix evolution**: Visualization of density matrix elements over time

**Existing code:**
- `compute_subradiance_decay()` exists but only returns simple population
- `evolve_density_matrix()` exists but no plotting wrapper

### 2. Trotter Expansion Analysis
**Status**: Missing visualization tools

**Missing:**
- **Trotter error plots**: Error vs number of Trotter steps
- **Convergence analysis**: Compare first-order vs second-order Trotter
- **Time step analysis**: Optimal step size determination plots
- **Energy conservation**: Check if energy is conserved during Trotter evolution
- **Comparison with exact evolution**: ZX Trotter vs matrix exponentiation

**Existing code:**
- `build_trotter_graph()` exists
- `time_evolution_trotter()` methods exist but no error analysis

### 3. F_jk Matrix Visualization
**Status**: Missing

**Missing:**
- **Heatmap of F_jk matrix**: Visualize coupling strengths between atoms
- **Distance vs coupling plots**: Plot F_jk vs inter-atomic distance
- **Spatial correlation plots**: Show how coupling depends on geometry
- **Complex F_jk visualization**: Real and imaginary parts separately

**Existing code:**
- `compute_F_jk()` and `compute_F_jk_equation5()` exist
- No visualization functions

### 4. Atom Position Visualization
**Status**: Missing

**Missing:**
- **3D/2D atom position plots**: Visualize spatial arrangement of atoms
- **Distance matrix heatmap**: Show all pairwise distances
- **Grid visualization**: For grid-based setups, show the grid structure
- **Position vs coupling correlation**: Scatter plot of distance vs F_jk

**Existing code:**
- `setup_positions_3d_grid()` and `setup_positions_2d_grid()` exist
- No visualization functions

### 5. Superradiance/Subradiance Analysis
**Status**: Partially implemented

**Missing:**
- **State classification plots**: Visualize which states are superradiant/subradiant
- **Enhancement/suppression factor plots**: Bar charts or scatter plots
- **Eigenvalue distribution**: More detailed histograms with statistical analysis
- **Comparison across different geometries**: How does spectrum change with atom arrangement?
- **Decay rate distribution**: Histogram of all decay rates

**Existing code:**
- `analyze_collective_decay_spectrum()` exists
- `plot_collective_decay_spectrum()` exists but could be enhanced

### 6. Method Comparison Plots
**Status**: Missing

**Missing:**
- **ZX vs Numpy eigenvalue comparison**: Scatter plots showing agreement
- **Error analysis plots**: Difference between methods vs system size
- **Performance comparison**: Computation time vs system size for different methods
- **Accuracy vs speed trade-offs**: Visualize when to use which method

**Existing code:**
- `compare_eigenvalue_methods()` exists but no plotting

### 7. ZX Graph Visualizations
**Status**: Basic visualization exists, but could be enhanced

**Missing:**
- **Systematic graph saving**: Save ZX graphs to files for documentation
- **Graph simplification visualization**: Before/after simplification
- **Trotter graph visualization**: Show the structure of Trotterized evolution
- **Graph statistics**: Number of vertices, edges, complexity metrics
- **Comparison of different Hamiltonians**: Side-by-side graph comparisons

**Existing code:**
- `zx.draw()` is used in notebooks but not systematically saved
- `build_graph()` and `build_trotter_graph()` exist

### 8. Energy and Observable Plots
**Status**: Missing

**Missing:**
- **Energy expectation vs time**: Plot ⟨H⟩(t) during evolution
- **Purity plots**: Tr(ρ²) vs time to check if state remains pure
- **Entanglement measures**: If applicable, plot entanglement entropy
- **Observable evolution**: Plot expectation values of various observables

**Existing code:**
- `expectation_value()` exists
- `compute_toy_state_properties()` exists but no plotting wrapper

### 9. Parameter Sweep Visualizations
**Status**: Missing

**Missing:**
- **Decay rate vs eigenvalues**: How does spectrum change with decay rate?
- **Detuning effects**: Plot eigenvalues vs detuning parameter
- **Wavelength dependence**: How does λ affect the spectrum?
- **Atom number scaling**: How do properties scale with number of atoms?
- **Grid spacing effects**: How does spacing multiplier m affect results?

### 10. Paper Comparison Plots
**Status**: Missing

**Missing:**
- **Reproduce paper figures**: Direct comparison with figures from referenced papers
- **Validation plots**: Show that results match expected theoretical predictions
- **Parameter space exploration**: Systematic exploration matching paper studies

### 11. Diagnostic and Debugging Plots
**Status**: Partially exists

**Missing:**
- **Tensor network structure visualization**: Already exists in `w_stuff.ipynb` but not as reusable function
- **Dimension mismatch diagnostics**: Visualize where dimension issues occur
- **Contraction path visualization**: Show optimal contraction paths
- **Memory usage plots**: Track memory consumption vs system size

**Existing code:**
- Some diagnostic code in `dimension_analysis.py`
- Tensor network drawing in notebooks

### 12. Missing Core Function
**Status**: Critical issue

**Missing:**
- **`compute_matrix()` method**: Referenced in code but not defined
  - Called in: `compute_eigenvalues()`, `time_evolution_numpy()`, `expectation_value()`
  - Should probably be an alias for `compute_matrix_with_trace()` or a wrapper

## Recommendations

### High Priority:
1. **Add `compute_matrix()` method** - Fix the missing method that's referenced
2. **Time evolution plotting functions** - Essential for understanding decay dynamics
3. **F_jk matrix visualization** - Important for understanding coupling structure
4. **Atom position visualization** - Helpful for understanding geometry effects

### Medium Priority:
5. **Trotter error analysis plots** - Important for validating Trotter method
6. **Method comparison visualizations** - Validate ZX calculus implementation
7. **Enhanced superradiance analysis plots** - Better understanding of collective effects

### Low Priority:
8. **Parameter sweep visualizations** - Useful for systematic studies
9. **Paper comparison plots** - Validation against literature
10. **Advanced diagnostic plots** - For debugging and optimization

## Notes

- The codebase has good analysis functions but lacks comprehensive plotting wrappers
- Many functions return data that could be visualized but no plotting functions exist
- Notebooks have some plots but they're not organized into reusable functions
- Consider creating a `visualization.py` or `plots.py` module to centralize plotting functions

