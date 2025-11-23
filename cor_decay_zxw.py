"""
Collective Decay Simulation using ZXW Calculus Method

This module implements the collective decay simulation from cor_decay1.m
using the ZXW calculus method for Hamiltonian construction.

The Hamiltonian is computed using equations 4 and 5 from paper 2408:
- Equation 4: Ĥ^jk = F_jk σ̂_+^j σ̂_-^k + F_kj σ̂_-^j σ̂_+^k
  Decomposes as: (F_jk/2) * (X_j X_k - Y_j Y_k)
- Equation 5: F_jk coupling coefficients from dipole-dipole interactions

The main goal is to compute eigenvalues of the off-diagonal Hamiltonian.
"""

import numpy as np
from typing import Tuple, List
import sys
import os

# Import ZXW functions directly (we're already in the same directory)
try:
    from pauli_hamiltonian_zx import PauliHamiltonianZX, create_collective_decay_hamiltonian
except ImportError:
    print("Warning: Could not import ZXW functions. ZXW method will not be available.")
    PauliHamiltonianZX = None
    create_collective_decay_hamiltonian = None


def compute_hamiltonian_matlab_style(
    lambda_val: float,
    del_omega: float,
    gam: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    N: int
) -> np.ndarray:
    """
    Compute the Hamiltonian matrix using the exact formula from cor_decay1.m
    
    This computes the non-Hermitian Hamiltonian in the single-excitation subspace.
    
    Args:
        lambda_val: Wavelength
        del_omega: Lamb shift (detuning)
        gam: Line width (decay rate)
        x: Array of x coordinates (N,)
        y: Array of y coordinates (N,)
        z: Array of z coordinates (N,)
        N: Number of atoms
        
    Returns:
        Hamiltonian matrix (N, N) as complex numpy array
    """
    H = np.zeros((N, N), dtype=complex)
    K = 2 * np.pi / lambda_val
    
    # Assigning values to hamiltonian
    for j in range(N):
        for k in range(N):
            if j != k:
                # Compute distance components
                dz = z[j] - z[k]
                dy = y[j] - y[k]
                dx = x[j] - x[k]
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                
                if r < 1e-10:  # Avoid division by zero
                    a = 0.0
                    b = 0.0
                else:
                    # a is the cosine of the angle between z-axis and separation vector
                    a = dz / r
                    b = K * r
                
                c = np.sin(b)
                d = np.cos(b)
                
                # Handle b=0 case (atoms at same position)
                if abs(b) < 1e-10:
                    # For b→0, use limiting values
                    # sin(b)/b → 1, cos(b)/b^2 → ∞, sin(b)/b^3 → ∞
                    # But the combination (cos(b)/b^2 - sin(b)/b^3) has a finite limit
                    # Using L'Hôpital's rule or series expansion:
                    # For small b: cos(b)/b^2 - sin(b)/b^3 ≈ 1/(3b) - b/30 + ...
                    # Actually, the limit is 0 for the combination when properly handled
                    term1 = (1 - a**2) * 1.0  # sin(b)/b → 1
                    term2 = 0.0  # The combination goes to 0 in the limit
                else:
                    term1 = (1 - a**2) * (c / b)
                    term2 = (1 - 3*a**2) * (d / b**2 - c / b**3)
                
                # MATLAB formula: H(j,k) = -(1i*(gam/2)+del_omega)*(3/(8*pi))*(4*pi*(1-a^2)*(c/b)+4*pi*(1-3*a^2)*(d/b^2-c/b^3))
                # Simplifies to: H(j,k) = -(1i*(gam/2)+del_omega)*(3/2)*((1-a^2)*(c/b)+(1-3*a^2)*(d/b^2-c/b^3))
                H[j, k] = -(1j * (gam/2) + del_omega) * (3/2) * (term1 + term2)
            elif j == k:
                H[j, k] = -1j * gam / 2
    
    return H


def compute_F_jk_equation5(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    lambda_val: float,
    gam: float,
) -> np.ndarray:
    """
    Compute F_jk coupling matrix using equation 5 from paper 2408.
    
    Equation 5: F_jk = F_kj = - (i Γ / 2) (3 / (8π)) [4π(1 - cos²θ_jk) (sin(k_a r_jk) / (k_a r_jk)) 
                + 4π(1 - 3 cos²θ_jk) ((cos(k_a r_jk) / (k_a r_jk)²) - (sin(k_a r_jk) / (k_a r_jk)³))]
    
    This simplifies to: F_jk = - (i Γ / 2) (3/2) [(1 - cos²θ_jk) (sin(k_a r_jk) / (k_a r_jk)) 
                + (1 - 3 cos²θ_jk) ((cos(k_a r_jk) / (k_a r_jk)²) - (sin(k_a r_jk) / (k_a r_jk)³))]
    
    For the off-diagonal Hamiltonian (equation 4), we need the real part of F_jk.
    
    Args:
        x: Array of x coordinates (N,)
        y: Array of y coordinates (N,)
        z: Array of z coordinates (N,)
        lambda_val: Wavelength
        gam: Decay rate Γ
        N: Number of atoms
        
    Returns:
        F matrix (N, N) with complex coupling coefficients (as per equation 5)
    """
    N=len(x)
    F = np.zeros((N, N), dtype=complex)
    k_a = 2 * np.pi / lambda_val  # Wavevector magnitude, may need to be a vector
    omega_a = 2 * np.pi / lambda_val  # Angular frequency (not used here)
    for j in range(N):
        for k in range(j,N):
            if j == k:
                F[j, k] = 1-1j*gam  # No self-coupling in off-diagonal part
            else:
                # Compute distance components
                dz = z[j] - z[k]
                dy = y[j] - y[k]
                dx = x[j] - x[k]
                r_jk = np.sqrt(dx**2 + dy**2 + dz**2)
                
                if r_jk < 1e-10:
                    F[j, k] = 0.0
                else:
                    # cos(θ_jk) is the cosine of angle between z-axis and separation vector
                    cos_theta_jk = dz / r_jk
                    k_a_r_jk = k_a * r_jk
                    
                    sin_kr = np.sin(k_a_r_jk)
                    cos_kr = np.cos(k_a_r_jk)
                    
                    # Handle k_a_r_jk = 0 case
                    if abs(k_a_r_jk) < 1e-10:
                        term1 = 0  # sin(kr)/(kr) → 1
                        term2 = 3*1j /2  # (cos(kr)/(kr)² - sin(kr)/(kr)³) → 0
                    else:
                        term1 = (1 - cos_theta_jk**2) * (sin_kr / k_a_r_jk)
                        term2 = (1 - 3 * cos_theta_jk**2) * (cos_kr / (k_a_r_jk**2) - sin_kr / (k_a_r_jk**3))
                    
                    # Equation 5: F_jk = - (i Γ / 2) (3/2) * (term1 + term2)
                    F[j, k] = -(1j * gam / 2) * (3/2) * (term1 + term2)
                    F[k, j] = F[j, k]  # F_jk = F_kj (symmetric)
    
    return F


def create_equation4_hamiltonian(
    num_atoms: int,
    F_matrix: np.ndarray
) -> List[Tuple[float, List[str]]]:
    """
    Create Hamiltonian for equation 4 from paper 2408 (off-diagonal part only).
    
    Equation 4: Ĥ^jk = F_jk σ̂_+^j σ̂_-^k + F_kj σ̂_-^j σ̂_+^k
    
    This decomposes into Pauli strings as:
    (F_jk/2) * (X_j X_k - Y_j Y_k)
    
    Note: This is MINUS Y_j Y_k, not PLUS (different from subradiance case).
    
    Args:
        num_atoms: Number of atoms
        F_matrix: Complex F_jk matrix from equation 5 (N, N)
        
    Returns:
        List of Pauli strings representing the off-diagonal Hamiltonian
    """
    pauli_strings = []
    
    # For equation 4, we only include off-diagonal terms (j != k)
    # The decomposition is: (F_jk/2) * (X_j X_k - Y_j Y_k)
    for j in range(num_atoms):
        for k in range(j, num_atoms):
            if j != k:  
                pauli_strings.append((F_matrix[j, k] / 2, [f"X{j}", f"X{k}"]))
                pauli_strings.append((-F_matrix[k, j] / 2, [f"Y{j}", f"Y{k}"]))
            else:
                pauli_strings.append((F_matrix[j, k]/2, [f"Z{j}"]))  # Identity term for self-coupling
                continue
    
    return pauli_strings


def setup_positions_3d_grid(N: int, m: float = 1.5, lambda_val: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Set up 3D grid positions for atoms (matching MATLAB code).
    
    Args:
        N: Number of atoms (should be a perfect cube: N = n^3)
        m: Spacing multiplier
        lambda_val: Wavelength
        
    Returns:
        Tuple of (x, y, z) coordinate arrays
    """
    n_per_dim = int(round(N**(1/3)))
    if n_per_dim**3 != N:
        print(f"Warning: N={N} is not a perfect cube. Using {n_per_dim**3} atoms.")
        N = n_per_dim**3
    
    sx = m * lambda_val / n_per_dim
    sy = m * lambda_val / n_per_dim
    sz = m * lambda_val / n_per_dim
    
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    
    index = 0
    for ind in range(n_per_dim):
        for ind1 in range(n_per_dim):
            for ind2 in range(n_per_dim):
                x[index] = sx * ind
                y[index] = sy * ind1
                z[index] = sz * ind2
                index += 1
    
    return x, y, z

def setup_positions_2d_grid(N: int, m: float = 1.5, lambda_val: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Set up 3D grid positions for atoms (matching MATLAB code).
    
    Args:
        N: Number of atoms (should be a perfect cube: N = n^3)
        m: Spacing multiplier
        lambda_val: Wavelength
        
    Returns:
        Tuple of (x, y, z) coordinate arrays
    """
    n_per_dim = int(round(N**(1/2)))
    if n_per_dim**2 != N:
        print(f"Warning: N={N} is not a perfect square. Using {n_per_dim**2} atoms.")
        N = n_per_dim**2
        print("n_dim =", n_per_dim, "N =", N)
    
    sx = m * lambda_val / n_per_dim
    sy = m * lambda_val / n_per_dim
    sz = m * lambda_val / n_per_dim
    
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    
    index = 0
    for ind in range(n_per_dim):
        for ind1 in range(n_per_dim):
            x[index] = sx * ind
            y[index] = sy * ind1
            z[index] = 0
            index += 1
    
    return x, y, z


def compute_equation4_eigenvalues(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    lambda_val: float,
    gam: float,
    use_zxw: bool = True
) -> np.ndarray:
    """
    Compute eigenvalues of the off-diagonal Hamiltonian from equation 4.
    
    Equation 4: Ĥ^jk = F_jk σ̂_+^j σ̂_-^k + F_kj σ̂_-^j σ̂_+^k
    Decomposes as: (F_jk/2) * (X_j X_k - Y_j Y_k)
    
    Args:
        x: Array of x coordinates (N,)
        y: Array of y coordinates (N,)
        z: Array of z coordinates (N,)
        lambda_val: Wavelength
        gam: Decay rate Γ
        use_zxw: If True, use ZXW method; if False, use direct numpy
        
    Returns:
        Array of eigenvalues
    """
    N = len(x)
    
    # Compute F_jk matrix using equation 5
    F_matrix = compute_F_jk_equation5(x, y, z, lambda_val, gam, N)
    
    if use_zxw:
        if PauliHamiltonianZX is None:
            raise ImportError("ZXW functions not available. Install required packages.")
        
        # Create Pauli string Hamiltonian for equation 4
        pauli_strings = create_equation4_hamiltonian(N, F_matrix)
        
        # Create ZXW Hamiltonian
        hamiltonian_zxw = PauliHamiltonianZX(pauli_strings)
        
        # Compute eigenvalues
        eigenvalues = hamiltonian_zxw.compute_eigenvalues(hermitian=True)
    else:
        # Direct numpy computation
        # Build the Hamiltonian matrix directly
        H = np.zeros((2**N, 2**N), dtype=complex)
        
        # This would require building the full 2^N dimensional matrix
        # For now, we'll use ZXW method
        raise NotImplementedError("Direct numpy method not implemented for equation 4. Use use_zxw=True.")
    
    return eigenvalues

