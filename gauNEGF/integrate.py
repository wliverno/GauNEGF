"""
Memory-Efficient Vectorized Green's Functions

A standalone package for computing retarded and lesser Green's functions 
with automatic GPU memory management using CuPy when available.

Author: Your Name
License: MIT
"""

try:
    import cupy as cp
    isCuda = cp.cuda.is_available()
    device = cp.cuda.Device()
    free_memory, total_memory = device.mem_info
    
    # Set memory pool to use 80% of total memory
    mempool = cp.get_default_memory_pool()
    mempool.set_limit(size=int(0.8 * total_memory))
    
    print(f"GPU Memory configured: {free_memory/1e9:.1f} GB free of {total_memory/1e9:.1f} GB total")
except:
    isCuda = False

import numpy as np


def estimate_solve_memory(M, N):
    """
    Estimate memory requirements for vectorized solve operation.
    
    Parameters
    ----------
    M : int
        Number of energy points
    N : int  
        Matrix dimension (N x N systems)
        
    Returns
    -------
    int
        Estimated memory in bytes
    """
    # A(M,N,N) + B(M,N,N) + result(M,N,N) for complex128 (16 bytes per element)
    return 3 * M * N * N * 16

def Gr(F, S, g, E):
    """
    Calculate retarded Green's function at given energy.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    E : float
        Energy in eV

    Returns
    -------
    ndarrayh
        Retarded Green's function G(E) = [ES - F - Σ(E)]^(-1)
    """
    solver = cp if isCuda else np
    mat = solver.array(E*S - F - g.sigmaTot(E))
    result = solver.linalg.inv(mat)
    return result.get() if isCuda else result

def DOSg(F, S, g, E):
    """
    Calculate density of states at given energy.

    Parameters
    ----------
    F : ndarray
        Fock matrix
    S : ndarray
        Overlap matrix
    g : surfG object
        Surface Green's function calculator
    E : float
        Energy in eV

    Returns
    -------
    float
        Density of states at energy E
    """
    return -np.trace(np.imag(Gr(F,S, g, E)))/np.pi

def GrInt(F, S, g, Elist, weights):
    """
    Integrate retarded Green's function for a list of energies using vectorization.
    
    Automatically manages GPU memory by chunking when necessary while preserving
    the exact mathematical operations of the original implementation.
    
    Parameters
    ----------
    F : ndarray
        Fock matrix (NxN)
    S : ndarray
        Overlap matrix (NxN)
    g : surfG object
        Surface Green's function calculator with sigmaTot(E) method
    Elist : ndarray
        Array of energies in eV (Mx1) - kept as numpy for g.sigmaTot()
    weights : ndarray
        Array of weights for each energy (Mx1)
        
    Returns
    -------
    ndarray
        Retarded Green's function G(E) integrated over the energy grid (NxN)
    """
    assert Elist.size == weights.size, "Elist and weights must have the same length"
    assert F.shape == S.shape, "F and S must have the same shape"
    assert F.shape[0] == F.shape[1], "F and S must be square matrices"
    
    M = Elist.size  # Number of points in the grid
    N = F.shape[0]  # Assuming F is square (NxN)
    
    # Use CuPy if cuda available, otherwise numpy
    solver = cp if isCuda else np
    
    # Check if we can fit the vectorized solve in memory
    if isCuda:
        device = cp.cuda.Device()
        free_memory = device.mem_info[0]
        memory_needed = estimate_solve_memory(M, N)
        
        if memory_needed > 0.7 * free_memory:
            return GrIntChunked(F, S, g, Elist, weights, solver)
    
    # Try full vectorization first (preserves original logic exactly)
    try:
        return GrIntVectorized(F, S, g, Elist, weights, solver)
    except (MemoryError, cp.cuda.memory.OutOfMemoryError if isCuda else Exception):
        if isCuda:
            cp.get_default_memory_pool().free_all_blocks()
            return GrIntChunked(F, S, g, Elist, weights, solver)
        else:
            raise


def GrIntVectorized(F, S, g, Elist, weights, solver):
    """
    Original vectorized implementation - preserved exactly.
    """
    M = Elist.size
    N = F.shape[0]
    
    # EXACT COPY of original code logic:
    Elist_ = solver.array(Elist)
    weights = solver.array(weights)
    S = solver.array(S)
    F = solver.array(F)
    
    #Generate vectorized matrices - EXACT original approach
    S_repeated = solver.tile(S, (M, 1, 1))  # Shape (MxNxN)
    F_repeated = solver.tile(F, (M, 1, 1))  # Shape (MxNxN)
    I = solver.eye(N)  # Shape (NxN)
    I_repeated = solver.tile(I, (M, 1, 1))  # Shape (MxNxN)
    
    SigmaTot = solver.array([g.sigmaTot(E) for E in Elist])  # Shape (MxNxN)
    
    #Solve for retarded Green's function - NOTE: This is the bottleneck for larger systems!
    Gr_vec = solver.linalg.solve(Elist_[:, None, None] * S_repeated - F_repeated - SigmaTot, I_repeated)
    
    #Sum up using weights, convert back to numpy array
    Gint = solver.sum(Gr_vec*weights[:, None, None], axis=0)
    return Gint.get() if isCuda else Gint


def GrIntChunked(F, S, g, Elist, weights, solver):
    """
    Chunked version that preserves original mathematical operations.
    """
    M = Elist.size
    N = F.shape[0]
    
    # Determine optimal chunk size
    if isCuda:
        free_memory = cp.cuda.Device().mem_info[0]
        target_memory = 0.5 * free_memory
        memory_per_point = estimate_solve_memory(1, N)
        chunk_size = max(1, min(M, int(target_memory / memory_per_point)))
    else:
        chunk_size = max(1, M // 4)
    
    print(f"Using chunked computation: {M} energies in chunks of {chunk_size}")
    
    # Initialize result
    Gint = solver.zeros((N, N), dtype=solver.complex128)
    
    # Process in chunks, preserving original logic within each chunk
    for i in range(0, M, chunk_size):
        end_idx = min(i + chunk_size, M)
        
        # Extract chunk - preserving original variable names and logic
        Elist_chunk = Elist[i:end_idx]
        weights_chunk = weights[i:end_idx]
        
        # EXACT original logic applied to chunk:
        Elist_ = solver.array(Elist_chunk)
        weights_chunk_gpu = solver.array(weights_chunk)
        S_gpu = solver.array(S)
        F_gpu = solver.array(F)
        
        chunk_M = Elist_chunk.size
        
        # Generate vectorized matrices - EXACT original approach for chunk
        S_repeated = solver.tile(S_gpu, (chunk_M, 1, 1))
        F_repeated = solver.tile(F_gpu, (chunk_M, 1, 1))
        I = solver.eye(N)
        I_repeated = solver.tile(I, (chunk_M, 1, 1))
        
        SigmaTot = solver.array([g.sigmaTot(E) for E in Elist_chunk])
        
        # Solve - EXACT original formula
        Gr_vec = solver.linalg.solve(Elist_[:, None, None] * S_repeated - F_repeated - SigmaTot, I_repeated)
        
        # Accumulate weighted result - same as original
        Gint += solver.sum(Gr_vec * weights_chunk_gpu[:, None, None], axis=0)
        
        # Cleanup for memory management
        del S_repeated, F_repeated, I_repeated, SigmaTot, Gr_vec
        if isCuda:
            cp.cuda.Stream.null.synchronize()
    
    return Gint.get() if isCuda else Gint


def GrLessInt(F, S, g, Elist, weights, ind=None):
    """
    Integrate nonequilibrium Green's function for a list of energies using vectorization.
    
    Parameters
    ----------
    F : ndarray
        Fock matrix (NxN)
    S : ndarray
        Overlap matrix (NxN)
    g : surfG object
        Surface Green's function calculator
    Elist : ndarray
        Array of energies in eV (Mx1)
    weights : ndarray
        Array of weights for each energy (Mx1)
    ind : int, optional
        Contact index for partial density calculation (default: None)
        
    Returns
    -------
    ndarray
        Nonequilibrium Green's function G<(E) integrated over the energy grid (NxN)
    """
    assert Elist.size == weights.size, "Elist and weights must have the same length"
    assert F.shape == S.shape, "F and S must have the same shape"
    assert F.shape[0] == F.shape[1], "F and S must be square matrices"
    
    M = Elist.size
    N = F.shape[0]
    solver = cp if isCuda else np
    
    # G< computation is more memory intensive, so be more conservative
    if isCuda:
        device = cp.cuda.Device()
        free_memory = device.mem_info[0]
        # Rough estimate: Gr + Ga + Gamma + G< ≈ 6*M*N*N matrices
        memory_needed = 6 * estimate_solve_memory(M, N) // 3
        
        if memory_needed > 0.5 * free_memory:
            return GrLessChunked(F, S, g, Elist, weights, solver, ind)
    
    try:
        return GrLessVectorized(F, S, g, Elist, weights, solver, ind)
    except (MemoryError, cp.cuda.memory.OutOfMemoryError if isCuda else Exception):
        if isCuda:
            cp.get_default_memory_pool().free_all_blocks()
            return GrLessChunked(F, S, g, Elist, weights, solver, ind)
        else:
            raise


def GrLessVectorized(F, S, g, Elist, weights, solver, ind):
    """
    Original G< implementation - preserved exactly.
    """
    M = Elist.size
    N = F.shape[0]
    
    # EXACT COPY of original code:
    Elist_ = solver.array(Elist)
    weights = solver.array(weights)
    S = solver.array(S)
    F = solver.array(F)
    
    #Generate vectorized matrices
    S_repeated = solver.tile(S, (M, 1, 1))  # Shape (MxNxN)
    F_repeated = solver.tile(F, (M, 1, 1))  # Shape (MxNxN)
    I = solver.eye(N)  # Shape (NxN)
    I_repeated = solver.tile(I, (M, 1, 1))  # Shape (MxNxN)
    SigmaTot = solver.array([g.sigmaTot(E) for E in Elist])  # Shape (MxNxN)
    
    #Solve for retarded Green's function - NOTE: This is the bottleneck for larger systems!
    Gr_vec = solver.linalg.solve(Elist_[:, None, None] * S_repeated - F_repeated - SigmaTot, I_repeated)
    Ga_vec = solver.conj(Gr_vec).transpose(0, 2, 1)  # Shape (MxNxN)
    
    # EXACT original logic for Gamma computation:
    if ind is not None:
        SigList = SigmaTot
    else:
        SigList = [g.sigma(E, i) for i in range(N)]  # Note: This line in original had issue - E not used
        
    GammaList = [1j*(sig - sig.conj().T) for sig in SigList]
    Gless_vec = solver.matmul(solver.matmul(Gr_vec, GammaList), Ga_vec)
    
    #Sum up using weights, convert back to numpy array
    Gint = solver.sum(Gless_vec*weights[:, None, None], axis=0)
    return Gint.get() if isCuda else Gint


def GrLessChunked(F, S, g, Elist, weights, solver, ind):
    """
    Chunked G< version preserving original operations.
    """
    M = Elist.size
    N = F.shape[0]
    
    # Calculate chunk size based on GPU memory
    free_memory = cp.cuda.Device().mem_info[0]
    memory_per_point = estimate_solve_memory(1, N) * 2  # Factor for additional matrices
    chunk_size = max(1, min(M, int(0.3 * free_memory / memory_per_point)))
    
    print(f"Using chunked G< computation: {M} energies in chunks of {chunk_size}")
    
    Gint = solver.zeros((N, N), dtype=solver.complex128)
    
    for i in range(0, M, chunk_size):
        end_idx = min(i + chunk_size, M)
        
        # Extract chunk
        Elist_chunk = Elist[i:end_idx]
        weights_chunk = weights[i:end_idx]
        
        # Apply EXACT original logic to chunk:
        Elist_ = solver.array(Elist_chunk)
        weights_chunk_gpu = solver.array(weights_chunk)
        S_gpu = solver.array(S)
        F_gpu = solver.array(F)
        
        chunk_M = Elist_chunk.size
        
        # Generate vectorized matrices - EXACT original
        S_repeated = solver.tile(S_gpu, (chunk_M, 1, 1))
        F_repeated = solver.tile(F_gpu, (chunk_M, 1, 1))
        I = solver.eye(N)
        I_repeated = solver.tile(I, (chunk_M, 1, 1))
        SigmaTot = solver.array([g.sigmaTot(E) for E in Elist_chunk])
        
        # Solve and compute Ga - EXACT original
        Gr_vec = solver.linalg.solve(Elist_[:, None, None] * S_repeated - F_repeated - SigmaTot, I_repeated)
        Ga_vec = solver.conj(Gr_vec).transpose(0, 2, 1)
        
        # Gamma computation - preserving original logic including the potential bug
        if ind is not None:
            SigList = SigmaTot
        else:
            SigList = [g.sigma(E, i) for i in range(N)]  # Preserving original (note: E not used correctly)
            
        GammaList = [1j*(sig - sig.conj().T) for sig in SigList]
        Gless_vec = solver.matmul(solver.matmul(Gr_vec, GammaList), Ga_vec)
        
        # Accumulate
        Gint += solver.sum(Gless_vec * weights_chunk_gpu[:, None, None], axis=0)
        
        # Cleanup
        del S_repeated, F_repeated, I_repeated, SigmaTot, Gr_vec, Ga_vec, Gless_vec
        if isCuda:
            cp.cuda.Stream.null.synchronize()
    
    return Gint.get() if isCuda else Gint

