"""
Memory-Efficient Vectorized Green's Functions

A standalone package for computing retarded and lesser Green's functions 
with automatic GPU memory management using CuPy when available.

Author: William Livernois
"""

try:
    import cupy as cp
    isCuda = cp.cuda.is_available()
    device = cp.cuda.Device()
    free_memory, total_memory = device.mem_info    
    print(f"GPU Memory configured: {free_memory/1e9:.1f} GB free of {total_memory/1e9:.1f} GB total")
except:
    isCuda = False

import numpy as np

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
    ndarray
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
    
    # Use CuPy if cuda available, otherwise numpy
    solver = cp if isCuda else np
    
    # Try full vectorization first, let CuPy handle memory
    try:
        return GrIntVectorized(F, S, g, Elist, weights, solver)
    except (MemoryError, (cp.cuda.memory.OutOfMemoryError if isCuda else Exception)):
        print("Memory error - switching to chunked computation")
        if isCuda:
            cp.get_default_memory_pool().free_all_blocks()
        return GrIntChunked(F, S, g, Elist, weights, solver)


def GrIntVectorized(F, S, g, Elist, weights, solver):
    """
    Original vectorized implementation - preserved exactly.
    """
    M = Elist.size
    N = F.shape[0]
    
    # Convert array types to match solver:
    Elist_ = solver.array(Elist, dtype=solver.complex128)
    weights = solver.array(weights)
    S = solver.array(S, dtype=solver.complex128)
    F = solver.array(F, dtype=solver.complex128)
    
    #Generate vectorized matrices conserving memory
    ES_minus_F_minus_Sig = Elist_[:, None, None] * solver.tile(solver.array(S), (M, 1, 1))
    ES_minus_F_minus_Sig -= solver.tile(solver.array(F), (M, 1, 1))
    ES_minus_F_minus_Sig -= solver.array([g.sigmaTot(E) for E in Elist])
    Gr_vec = solver.linalg.solve(ES_minus_F_minus_Sig, solver.tile(solver.eye(N), (M, 1, 1)))
    del ES_minus_F_minus_Sig

    #Sum up using weights, convert back to numpy array
    Gint = solver.sum(Gr_vec*weights[:, None, None], axis=0)
    del Gr_vec 
    return Gint.get() if isCuda else Gint


def GrIntChunked(F, S, g, Elist, weights, solver):
    """
    Chunked version that preserves original mathematical operations.
    """
    M = Elist.size
    N = F.shape[0]
    
    # Start with reasonable chunk, adapt based on CuPy feedback
    chunk_size = max(1, M // 2)
    
    # Initialize result
    Gint = solver.zeros((N, N), dtype=solver.complex128)
    
    i = 0
    while i < M:
        end_idx = min(i + chunk_size, M)
        
        try:
            Elist_chunk = Elist[i:end_idx]
            weights_chunk = weights[i:end_idx]
            Gint += GrIntVectorized(F, S, g, Elist_chunk, weights_chunk, solver)            
            i = end_idx  # Success - move to next chunk
            
        except (MemoryError, (cp.cuda.memory.OutOfMemoryError if isCuda else Exception)):
            # Still too big - make chunk smaller
            if isCuda:
                cp.get_default_memory_pool().free_all_blocks()
            chunk_size = max(1, chunk_size // 2)
            print(f"Reducing Gr chunk size to {chunk_size}")
            # Don't increment i - retry this chunk
    
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
    
    solver = cp if isCuda else np
    
    # Try full vectorization first, let CuPy handle memory
    try:
        return GrLessVectorized(F, S, g, Elist, weights, solver, ind)
    except (MemoryError, (cp.cuda.memory.OutOfMemoryError if isCuda else Exception)):
        print("Memory error - switching to chunked computation")
        if isCuda:
            cp.get_default_memory_pool().free_all_blocks()
        return GrLessChunked(F, S, g, Elist, weights, solver, ind)



def GrLessVectorized(F, S, g, Elist, weights, solver, ind):
    """
    Full Vectorized G< implementation.
    """
    M = Elist.size
    N = F.shape[0]
    
    # Convert array types to match solver:
    Elist_ = solver.array(Elist, dtype=solver.complex128)
    weights = solver.array(weights)
    S = solver.array(S, dtype=solver.complex128)
    F = solver.array(F, dtype=solver.complex128)
    
    #Generate Gr and Ga vectorized matrices conserving memory
    ES_minus_F_minus_Sig = Elist_[:, None, None] * solver.tile(S, (M, 1, 1))
    ES_minus_F_minus_Sig -= solver.tile(F, (M, 1, 1))
    SigmaTot = solver.array([g.sigmaTot(E) for E in Elist])
    ES_minus_F_minus_Sig -= SigmaTot
    
    Gr_vec = solver.linalg.solve(ES_minus_F_minus_Sig, solver.tile(solver.eye(N), (M, 1, 1)))
    del ES_minus_F_minus_Sig
    
    Ga_vec = solver.conj(Gr_vec).transpose(0, 2, 1)
    
    # Calculate Gamma:
    if ind is not None:
        SigList = SigmaTot
    else:
        del SigmaTot
        SigList = solver.array([g.sigma(E, ind) for E in Elist])  # Note: This line in original had issue - E not used
        
    GammaList = 1j * (SigList - solver.conj(SigList).transpose(0, 2, 1))
    del SigList

    # Calculate Gless:
    Gless_vec = solver.matmul(solver.matmul(Gr_vec, GammaList), Ga_vec)
    del Gr_vec, Ga_vec, GammaList
    
    #Sum up using weights, convert back to numpy array
    Gint = solver.sum(Gless_vec*weights[:, None, None], axis=0)
    return Gint.get() if isCuda else Gint


def GrLessChunked(F, S, g, Elist, weights, solver, ind):
    """
    Chunked G< version preserving original operations.
    """
    M = Elist.size
    N = F.shape[0]
    
    # Start with reasonable chunk, adapt based on CuPy feedback
    chunk_size = max(1, M // 2)

    # Initialize result
    Gint = solver.zeros((N, N), dtype=solver.complex128)
    
    i = 0
    while i < M:
        end_idx = min(i + chunk_size, M)
        
        try:
            Elist_chunk = Elist[i:end_idx]
            weights_chunk = weights[i:end_idx]
            Gint += GrLessVectorized(F, S, g, Elist_chunk, weights_chunk, solver, ind)            
            i = end_idx  # Success - move to next chunk
            
        except (MemoryError, (cp.cuda.memory.OutOfMemoryError if isCuda else Exception)):
            # Still too big - make chunk smaller
            if isCuda:
                cp.get_default_memory_pool().free_all_blocks()
            chunk_size = max(1, chunk_size // 2)
            print(f"Reducing G< chunk size to {chunk_size}")
            # Don't increment i - retry this chunk
    
    return Gint.get() if isCuda else Gint

