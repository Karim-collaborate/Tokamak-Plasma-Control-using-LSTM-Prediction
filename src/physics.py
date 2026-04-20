import numpy as np
from scipy import special
# =============================================
# Calculation of magnetic field of thin coil (N turns)
# =============================================
def coil_field(r, z, R, Z0, I, N=1):
    """
    Poloidal coils
    Calculates Br and Bz of a thin coil (N turns) in (r, z).
    """
    mu0 = 4e-7 * np.pi  # vacuum permeability
    k_squared = (4 * r * R) / ((r + R)**2 + (z - Z0)**2)
    
    # Elliptic integrals
    K = special.ellipk(k_squared)
    E = special.ellipe(k_squared)
    
    # Common terms
    denominator = np.sqrt((r + R)**2 + (z - Z0)**2)
    Br_prefactor = (mu0 * I * N * (z - Z0)) / (2 * np.pi * r * denominator)
    Bz_prefactor = (mu0 * I * N) / (2 * np.pi * denominator)
    
    # Br and Bz components
    Br = Br_prefactor * (-K + ((r**2 + R**2 + (z - Z0)**2) / ((r - R)**2 + (z - Z0)**2) * E))
    Bz = Bz_prefactor * (K + ((R**2 - r**2 - (z - Z0)**2) / ((r - R)**2 + (z - Z0)**2) * E))
    
    return Br, Bz
