import numpy as np
import scipy
import scipy.linalg

def coherent_trans(f, NA, n, delta_z):
    """ Coherent transmission function for aberration-less optical system.

    Args:
        f: spatial frequencies [fx, fy], normalized such that wavelength is 1
        NA: numerical aperture
        n: refractive index of imaging medium
        delta_z: defocus distance

    Returns:
        K: coherent transmission function as function of f
    """

    f2 = f[0]**2 + f[1]**2
    mask = f2 < NA**2
    return mask * np.exp(2j * np.pi * delta_z * np.sqrt(mask*(n**2 - f2)))

def source_circ(f, ctr, r, I0=1):
    """ Mutual intensity function J(f) in freq domain for circular source.
    
    Args:
        f: spatial frequencies [fx, fy], normalized such that wavelength is 1
	ctr: center spatial frequency [cx, cy]
        r: radius
        I0: intensity of source

    Returns:
        J: mutual intensity as function of f
    """

    f2 = (f[0] - ctr[0])**2 + (f[1] - ctr[1])**2
    mask = f2 < r**2
    return I0 * mask

def source_annular(f, ctr, r_min, r_max, I0=1):
    """ Mutual intensity function J(f) in freq domain for annular source.
    
    Args:
        f: spatial frequencies [fx, fy], normalized such that wavelength is 1
	ctr: center spatial frequency [cx, cy]
        r_min: inner radius
        r_max: outer radius
        I0: intensity of source

    Returns:
        J: mutual intensity as function of f
    """

    f2 = (f[0] - ctr[0])**2 + (f[1] - ctr[1])**2
    mask = (f2 >= r_min**2) * (f2 <= r_max**2)
    return I0 * mask

def source_arc(f, ctr, r_min, r_max, phi_min, phi_max, I0=1):
    """ Mutual intensity function J(f) in freq domain for arc source.
    
    Args:
        f: spatial frequencies [fx, fy], normalized such that wavelength is 1
	ctr: center spatial frequency [cx, cy]
        r_min: inner radius
        r_max: outer radius
        phi_min: start angle (degrees)
        phi_max: stop angle (degrees)
        I0: intensity of source
    
    Returns:
        J: mutual intensity as function of f

    Notes:
        The source starts at phi_min and stops at phi_max, moving in a
        counter-clockwise direction.
    """

    phi = np.mod( 180 / np.pi * np.arctan2(f[1] - ctr[1], f[0] - ctr[0]), 360)
    
    phi_min = np.mod(phi_min, 360)
    phi_max = np.mod(phi_max, 360)

    if phi_min < phi_max:
        phi_mask = (phi >= phi_min) & (phi <= phi_max)
    else:
        phi_mask = (phi >= phi_min) | (phi <= phi_max)
    
    return phi_mask * source_annular(f, ctr, r_min, r_max, I0)

def trans_cross_coeff(K, J, df, fmax):
    """ Computes transmission cross-coefficients (TCC) in frequency domain.

    Args:
        K: function of the form K(f), which represents the coherent
           transfer function in the frequency domain
        J: function of the form J(f), which represents the mutual intensity
           function in the frequency domain
        df: frequency spacing for computing TCC
        fmax: maximum frequency for computing TCC

    Returns:
       TCC: 4D array representing transmission cross-coefficients, of the form
            TCC[f1_x, f1_y, f2_x, f2_y]
       f: 1D array containing spatial frequencies at which TCC was evaluated.

    """

    # Frequencies to evaluate  
    Np = np.floor(fmax/df)
    n = np.arange(-Np, Np+1, dtype=np.int64)
    N = n.size

    f = df * n

    # Evaluate K and J at grid points
    fxy = np.meshgrid(f, f)
    Jm = J(fxy)
    Km = K(fxy)

    # Compute TCC via FFT
    Jmf = np.fft.fft2(Jm)
    Kmf = np.fft.fft2(Km)

    i = np.broadcast_to(np.reshape(np.arange(N), (N, 1, 1, 1)), [N]*4)
    j = np.broadcast_to(np.reshape(np.arange(N), (1, N, 1, 1)), [N]*4)
    k = np.broadcast_to(np.reshape(np.arange(N), (1, 1, N, 1)), [N]*4)
    l = np.broadcast_to(np.reshape(np.arange(N), (1, 1, 1, N)), [N]*4)

    TCCf = df**2 * (Jmf[np.mod(-i - k, N), np.mod(-j - l, N)]
        * Kmf[np.mod(i, N), np.mod(j, N)]
        * np.conj(Kmf[np.mod(-k, N), np.mod(-l, N)]))

    TCC = np.fft.fftshift(np.fft.ifftn(TCCf))

    return TCC, f


def optical_kernels(K, J, Lxy, fmax, num):
    """ Computes kernels of Optimal Coherent Approximation (OCA).
    
    Args:
        K: function of the form K(f), which represents the coherent
           transfer function in the frequency domain
        J: function of the form J(f), which represents the mutual intensity
           function in the frequency domain
        Lxy: width of square domain over which to compute kernels
        fmax: maximum spatial frequency for OCA calculation; must be larger
             than support of transmission cross-coefficient operator
        num: number of kernels to compute

    Returns:
        coeff: OCA coefficients [l_0, ... l_num-1] 
        phi: OCA kernels [phi_0, ... phi_num-1], where phi_i is a 2D array
             representing the complex fields of the i-th OCA kernel
    """

    df = 1/Lxy

    # Compute TCC
    TCC, f = trans_cross_coeff(K, J, df, fmax)
    
    # Convert to 2D matrix
    N = f.size
    TCCm = np.reshape(np.reshape(TCC, (N, N, -1)), (N**2, -1))

    # Compute largest eigenvalues and eigenvectors
    w, v = scipy.linalg.eigh(TCCm, eigvals=(N**2 - num, N**2 - 1))
    w = np.flip(w, 0)
    v = np.flip(v, 1)

    # Get real-space optical kernels
    phi = []
    for i in range(num):
        Phi = np.fft.ifftshift(np.reshape(v[:,i], [N, N]))
        #phi.append(np.fft.fftshift(np.fft.ifft2(Phi)))
        phi.append(np.fft.ifft2(Phi))
    
    return w, phi

