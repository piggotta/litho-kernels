import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.optimize
import subprocess
import json
import os

def make_odd(phi1, phi2):
    """ Returns linear combinations of orthogonal fields phi1 and phi2
        which are maximally odd functions across the two diagonal axes. 
        
        Args:
            phi1, phi2: 2D arrays representing the orthogonal input fields

        Returns:
            phi1_new, phi2_new: maximally odd fields, guaranteed orthogonal
    """
    A = phi1 / np.sqrt(np.sum(np.abs(phi1) ** 2))
    B = phi2 / np.sqrt(np.sum(np.abs(phi2) ** 2))

    # Define symmetrized fields
    As = A + A.T
    Bs = B + B.T

    # Pre-compute quantities
    As2 = np.abs(As) ** 2
    Bs2 = np.abs(Bs) ** 2
    AsBs = np.conj(As) * Bs

    # Define objective function
    def f(x):
        a = np.maximum(np.minimum(x[0], 1), 0)
        b = np.sqrt(1 - a**2)
        c = x[1]

        obj = np.sum(a**2 * As2 + b**2 * Bs2 
                + 2 * a * b * np.real(AsBs * np.exp(1j * c)))
        return obj
   
    # Find parameters that produce most ''odd'' field
    f_eval = 1
    while f_eval > 1e-3:
        x0 = [np.random.rand(), 2 * np.pi * np.random.rand()]
        res = scipy.optimize.minimize(f, x0, 
            bounds=[(0, 1), (-2*np.pi, 4*np.pi)])
        f_eval = f(res.x)

    a = res.x[0]
    b = np.sqrt(1 - a**2) * np.exp(1j * res.x[1])

    # Convert to new basis
    phi1_new = a * phi1 + b * phi2
    phi2_new = np.conj(b) * phi1 - np.conj(a) * phi2

    return phi1_new, phi2_new


class CompareMentor():
    """ Compares locally calculated OCA kernels with Mentor data. """

    def __init__(self):
        """ Computes OCA kernels and performs comparison. """

        # Files
        sourcemap_file = 'mentor/sourcemap.mat'
        kernels_file = 'mentor/kernels.mat'
        weights_file = 'mentor/weights.txt'

        lithosys_file = 'mentor/lithosys.json'
        oca_file = 'mentor/oca.npz'

        # Basic information
        NA = 1.35       # Numerical aperture
        n = 1.43664     # Refractive index (immersion water)
        wl = 193        # Wavelength (nm)

        # Source details
        sigma_min = 0.7
        sigma_max = 0.9
        delta_phi = 30

        # Load Mentor data from file
        sourcemap = scipy.io.loadmat(sourcemap_file)
        kernels = scipy.io.loadmat(kernels_file)
        weights = np.loadtxt(weights_file)

        step = sourcemap['step'][0,0]
        src_grid = sourcemap['src_grid']

        dx = kernels['dx'][0,0]
        phi2_M = kernels['kernels'][0]

        coeff_M = weights

        # Generate litho system
        Nxy = 2048
        Lxy = Nxy * dx / wl

        source = []
        for i in range(4):
            source.append({
                'type': 'arc',
                'ctr': [0, 0],
                'sigma_min': sigma_min,
                'sigma_max': sigma_max,
                'phi_min': 45 + i*90 - delta_phi/2,
                'phi_max': 45 + i*90 + delta_phi/2
                })

        lithosys = {
            'source': source,
            'NA': NA,
            'n': n,
            'delta_z': 0,
            'fmax': 3,
            'Lxy': Lxy,
            'Nxy': Nxy,
            'num': 10
            }

        with open(lithosys_file, 'w') as f:
            f.write(json.dumps(lithosys, indent=4))

        # Compute OCA
        subprocess.run(['python3', 'compute_oca.py', lithosys_file, oca_file])

        # Load OCA
        oca = np.load(oca_file)
        coeff = oca['coeff']
        phi = []
        for p in oca['phi']:
            phi.append(p)

        # Make pairs of modes odd to match Mentor convention
        phi[1], phi[2] =  make_odd(phi[1], phi[2])
        phi[6], phi[5] =  make_odd(phi[5], phi[6])

        # Sample OCA kernels over same region as Mentor data
        Nxy_M = phi2_M[0].shape[0]
        for i, p in enumerate(phi):
            p = np.fft.ifftshift(p)
            p = np.concatenate((p[-(Nxy_M//2) : , :],
                p[ : Nxy_M//2 + 1, :]), axis=0)
            p = np.concatenate((p[:, -(Nxy_M//2) : ],
                p[:,  : Nxy_M//2 + 1]), axis=1)
            phi[i] = p

        # Normalize all coefficients and kernels
        coeff = coeff / coeff[0]
        phi2 = []
        for i, p in enumerate(phi):
            phi2.append(np.abs(p)**2) 
            phi[i] = phi[i] / np.sqrt(np.sum(phi2[i]))
            phi2[i] = phi2[i] / np.sum(phi2[i])

        coeff_M = coeff_M / coeff_M[0]
        for i, p in enumerate(phi):
            phi2_M[i] = phi2_M[i] / np.sum(phi2_M[i])

        # Compute difference between kernels
        delta = []
        norm = []
        norm_M = []
        for i, p in enumerate(phi2):
            norm.append(np.sum(p))
            norm_M.append(np.sum(phi2_M[i]))
            delta.append(np.sum(np.abs(p- phi2_M[i])))

        # Delete temporary files
        os.remove(lithosys_file)
        os.remove(oca_file)

        # Store results
        self.dx = dx
        self.Nxy_M = Nxy_M
        self.src_grid = src_grid

        self.phi = phi
        self.phi2 = phi2
        self.coeff = coeff

        self.phi2_M = phi2_M
        self.coeff_M = coeff_M
         
        self.norm = norm
        self.norm_M = norm
        self.delta = delta

    def print_results(self):
        """ Prints results of comparison. """

        # Compare calculations to Mentor data
        print('OCA coefficients:')
        print('Calc    Mentor')
        for i, c in enumerate(self.coeff):
            print('%.4f  %.4f' % (self.coeff[i], self.coeff_M[i]))

        print()
        print('OCA kernels:')
        print('Diff    Norm_C  Norm_M')
        for i, p in enumerate(self.phi2):
            print('%.4f  %.4f  %.4f' 
                % (self.delta[i], self.norm[i], self.norm_M[i]))

    def plot_results(self):
        """ Plots results of comparison and saves to file. """

        # Generate plots
        ind = np.arange(len(self.phi2)) + 1
        plt.figure()
        plt.semilogy(ind, self.coeff, 'k', ind, self.coeff_M, 'r')
        plt.xlim([1, 10])
        plt.xlabel('Index')
        plt.ylabel('OCA coefficient')
        plt.legend(['Calculation', 'Mentor'])
        plt.title('Comparing OCA coefficients')
        plt.savefig('mentor/plots/compare_coeff.pdf', bbox_inches='tight')

        plt.figure()
        plt.plot(ind, self.delta, 'k')
        plt.xlim([1, 10])
        plt.ylim([0, 1])
        plt.xlabel('Index')
        plt.ylabel('$L_1( |\phi_{c}|^2 - |\phi_{M}|^2)$')
        plt.title('Difference in OCA kernels')
        plt.savefig('mentor/plots/compare_kernels.pdf', bbox_inches='tight')

        plt.figure()
        plt.imshow(self.src_grid, extent=[-1, 1, -1, 1])
        plt.xlabel('$\sigma_x$')
        plt.ylabel('$\sigma_y$')
        plt.title('Source')
        plt.axis('scaled')
        plt.colorbar()
        plt.savefig('mentor/plots/source.png', bbox_inches='tight', dpi=300)

        x = self.dx * np.asarray([0, self.Nxy_M-1])
        x = x - np.mean(x)
        extent = np.concatenate((x, x))

        for i, p in enumerate(self.phi2):
            fig, axarr = plt.subplots(2, 1)

            im = axarr[0].imshow(self.phi2[i], extent=extent)
            axarr[0].set_xlabel('x (nm)')
            axarr[0].set_ylabel('y (nm)')
            axarr[0].set_title('Calculated')
            axarr[0].axis('scaled')
         
            im = axarr[1].imshow(self.phi2_M[i], extent=extent)
            axarr[1].set_xlabel('x (nm)')
            axarr[1].set_ylabel('y (nm)')
            axarr[1].set_title('Mentor')
            axarr[1].axis('scaled')

            plt.tight_layout()
            plt.savefig('mentor/plots/kernel_%d.png' % i,
                bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    comp = CompareMentor()
    comp.print_results()
    comp.plot_results()

