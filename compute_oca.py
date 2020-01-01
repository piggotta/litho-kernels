import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import json
import argparse

import oca

if __name__ == '__main__':

    # Parse command line input
    parser = argparse.ArgumentParser()
    parser.add_argument('input', 
        help='Input .json file describing lithography system.')
    parser.add_argument('output',
        help='Output .npz file containing optical kernels and coefficients.')
    parser.add_argument('-p', '--plot', help='Plot optical kernels',
        action='store_true')
    parser.add_argument('-v', '--verbose', help='Verbose output',
        action='store_true')
    args = parser.parse_args()

    # Read json file
    with open(args.input, 'r') as f:
        lithosys = json.loads(f.read())

    if args.verbose:
        print('Input:', args.input)
        print('Output:', args.output)

    # Get parameters
    NA = lithosys['NA']
    n = lithosys['n']
    delta_z = lithosys['delta_z']
    fmax = lithosys['fmax']
    Lxy = lithosys['Lxy']
    Nxy = lithosys['Nxy']
    num = lithosys['num']
    source = lithosys['source']

    # Generate coherent transfer function
    def K(f):
        return oca.coherent_trans(f, NA, n, delta_z)

    # Generate mutual intensity function
    def J(f):
        J0 = 0
        for src in source:
            if src['type'] == 'circ':
                ctr = NA * np.asarray(src['ctr'])
                r = NA * src['sigma']
                I0 = src.get('IO', 1)
                J0 += oca.source_circ(f, ctr, r, I0)
            
            elif src['type'] == 'annular':
                ctr = NA * np.asarray(src['ctr'])
                r_min = NA * src['sigma_min']
                r_max = NA * src['sigma_max']
                I0 = src.get('IO', 1)
                J0 += oca.source_annular(f, ctr, r_min, r_max, I0)

            elif src['type'] == 'arc':
                ctr = NA * np.asarray(src['ctr'])
                r_min = NA * src['sigma_min']
                r_max = NA * src['sigma_max']
                phi_min = src['phi_min']
                phi_max = src['phi_max']
                I0 = src.get('IO', 1)
                J0 += oca.source_arc(f, ctr, r_min, r_max, phi_min, phi_max, I0)

        return J0 

    # Compute OCA kernels
    start = time.time()
    coeff, phi = oca.optical_kernels(K, J, Lxy, fmax, num)
    time_oca = time.time() - start

    # Fourier interpolation to final resolution
    start = time.time()
    Nxy_init = phi[0].shape[0]
    N_pad = Nxy - Nxy_init
    i_insert = int(np.ceil(Nxy_init/2))

    for i in range(num): 
        Phi = np.fft.fftn(phi[i])
        Phi_pad = np.insert(Phi, i_insert, np.zeros((N_pad, 1)), axis=0)
        Phi_pad = np.insert(Phi_pad, i_insert, np.zeros((N_pad, 1)), axis=1)
        phi[i] = np.fft.ifftn(Phi_pad)

    time_interp = time.time() - start

    # Center the kernels
    for i, p in enumerate(phi):
        phi[i] = np.fft.fftshift(p)

    # Export OCA kernels and coefficients
    np.savez(args.output, Lxy=Lxy, coeff=coeff, phi=phi)

    # Print computation times
    if args.verbose:
        print('OCA compute time:   %.3f s' % time_oca)
        print('Interpolation time: %.3f s' % time_interp)
         
    # Plot source and OCA kernels
    if args.plot:

        # Source
        f = np.linspace(-NA, NA, Nxy)
        fx, fy = np.meshgrid(f, f)
        plt.figure()
        plt.pcolormesh(f/NA, f/NA, np.abs(J([fx, fy])))
        plt.xlabel('$\sigma_x$')
        plt.ylabel('$\sigma_y$')
        plt.axis('scaled')
        plt.title('Source')

        # OCA kernels
        x = np.linspace(0, Lxy, Nxy+1)[:-1]
        x = x - np.mean(x)

        for i, p in enumerate(phi):
            plt.figure()
            plt.pcolormesh(x, x, np.abs(p)**2)
            plt.xlabel('$x$ ($\lambda$)')
            plt.ylabel('$y$ ($\lambda$)')
            plt.axis('scaled')
            plt.title('Order %d: coefficient = %f' % (i, coeff[i]))

        plt.show()

