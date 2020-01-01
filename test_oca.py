""" Test generation of OCA kernels """

import numpy as np
import scipy.signal as signal
import unittest
import itertools

import oca
import compare_mentor

class TestTCC(unittest.TestCase):
    """ Tests TCC computation """

    def test_trans_cross_coeff(self):
        """ Tests that the TCC computed by trans_cross_coeff matches
        the brute-force calculation."""
   
        # Parameters
        n = 1.3
        NA = 1
        sigma = 0.8
        delta_z = 0

        fmax = 3 * NA
        df = 0.5
        
        # Coherent transfer function
        def K(f):
            return oca.coherent_trans(f, NA, n, delta_z)

        # Mutual intensity function
        def J(f):
            return oca.source_circ(f, [0, 0], NA * sigma)

        # Brute force calculation of TCC 
        def trans_cross_coeff_bf(K, J, df, fmax):
            # Frequencies to evaluate  
            Np = np.floor(fmax/df)
            n = np.arange(-Np, Np+1, dtype=np.int64)
            N = n.size

            f = df * n

            # Evaluate K and J at grid points
            fxy = np.meshgrid(f, f)
            Jm = J(fxy)
            Km = K(fxy)

            # Compute TCC via brute-force convolution
            TCC = np.zeros([n.size] * 4, dtype=np.complex128)
            for i, j, k, l in itertools.product(
                *([range(N) for a in range(4)])):

                TCC[i, j, k, l] = np.sum(Jm 
                    * np.roll(Km, (n[i], n[j]), axis=(0, 1)) 
                    * np.roll(np.conj(Km), (n[k], n[l]), axis=(0, 1)))

            TCC = TCC * df**2

            return TCC, f
        
        # Compute TCC using library
        TCC, f = oca.trans_cross_coeff(K, J, df, fmax)
        
        # Compute TCC using brute force method
        TCC_bf, f_bf = trans_cross_coeff_bf(K, J, df, fmax)

        # Compare results from different methods
        norm = np.sum(np.abs(TCC))              # Norm of TCC, library
        norm_bf = np.sum(np.abs(TCC_bf))        # Norm of TCC, brute force
        delta = np.sum(np.abs(TCC - TCC_bf))    # Sum of differences of TCCs

        self.assertTrue(abs(norm_bf / norm - 1) < 1e-10)
        self.assertTrue(delta / norm < 1e-10)

class TestMentor(unittest.TestCase):
    """ Compares computed OCA kernels and coefficients with Mentor data. """

    def test_compare_mentor(self):
        """ Compute OCA and compare with Mentor data. """

        comp = compare_mentor.CompareMentor()

        # Check coefficients are within acceptable tolerances
        self.assertTrue(np.all(np.logical_or(
            np.less(comp.coeff - comp.coeff_M, 0.01),
            np.less(np.abs(comp.coeff - comp.coeff_M) / comp.coeff, 0.1))))

        # Check kernels are within acceptable tolerances
        self.assertTrue(np.all(np.less(comp.delta[0:3], 0.1)))
        self.assertTrue(np.all(np.less(comp.delta[3:8], 0.3)))
