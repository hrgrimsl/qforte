import qforte

import copy
import numpy as np
import math

def grad_solve(self, residual_gradient, rtol = 1e-6):
    t = copy.deepcopy(self._tamps)
    Done = False
    iter = 0

    #Numerical Check
    '''
    residual = self.get_residual_vector(t)
    print("Numerical residual")
    print(residual)
    print("Numerical Jacobian")

    for i in range(0, len(t)):
        h = 1e-3
        tplus = copy.deepcopy(t)
        tplus[i] += h
        rplus = np.array(self.get_residual_vector(tplus))
        tminus = copy.deepcopy(t)
        tminus[i] -= h
        rminus = np.array(self.get_residual_vector(tminus))
        print((rplus - rminus)/(2*h))
    '''
    while Done == False:
        energy, residual, jacobian = residual_gradient(t)
        '''
        print(f"Analytical Residual")
        print(residual)
        print(f"Analytical Jacobian")
        print(jacobian)
        exit()
        '''
        print(f"Iteration:      {iter}")
        print(f"Energy:         {energy.real}")
        rnorm = np.linalg.norm(residual)
        print(f"Residual Norm:  {rnorm}")
        if rnorm < rtol:
            Done = True
        else:
            t -= (np.linalg.inv(jacobian)@residual).real
            iter += 1

    self._Egs = energy.real
    self._tamps = t

def diis_solve(self, residual, max_diis_dim = 12):
    """This function attempts to minimize the norm of the residual vector
    by using a quasi-Newton update procedure for the amplitudes paired with
    the direct inversion of iterative subspace (DIIS) convergence acceleration.
    """
    # draws heavy inspiration from Daniel Smith's ccsd_diis.py code in psi4 numpy
    
    diis_dim = 0
    t_diis = [copy.deepcopy(self._tamps)]
    e_diis = []
    rk_norm = 1.0
    Ek0 = self.energy_feval(self._tamps)

    print('\n    k iteration         Energy               dE           Nrvec ev      Nrm ev*        ||r||')
    print('---------------------------------------------------------------------------------------------------', flush=True)

    if (self._print_summary_file):
        f = open("summary.dat", "w+", buffering=1)
        f.write('\n#    k iteration         Energy               dE           Nrvec ev      Nrm ev*        ||r||')
        f.write('\n#--------------------------------------------------------------------------------------------------')
        f.close()

    for k in range(1, self._opt_maxiter+1):

        t_old = copy.deepcopy(self._tamps)

        #do regular update
        r_k = residual(self._tamps)
        rk_norm = np.linalg.norm(r_k)

        r_k = self.get_res_over_mpdenom(r_k)
        #r_k = [-i for i in r_k]
        self._tamps = list(np.add(self._tamps, r_k))

        Ek = self.energy_feval(self._tamps)
        dE = Ek - Ek0
        Ek0 = Ek

        print(f'     {k:7}        {Ek:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {rk_norm:+12.10f}', flush=True)

        if (self._print_summary_file):
            f = open("summary.dat", "a", buffering=1)
            f.write(f'\n     {k:7}        {Ek:+12.12f}      {dE:+12.12f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {rk_norm:+12.12f}')
            f.close()

        if(rk_norm < self._opt_thresh):
            self._Egs = Ek
            break

        t_diis.append(copy.deepcopy(self._tamps))
        e_diis.append(np.subtract(copy.deepcopy(self._tamps), t_old))
        if len(e_diis) > max_diis_dim:
            t_diis.pop(0)
            e_diis.pop(0)
        if(k >= 1):
            diis_dim = len(t_diis) - 1

            # Construct diis B matrix (following Crawford Group github tutorial)
            B = np.ones((diis_dim+1, diis_dim+1)) * -1
            bsol = np.zeros(diis_dim+1)

            B[-1, -1] = 0.0
            bsol[-1] = -1.0
            for i, ei in enumerate(e_diis):
                for j, ej in enumerate(e_diis):
                    B[i,j] = np.dot(np.real(ei), np.real(ej))

            B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

            x = np.linalg.solve(B, bsol)

            t_new = np.zeros(( len(self._tamps) ))
            for l in range(diis_dim):
                temp_ary = x[l] * np.asarray(t_diis[l+1])
                t_new = np.add(t_new, temp_ary)

            self._tamps = copy.deepcopy(t_new)

    self._n_classical_params = len(self._tamps)
    self._n_cnot = self.build_Uvqc().get_num_cnots()
    self._n_pauli_trm_measures += 2*self._Nl*k*len(self._tamps) + self._Nl*k
    self._Egs = Ek

