"""
Classes for Gibbs State ADAPT-VQE
====================================
"""

import qforte as qf

from qforte.abc.uccvqeabc import UCCVQE

import numpy as np
import scipy
import copy

kb = 3.1668115634564068e-06


class Gibbs_ADAPT(UCCVQE):
    def run(self, ref=None, pool_type="GSD", max_depth=10, T = 0, opt_thresh = 1e-16):
        self.opt_thresh = opt_thresh
        self.Sz = qf.total_spin_z(self._nqb)
        self.S2 = qf.total_spin_squared(self._nqb)
        self._pool_type = pool_type
        self._compact_excitations = True
        self.fill_pool()
        self._ref = ref
        self.T = T
         
        self.C = None
        self.p = None
        
        if self.T != 0 and self.T != "Inf":
            self.beta = 1 / (kb * self.T)
        if self.T == "Inf":
            self.beta = 0
        print("*" * 30)
        print("PEPSI-ADAPT-VQE\n")        
        print("*" * 30)
        self._adapt_iter = 0
        self._tops = []
        self._tamps = []

        while len(self._tops) < max_depth:
            print("\n")
            print("*" * 32)
            print("\n", flush=True)
            print(f"ADAPT Iteration {self._adapt_iter}")
            print("\n")
            self.dm_update()
            self.report_dm()
            
            self._adapt_iter += 1
            op_grads = self.compute_dF3()
            idx = np.argsort(abs(op_grads))
            print("\n")

            if len(self._tops) != 0 and self._tops[-1] == idx[-1]:
                print(f"PEPSI-ADAPT-VQE is stuck on the same operator.  Aborting.")                

            else:
                print(f"Operator Addition Gradients:")
                print(f"Norm of Gradients: {np.linalg.norm(op_grads)}")
                print(f"Adding operator {idx[-1]} with gradient {op_grads[idx[-1]]}")
                print(self._pool_obj[idx[-1]][1])
                self._tops.append(idx[-1])
                
                self._tamps = np.array(list(self._tamps) + [0.0])
                self._tamps = self.Gibbs_VQE(self._tamps)

                print(f"\nOperators at {self._adapt_iter} iterations:", *self._tops)
                print(f"\nAmplitudes at {self._adapt_iter} iterations:", *self._tamps)
        
                print(f"Ansatz (First Operator Applied First to Reference)\n")
                for i in range(len(self._tops)):
                    print(
                        f"{self._tops[i]:<4}  {self._tamps[i]:+8.12f}  {self._pool_obj[self._tops[i]][1].terms()[1][2]} <--> {self._pool_obj[self._tops[i]][1].terms()[1][1]}"
                        )
                print("\n")
        return self.U, self.S, self.F

    def Gibbs_VQE(self, x):
        self.vqe_iter = 0
        print(f"VQE Iter.      Free Energy (Eh)     gnorm")
        self.F = self.compute_F(x)
        self.compute_dF(x)
        self.F_callback(x)
        res = scipy.optimize.minimize(
            self.compute_F,
            self._tamps,
            jac = self.compute_dF,  
            callback=self.F_callback,
            method="bfgs",
            options={"gtol": self.opt_thresh, "disp": True}
        )
        return res.x

    def F_callback(self, x):
        self.vqe_iter += 1
        print(f"{self.vqe_iter:>6}          {self.F:+20.16f}        {self.dF_norm:+20.16f}")     
        
    def report_dm(self):
        print("ρ = ")
        Sz, S2 = self.compute_spins(self._tamps)
        for i in range(len(self._ref)):
            print(
                f"{self.p[i]:+20.16f} |{i}><{i}| (Sz = {Sz[i]:+20.16f}, S^2 = {S2[i]:20.16f}, Energy = {self.w[i]})"
            )
        print("\n")
        print(f"Internal Energy         U  = {self.U:+20.16f}")
        print(f"Entropy                 S  = {self.S:+20.16f}")
        print(f"Helmholtz Free Energy   F  = {self.F:+20.16f}")
        print(f"Thermal Averaged Sz     Sz = {self.p.T@Sz:+20.16f}")
        print(f"Thermal Averaged S2     S2 = {self.p.T@S2:+20.16f}")
        
    def dm_update(self):
        if self._state_prep_type == "computer":
            sigmas = []
            kets = []
            # Diagonalize effective H in subspace
            U = self.build_Uvqc()
            for i, det in enumerate(self._ref):
                sigma = qf.Computer(self._nqb)
                sigma.set_coeff_vec(det.get_coeff_vec())
                sigma.apply_circuit(U[i])
                kets.append(sigma.get_coeff_vec())
                sigma.apply_operator(self._qb_ham)
                sigmas.append(sigma.get_coeff_vec())
            sigma = np.array(sigmas).real
            kets = np.array(kets).real
            H_eff = sigma @ kets.T
            self.w, self.C = np.linalg.eigh(H_eff)
            # Compute Boltzmann probabilities
            if self.T == "Inf":
                q = np.ones(len(self.w))/len(self.w)
            elif self.T == 0:
                q = np.zeros(len(self.w))
                q[0] = 1
            else:
                q = np.exp(-self.beta * (self.w - self.w[0]))
            Z = np.sum(q)
            self.p = q / Z
            self.U = self.w.T @ self.p
            plogp = [p * np.log(p) if p > 0 else 0 for p in self.p]
            self.S = -sum(plogp)
            self.F = self.U - (1 / self.beta) * self.S
    
    def compute_F(self, x): 
        if self._state_prep_type == "computer":
            sigmas = []
            kets = []
            U = self.build_Uvqc(x)
            for i, det in enumerate(self._ref):
                sigma = qf.Computer(self._nqb)
                sigma.set_coeff_vec(det.get_coeff_vec())
                sigma.apply_circuit(U[i])
                kets.append(sigma.get_coeff_vec())
                sigma.apply_operator(self._qb_ham)
                sigmas.append(sigma.get_coeff_vec())
            sigma = np.array(sigmas).real
            kets = np.array(kets).real
            H_eff = sigma @ kets.T
            self.w, self.C = np.linalg.eigh(H_eff)
            # Compute Boltzmann probabilities
            if self.T == "Inf":
                q = np.ones(len(self.w))/len(self.w)
            elif self.T == 0:
                q = np.zeros(len(self.w))
                q[0] = 1
            else:
                q = np.exp(-self.beta * (self.w - self.w[0]))
            Z = np.sum(q)
            self.p = q / Z
            self.U = self.w.T @ self.p
            plogp = [p * np.log(p) if p > 0 else 0 for p in self.p]
            self.S = -sum(plogp)
            self.F = self.U - (1 / self.beta) * self.S
            H_eff = self.C.T @ H_eff @ self.C
            return self.F

    def compute_spins(self, x):
        Sz_sigmas = []
        S2_sigmas = []
        kets = []
        U = self.build_Uvqc(x)
        for i, det in enumerate(self._ref):
            Sz_sigma = qf.Computer(self._nqb)
            Sz_sigma.set_coeff_vec(det.get_coeff_vec())
            Sz_sigma.apply_circuit(U[i])
            S2_sigma = qf.Computer(Sz_sigma)
            kets.append(Sz_sigma.get_coeff_vec())
            Sz_sigma.apply_operator(self.Sz)
            Sz_sigmas.append(Sz_sigma.get_coeff_vec())
            S2_sigma.apply_operator(self.S2)
            S2_sigmas.append(S2_sigma.get_coeff_vec())
        Sz_sigma = np.array(Sz_sigmas).real
        S2_sigma = np.array(S2_sigmas).real
        kets = np.array(kets).real
        Sz_eff = Sz_sigma @ kets.T
        S2_eff = S2_sigma @ kets.T
        Sz_eff = self.C.T @ Sz_eff @ self.C
        S2_eff = self.C.T @ S2_eff @ self.C
        return np.diag(Sz_eff), np.diag(S2_eff)
    
    def compute_dF3(self):
        # We need to build dH[j,k,mu] = derivative of <j|U'HU|k> w.r.t theta_mu

        alphas = np.zeros((len(self._ref), len(self._pool_obj), pow(2, self._nqb)))
        sigmas = np.zeros((len(self._ref), pow(2, self._nqb)))
        U = self.build_Uvqc(self._tamps)
        # A - A'
        Kmus = []
        for mu in range(len(self._pool_obj)):
            Kmu = self._pool_obj[mu][1].jw_transform(self._qubit_excitations)
            Kmu.mult_coeffs(self._pool_obj[mu][0])
            Kmus.append(Kmu)

        if self._state_prep_type == "computer":
            for i, ref in enumerate(self._ref):
                sigma = qf.Computer(ref)
                sigma.apply_circuit(U[i])
                sigma.apply_operator(self._qb_ham)

                sigmas[i, :] = np.array(sigma.get_coeff_vec()).real

            for i, ref in enumerate(self._ref):
                alpha = qf.Computer(ref)
                alpha.apply_circuit(U[i])
                for j in range(len(Kmus)):
                    atemp = qf.Computer(alpha)
                    atemp.apply_operator(Kmus[j])

                    alphas[i, j, :] = np.array(atemp.get_coeff_vec()).real

        dH = np.einsum("iv,juv->iju", sigmas, alphas)
        dH += np.einsum("jv,iuv->iju", sigmas, alphas)

        dF = np.einsum("ji,jku,ki->iu", self.C, dH, self.C)
        dF = np.einsum("i,iu->u", self.p, dF)
        return dF

    def compute_dF(self, x):
        # We need to build dH[j,k,mu] = derivative of <j|U'HU|k> w.r.t theta_mu

        alphas = np.zeros((len(self._ref), len(x), pow(2, self._nqb)))
        sigmas = np.zeros((len(self._ref), len(x), pow(2, self._nqb)))
        U = self.build_Uvqc(x)
        # A - A'
        Kmus = []
        # Exp(-t_mu(A - A'))
        Umus = []
        for mu, t in enumerate(x):
            Kmu = self._pool_obj[self._tops[mu]][1].jw_transform(
                self._qubit_excitations
            )
            Kmu.mult_coeffs(self._pool_obj[self._tops[mu]][0])
            Kmus.append(Kmu)
            Umu = qf.Circuit()
            Umu.add(
                qf.compact_excitation_circuit(
                    -t * self._pool_obj[self._tops[mu]][1].terms()[1][0],
                    self._pool_obj[self._tops[mu]][1].terms()[1][1],
                    self._pool_obj[self._tops[mu]][1].terms()[1][2],
                    self._qubit_excitations,
                )
            )
            Umus.append(Umu)
        if self._state_prep_type == "computer":
            for i, ref in enumerate(self._ref):
                sigma = qf.Computer(ref)
                sigma.apply_circuit(U[i])
                sigma.apply_operator(self._qb_ham)
                for j in range(len(self._tamps)):
                    sigmas[i, -j - 1, :] = np.array(sigma.get_coeff_vec()).real
                    sigma.apply_circuit(Umus[-j - 1])

            for i, ref in enumerate(self._ref):
                alpha = qf.Computer(ref)
                alpha.apply_circuit(U[i])
                for j in range(len(self._tamps)):
                    atemp = qf.Computer(alpha)
                    atemp.apply_operator(Kmus[-j - 1])
                    alphas[i, -j - 1, :] = np.array(atemp.get_coeff_vec()).real
                    alpha.apply_circuit(Umus[-j - 1])

        dH = np.einsum("iuv,juv->iju", sigmas, alphas)
        dH += np.einsum("juv,iuv->iju", sigmas, alphas)
        dF = np.einsum("ji,jku,ki->iu", self.C, dH, self.C)
        dF = np.einsum("i,iu->u", self.p, dF)
        self.dF_norm = np.linalg.norm(dF)
        return dF

    def get_num_commut_measurements(self):
        pass

    def get_num_ham_measurements(self):
        pass

    def print_options_banner(self):
        pass

    def print_summary_banner(self):
        pass

    def run_realistic(self):
        pass

    def solve(self):
        pass

    def verify_run(self):
        pass
