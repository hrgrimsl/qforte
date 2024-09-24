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
    def run(self, ref=None, pool_type="GSD", verbose=False, max_depth=10, T_schedule = []):
        self.Sz = qf.total_spin_z(self._nqb)
        self.S2 = qf.total_spin_squared(self._nqb)
        self._pool_type = pool_type
        self._compact_excitations = True
        self.fill_pool()
        self._ref = ref
        self.T = T_schedule[-1] 
        self.T_schedule = T_schedule 
        
        self.C = None
        self.p = None
        try:
            self.beta = 1 / (kb * self.T)
        except:
            print(
                f"""T0 should be a positive float. 
                  (You can obtain T = 0 results through normal ADAPT-VQE,
                  or by using a single reference with any temperature.)"""
            )
        self.history = []

        print("*" * 30)
        print("Gibbs ADAPT-VQE\n")

        print(f"Dimension of ρ = {len(self._ref)}")
        print(f"Dimension of Fock Space = {pow(2,self._nqb)}")
        print(f"({100*len(self._ref)/pow(2, self._nqb):1.4f}% Saturation)")
        self._adapt_iter = 0
        self._tops = []
        self._tamps = []

        while len(self._tops) < max_depth:
            print("\n")
            print("*" * 32)
            print("\n", flush=True)
            print(f"ADAPT Iteration {self._adapt_iter} ({len(self._tops)} Operators)")
            print("\n")
            self.dm_update()
            self.report_dm()
            self._adapt_iter += 1
            op_grads = self.compute_dF3()
            idx = np.argsort(abs(op_grads))
            print("\n")

            if len(self._tamps) != 0 and self._tamps[-1] == 0:
                self._tamps = list(np.array(self._tamps[:-1]))
                self._tops = list(np.array(self._tops[:-1]))
                print("ADAPT-VQE cannot optimize further.")
                break

            elif len(self._tops) == 0 or self._tops[-1] != idx[-1]:
                print(f"Operator Addition Gradients:")
                print(f"Norm of Gradients: {np.linalg.norm(op_grads)}")
                print(f"Adding operator {idx[-1]} with gradient {op_grads[idx[-1]]}")
                print(self._pool_obj[idx[-1]][1])
                self._tops.append(idx[-1])
                self._tamps.append(0.0)

            else:
                print("ADAPT is attempting to add the same operator. Re-optimizing.")

            
            self._tamps = list(self.Gibbs_VQE(self._tamps))
            if np.amin(self.p) <= 1e-12:
                print("A state is missing entirely. Going hot.")    
                self.beta = 1e14
            else:
                self.beta = 1 / (kb * self.T)
            
            print("\ntoperators included from pool: \n", self._tops)
            print("\ntamplitudes for tops: \n", self._tamps)
        print(f"\nADAPT-VQE Ended With {len(self._tamps)} Operators.\n")

        print(f"Ansatz (First Operator Applied First to Reference)\n")
        for i in range(len(self._tops)):
            print(
                f"{self._tops[i]:<4}  {self._tamps[i]:+8.12f}  {self._pool_obj[self._tops[i]][1].terms()[1][2]} <--> {self._pool_obj[self._tops[i]][1].terms()[1][1]}"
            )
        print("\n")
        self.report_dm()
        print("\n")
        return self.U, self.S, self.F

    def Gibbs_VQE(self, x):
        self.vqe_iter = 0
        print(f"VQE Iter.      Free Energy (Eh)     gnorm")
        print(
            f"{self.vqe_iter:>5}          {self.compute_F(x):16.12f}      {np.linalg.norm(self.compute_dF(x)):16.12f}"
        )
        res = scipy.optimize.minimize(
            self.compute_relaxed_F,
            self._tamps,
            callback=self.F_callback,
            method="bfgs",
            options={"gtol": 1e-16, "disp": True},
            jac=self.compute_relaxed_dF,
        )
        return res.x

    def F_callback(self, x):
        self._tamps = list(x)
        self.dm_update()
        self.vqe_iter += 1
        print(
            f"{self.vqe_iter:>5}          {self.compute_F(x):16.12f}      {np.linalg.norm(self.compute_dF(x)):16.12f}"
        )
        

    def report_dm(self):
        b = copy.deepcopy(self.beta)
        self.beta = (1/(kb*self.T_schedule[-1]))
        self.dm_update()
        print("ρ = ")
        Sz, S2 = self.compute_spins(self._tamps)
        for i in range(len(self._ref)):
            print(
                f"{self.p[i]:+20.16f} |{i}><{i}| (Sz = {Sz[i]:+20.16f}, S^2 = {S2[i]:20.16f})"
            )
        print("\n")
        print(f"Current Temperature     T  = {self.T:+20.16f}")
        print(f"Internal Energy         U  = {self.U:+20.16f}")
        print(f"Entropy                 S  = {self.S:+20.16f}")
        print(f"Helmholtz Free Energy   F  = {self.F:+20.16f}")
        print(f"Thermal Averaged Sz     Sz = {self.p.T@Sz:+20.16f}")
        print(f"Thermal Averaged S2     S2 = {self.p.T@S2:+20.16f}")
        self.beta = b
        self.dm_update()

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
            w, self.C = np.linalg.eigh(H_eff)

            # Compute Boltzmann probabilities
            q = np.exp(-self.beta * (w - w[0]))
            Z = np.sum(q)
            self.p = q / Z
            self.U = w.T @ self.p
            plogp = [p * np.log(p) if p > 0 else 0 for p in self.p]
            self.S = -sum(plogp)
            self.F = self.U - (1 / self.beta) * self.S

    def compute_relaxed_F(self, x):
        self._tamps = list(x)
        self.dm_update()
        return self.compute_F(x)

    def compute_F(self, x):
        # Compute F without any relaxation of p and C.
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
            H_eff = self.C.T @ H_eff @ self.C

            # Compute Boltzmann probabilities
            U = np.diag(H_eff) @ self.p
            plogp = [p * np.log(p) if p > 0 else 0 for p in self.p]
            S = -sum(plogp)
            F = U - (1 / self.beta) * S
            return F

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

    def compute_relaxed_dF(self, x):
        self._tamps = list(x)
        self.dm_update()
        return self.compute_dF(x)

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
            try:
                assert len(self._pool_obj[self._tops[mu]][1].terms()[1]) == 3
            except:
                print("Gibbs ADAPT-VQE does not currently support ")
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
