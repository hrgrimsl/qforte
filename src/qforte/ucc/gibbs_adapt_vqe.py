"""
Classes for Gibbs State ADAPT-VQE
====================================
"""

import qforte as qf

from qforte.abc.uccvqeabc import UCCVQE

import numpy as np
import scipy
import os

import warnings
from scipy.optimize import OptimizeWarning

warnings.filterwarnings("ignore", category=OptimizeWarning)


kb = 3.1668115634564068e-06


class Gibbs_ADAPT(UCCVQE):
    def run(
        self,
        pool_type="GSD",
        T=0,
        max_depth=10,
        opt_thresh=1e-16,
        restart_file=False,
        verbose=True,
        freeze_pC = True
    ):
        """
        pool_type, string: operators in pool
        T, float: temperature in K
        max_depth, int: Maximum number of operators to use in ansatz
        opt_thresh, float: gtol in bfgs
        restart_file, bool/string: Gives another Gibbs-ADAPT-VQE calculation to restart from.
        verbose, bool: Print more detailed output than necessary?
        """

        self.opt_thresh = opt_thresh
        self.Sz = qf.total_spin_z(self._nqb)
        self.S2 = qf.total_spin_squared(self._nqb)
        self._pool_type = pool_type
        self._compact_excitations = True
        self.verbose = verbose
        self.freeze_pC = freeze_pC
        self.fill_pool()

        self.T = T
        if self.T != 0:
            self.beta = 1 / (kb * self.T)

        print("\n")
        print("*" * 100)
        print("HOT-ADAPT-VQE".center(100))
        print("Code by H.R. Grimsley".center(100))
        print("*" * 100)
        print("\n", flush=True)

        self._summary_string = "Summary of HOT-ADAPT-VQE Iterations:\n"
        self._summary_string += "-" * 100
        self._summary_string += (
            f"\n{'Iter.':>8} {'U':>20} {'S':>20} {'F':>20} {'gmax':>20}"
        )


        if restart_file != False:
            self.parse_existing_file(restart_file)
            self.compute_F(self._tamps, assign=True)
        else:
            self.dm_update()

        while len(self._tops) <= max_depth:
            op_grads = self.compute_dF3()
            idx = np.argsort(abs(op_grads))

            self._summary_string += f"\n{len(self._tamps):8d} {self.U:+20.16f} {self.S:+20.16f} {self.F:+20.16f} {op_grads[idx[-1]]:+20.16f}"
            print(self._summary_string)
            print("-" * 100, flush=True)

            if self.verbose == True:
                self.report_dm()
                print(f"Operator {idx[-1]} has max gradient {op_grads[idx[-1]]}:")
                print(f"{self._pool_obj[idx[-1]][1]}\n")

            if len(self._tops) == max_depth:
                print("Maximum number of operators reached.")
                return self.U, self.S, self.F

            if len(self._tops) != 0 and self._tops[-1] == idx[-1]:
                print(
                    f"""PEPSI-ADAPT-VQE is stuck on the same operator.
                      Aborting instead of adding re-adding it.
                      No re-optimization will take place."""
                )
                return self.U, self.S, self.F
            else:
                self._tops.append(idx[-1])
                self._tamps = np.array(list(self._tamps) + [0.0])
                self._tamps = self.Gibbs_VQE(self._tamps)

                if verbose == True:
                    print(f"\nOperators at {len(self._tops)} iterations:", *self._tops)
                    print(
                        f"\nAmplitudes at {len(self._tops)} iterations:",
                        *list(self._tamps),
                    )

                print(f"\nIteration {len(self._tops)} Ansatz:\n")
                print("-" * 50)
                for i in range(len(self._tops)):
                    print(
                        f"{self._tops[i]:<4}  {self._tamps[i]:+8.12f}  {self._pool_obj[self._tops[i]][1].terms()[1][2]} <--> {self._pool_obj[self._tops[i]][1].terms()[1][1]}"
                    )
                print("-" * 50)
                print("\n")
        return self.U, self.S, self.F

    def Gibbs_VQE(self, x):
        print("Running HOT-VQE...\n")
        prev_res = self.compute_F(x)
        self.vqe_iter = 0
        print(f"HOT-VQE Iter.      Free Energy (Eh)     gnorm")
        while True:
            self.compute_dF(x)
            self.F_callback(x)
            prev_res = self.F
            res = scipy.optimize.minimize(
                self.compute_F,
                x,
                jac=self.compute_dF,
                callback=self.F_callback,
                method="bfgs",
                options={"gtol": self.opt_thresh},
            )
            x = res.x
            self._tamps = res.x
            print("Updating ensemble...", flush=True)
            self.dm_update()
            if abs(self.F - prev_res) < 1e-16 or self.freeze_pC == False:
                print("HOT-VQE Done.", flush=True)
                return res.x
            

    def F_callback(self, x):
        print(
            f"{self.vqe_iter:>6}          {self.compute_F(x):+20.16f}        {self.dF_norm:+20.16f}"
        )
        self.vqe_iter += 1

    def report_dm(self):
        print("ρ = ")
        Sz, S2 = self.compute_spins(self._tamps)
        for i in range(len(self._ref)):
            print(
                f"{self.p[i]:+20.16f} |{i}><{i}| (Sz = {Sz[i]:+20.16f}, S^2 = {S2[i]:20.16f}, Energy = {self.w[i]:+20.16f})"
            )
        print("\n")
        print(f"Internal Energy         U  = {self.U:+20.16f}")
        print(f"Entropy                 S  = {self.S:20.16f}")
        print(f"Helmholtz Free Energy   F  = {self.F:+20.16f}")
        print(f"Thermal Averaged Sz     Sz = {self.p.T@Sz:+20.16f}")
        print(f"Thermal Averaged S2     S2 = {self.p.T@S2:+20.16f}")
        print(f"\nCI Coefficients at {len(self._tamps)} ADAPT iterations:\n")
        for i in range(self.C.shape[0]):
            print(*list(self.C[i, :]))
        print("\n")

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
            if self.T == 0:
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

    def compute_F(self, x, assign=False):
        if self.freeze_pC == False:
            self._tamps = x
            self.dm_update()
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
            w = np.diag(self.C.T @ H_eff @ self.C)

            if assign:
                self.U = w @ self.p
                plogp = [p * np.log(p) if p > 0 else 0 for p in self.p]
                self.S = -sum(plogp)

            if self.T != "Inf":
                F = w @ self.p - (1 / self.beta) * self.S
            else:
                F = w @ self.p

            if assign:
                self.w = w
                self.F = F
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
        if self.freeze_pC == False:
            self.dm_update()
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
        if self.freeze_pC == False:
            self._tamps = x
            self.dm_update()
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

    def parse_existing_file(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].startswith("CI Coefficients"):
                    start_idx = i + 2
                    break
        first_line = lines[start_idx].strip().split()
        n = len(first_line)
        block_lines = lines[start_idx : start_idx + n]
        data = [list(map(float, line.strip().split())) for line in block_lines]
        self.C = np.array(data)

        with open(filename, "r") as f:
            lines = f.readlines()
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith("ρ ="):
                    start_idx = i + 1
                    break
        coeffs = []
        for line in lines[start_idx:]:
            line = line.strip()
            if not line or not line.startswith(("+", "-")):
                break
            coeff = float(line.split()[0])
            coeffs.append(coeff)
        self.p = np.array(coeffs)

        with open(filename, "r") as f:
            lines = f.readlines()
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].startswith("Amplitudes at"):
                    self._tamps = np.array(
                        list(map(float, lines[i].split(":")[1].strip().split()))
                    )
                    break
        with open(filename, "r") as f:
            lines = f.readlines()
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].startswith("Operators at"):
                    self._tops = list(map(int, lines[i].split(":")[1].strip().split()))
                    break

        assert len(self._tops) == len(self._tamps)
        assert len(self.p) == self.C.shape[0] == self.C.shape[1] == len(self._ref)

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
