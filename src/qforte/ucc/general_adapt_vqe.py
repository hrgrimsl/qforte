import qforte as qf

from qforte.abc.uccvqeabc import UCCVQE

import numpy as np
import scipy
import warnings
from scipy.optimize import OptimizeWarning

warnings.filterwarnings("ignore", category=OptimizeWarning)

kb = 3.1668115634564068e-06


class General_ADAPT(UCCVQE):
    def run(
        self,
        algorithm="hot-adapt-vqe",
        pool_type="GSD",
        max_depth=1000,
        opt_thresh=1e-16,
        weights=None,
        restart_file=False,
        verbose=True,
        coupling=None,
        T=0,
    ):
        """
        algorithm, string: Choices are adapt-vqe, more-adapt-vqe, and hot-adapt-vqe
        pool_type, string: operators in pool
        max_depth, int: Maximum number of operators to use in ansatz
        opt_thresh, float: gtol in bfgs
        weights, list of floats: Used in MORE-ADAPT-VQE
        restart_file, bool/string: Gives another Gibbs-ADAPT-VQE calculation to restart from
        verbose, bool: Print more detailed output than necessary?
        T, float: Temperature, only needed for HOT-ADAPT-VQE
        """
        self.algorithm = algorithm
        self.max_depth = max_depth
        self.opt_thresh = opt_thresh
        self._weights = weights
        self.Sz = qf.total_spin_z(self._nqb)
        self.S2 = qf.total_spin_squared(self._nqb)
        self._pool_type = pool_type
        self._compact_excitations = True
        self.verbose = verbose
        self.fill_pool()

        print("\n")
        print("*" * 100)
        print(f"{algorithm.upper()}".center(100))
        print("*" * 100)
        print("\n", flush=True)

        if algorithm == "more-adapt-vqe":
            if coupling == None:
                self.coupling = False
            else:
                self.coupling = coupling
            self.T = 0
            self.p = np.array(self._weights)
            self.beta = 0
        elif algorithm == "hot-adapt-vqe":
            if coupling == None:
                self.coupling = True
            else:
                self.coupling = coupling

            self.T = T
            if self.T != 0 and self.T != "Inf":
                self.beta = 1 / (kb * self.T)
            if restart_file != False:
                self.parse_existing_hot_adapt_vqe_file(restart_file)
        else:
            return NotImplementedError("Invalid choice of algorithm.")
        return self.run_hot_adapt_vqe()

    def run_hot_adapt_vqe(self):
        self.compute_F(self._tamps)
        while True:
            if self.coupling == True:
                op_grads = self.compute_dF3()
            else:
                op_grads = self.compute_uncoupled_dF3()
            idx = np.argsort(abs(op_grads))
            if self.verbose == True:
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
                self.report_dm()
                print(f"Operator {idx[-1]} has max gradient {op_grads[idx[-1]]}:")

            if len(self._tops) >= self.max_depth:
                print("Maximum number of operators already reached.", flush=True)
                return self.U, self.S, self.F

            if len(self._tops) != 0 and self._tops[-1] == idx[-1]:
                print(
                    f"HOT-ADAPT-VQE is stuck on the same operator. Aborting.",
                    flush=True,
                )
                return self.U, self.S, self.F

            self._tops.append(idx[-1])
            self._tamps = np.array(list(self._tamps) + [0.0])
            self._tamps = self.HOT_VQE(self._tamps)

    def HOT_VQE(self, x):
        print("Running HOT-VQE...\n")
        self.vqe_iter = 0
        print(f"HOT-VQE Iter.      Free Energy (Eh)     gnorm")
        while True:
            if self.coupling == True:
                dF_function = self.compute_dF
            else:
                dF_function = self.compute_uncoupled_dF
            dF_function(x)
            self.F_callback(x)
            res = scipy.optimize.minimize(
                dF_function,
                x,
                jac=True,
                callback=self.F_callback,
                options={"gtol": self.opt_thresh},
            )
            self._tamps = res.x
            return res.x

    def F_callback(self, x):
        if self.verbose == True:
            if self.algorithm == "hot-adapt-vqe":
                print(
                    f"{self.vqe_iter:>6}          {self.compute_F(x):+20.16f}        {self.dF_norm:+20.16f}",
                    flush=True,
                )
            else:
                self.compute_F(x)
                print(
                    f"{self.vqe_iter:>6}          {self.U:+20.16f}        {self.dF_norm:+20.16f}",
                    flush=True,
                )
        self.vqe_iter += 1

    def report_dm(self):
        self.compute_F(self._tamps)
        if self.algorithm == "hot-adapt-vqe":
            print(f"\nρ_{len(self._tamps)} = ")
        else:
            print(f"Effective ensemble {len(self._tamps)} = ")
        Sz, S2 = self.compute_spins(self._tamps)
        for i in range(len(self._ref)):
            print(
                f"{self.p[i]:+20.16f} |{i}><{i}| (Sz = {Sz[i]:+20.16f}, S2 = {S2[i]:20.16f}, E = {self.w[i]:+20.16f})"
            )
        print("\n")
        print(f"U:  {self.U:+20.16f}")
        print(f"Sz: {self.p.T@Sz:+20.16f}")
        print(f"S2: {self.p.T@S2:+20.16f}")
        if self.algorithm == "hot-adapt-vqe":
            print(f"S:  {self.S:20.16f}")
            print(f"F:  {self.F:+20.16f}")
        if self.coupling == True:
            print(f"\nCI Coefficients:\n")
            for i in range(self.C.shape[0]):
                print(*list(self.C[i, :]))
        print("\n")

    def compute_F(self, x):
        if self.coupling == True:
            H_eff = self.compute_H_eff(x)
            self.w, self.C = np.linalg.eigh(H_eff)
        else:
            self.w = self.compute_uncoupled_Es(x)

        if self.T == 0:
            q = np.zeros(len(self.w))
            q[0] = 1
        else:
            q = np.exp(-self.beta * (self.w - np.amin(self.w)))
        Z = np.sum(q)
        if self.algorithm == "hot-adapt-vqe":
            self.p = q / Z

        self.U = self.w.T @ self.p
        plogp = np.array([p * np.log(p) if p > 0 else 0 for p in self.p])
        self.S = -np.sum(plogp)
        if self.beta != 0:
            self.F = self.U - (1 / self.beta) * self.S
        else:
            self.F = self.U
        return self.F

    def compute_H_eff(self, x):
        sigmas = []
        kets = []
        Uvqc = self.build_Uvqc(x)
        for ref in self._ref:
            sigma = qf.Computer(ref)
            sigma.apply_circuit(Uvqc)
            kets.append(sigma.get_coeff_vec())
            sigma.apply_operator(self._qb_ham)
            sigmas.append(sigma.get_coeff_vec())
        sigmas = np.array(sigmas).real
        kets = np.array(kets).real
        H_eff = sigmas @ kets.T
        return H_eff

    def compute_uncoupled_Es(self, x):
        Uvqc = self.build_Uvqc(x)
        w = np.zeros(len(self._ref))
        for i in range(len(self._ref)):
            sigma = qf.Computer(self._ref[i])
            sigma.apply_circuit(Uvqc)
            ket = np.array(sigma.get_coeff_vec())
            sigma.apply_operator(self._qb_ham)

            w[i] = np.array(sigma.get_coeff_vec()).T.real @ ket.real

        return w

    def compute_spins(self, x):
        Sz_sigmas = []
        S2_sigmas = []
        kets = []
        Uvqc = self.build_Uvqc(x)
        for ref in self._ref:
            Sz_sigma = qf.Computer(ref)
            Sz_sigma.apply_circuit(Uvqc)
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

        if self.coupling == True:
            Sz_eff = self.C.T @ Sz_eff @ self.C
            S2_eff = self.C.T @ S2_eff @ self.C
        return np.diag(Sz_eff), np.diag(S2_eff)

    def compute_dF3(self):
        # We need to build dH[j,k,mu] = derivative of <j|U'HU|k> w.r.t theta_mu
        alphas = np.zeros((len(self._ref), len(self._pool_obj), pow(2, self._nqb)))
        sigmas = np.zeros((len(self._ref), pow(2, self._nqb)))
        Uvqc = self.build_Uvqc(self._tamps)
        # A - A'
        Kmus = []
        for mu in range(len(self._pool_obj)):
            Kmu = self._pool_obj[mu][1].jw_transform(self._qubit_excitations)
            Kmu.mult_coeffs(self._pool_obj[mu][0])
            Kmus.append(Kmu)

        for i, ref in enumerate(self._ref):
            sigma = qf.Computer(ref)
            sigma.apply_circuit(Uvqc)
            sigma.apply_operator(self._qb_ham)
            sigmas[i, :] = np.array(sigma.get_coeff_vec()).real

        for i, ref in enumerate(self._ref):
            alpha = qf.Computer(ref)
            alpha.apply_circuit(Uvqc)
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
        F = self.compute_F(x)
        # We need to build dH[j,k,mu] = derivative of <j|U'HU|k> w.r.t theta_mu
        alphas = np.zeros((len(self._ref), len(x), pow(2, self._nqb)))
        sigmas = np.zeros((len(self._ref), len(x), pow(2, self._nqb)))
        Uvqc = self.build_Uvqc(x)
        Kmus, Umus = self.get_gradient_components(x)

        for i, ref in enumerate(self._ref):
            sigma = qf.Computer(ref)
            sigma.apply_circuit(Uvqc)
            sigma.apply_operator(self._qb_ham)
            for j in range(len(self._tamps)):
                sigmas[i, -j - 1, :] = np.array(sigma.get_coeff_vec()).real
                sigma.apply_circuit(Umus[-j - 1])

        for i, ref in enumerate(self._ref):
            alpha = qf.Computer(ref)
            alpha.apply_circuit(Uvqc)
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
        return F, dF

    def compute_uncoupled_dF3(self):
        Uvqc = self.build_Uvqc(self._tamps)
        # A - A'
        Kmus = []
        for mu in range(len(self._pool_obj)):
            Kmu = self._pool_obj[mu][1].jw_transform(self._qubit_excitations)
            Kmu.mult_coeffs(self._pool_obj[mu][0])
            Kmus.append(Kmu)
        dH = np.zeros(len(self._pool_obj))

        for i, ref in enumerate(self._ref):
            sigma = qf.Computer(ref)
            sigma.apply_circuit(Uvqc)
            alpha = qf.Computer(sigma)
            sigma.apply_operator(self._qb_ham)
            for k, K in enumerate(Kmus):
                atemp = qf.Computer(alpha)
                atemp.apply_operator(K)
                dH[k] += (
                    2
                    * self.p[i]
                    * np.array(atemp.get_coeff_vec()).T.real
                    @ np.array(sigma.get_coeff_vec()).real
                )
        self.dF_norm = np.linalg.norm(dH)
        return dH

    def compute_uncoupled_dF(self, x):
        F = self.compute_F(x)
        Uvqc = self.build_Uvqc(x)
        Kmus, Umus = self.get_gradient_components(x)
        dH = np.zeros(len(x))
        for i, ref in enumerate(self._ref):
            alpha = qf.Computer(ref)
            alpha.apply_circuit(Uvqc)
            sigma = qf.Computer(alpha)
            sigma.apply_operator(self._qb_ham)
            for j in range(len(self._tamps)):
                atemp = qf.Computer(alpha)
                atemp.apply_operator(Kmus[-j - 1])
                dH[-j - 1] += (
                    2
                    * self.p[i]
                    * np.array(atemp.get_coeff_vec()).T.real
                    @ np.array(sigma.get_coeff_vec()).real
                )
                if j != len(self._tamps) - 1:
                    sigma.apply_circuit(Umus[-j - 1])
                    alpha.apply_circuit(Umus[-j - 1])
        self.dF_norm = np.linalg.norm(dH)
        return F, dH

    def get_gradient_components(self, x):
        # A-A'
        Kmus = []
        # exp(t(A-A'))
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
        return Kmus, Umus

    def parse_existing_hot_adapt_vqe_file(self, filename):
        try:
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
                    if lines[i].strip().startswith("ρ") or lines[i].strip().startswith(
                        "Effective"
                    ):
                        start_idx = i + 1
                        break

            coeffs = []

            for line in lines[start_idx:]:
                line = line.strip()
                if not line or not line[0] in "+-":
                    break
                coeff = float(line.split()[0])
                coeffs.append(coeff)
            self.p = np.array(coeffs)
        except:
            pass

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

        self.compute_F(self._tamps)

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
