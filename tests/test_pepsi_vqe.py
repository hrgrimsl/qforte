from pytest import approx
import scipy
import qforte as qf
import numpy as np
import copy


class TestPEPSIADAPTVQE:
    def test_pepsi_adapt_vqe(self):
        geom = [("F", (0, 0, 0)), ("H", (0, 0, 0.9168))]

        mol = qf.system_factory(
            system_type="molecule",
            mol_geometry=geom,
            build_type="psi4",
            basis="sto-3g",
            dipole=True,
            num_frozen_docc=0,
            num_frozen_uocc=0,
            symmetry="C2v",
        )

        ref_dets = [
            int("001111111111", 2),
            int("101101111111", 2),
            int("101110111111", 2),
            int("011101111111", 2),
            int("011110111111", 2),
            int("101011111111", 2),
            int("100111111111", 2),
            int("011011111111", 2),
            int("010111111111", 2),
        ]

        refs = []
        for i in ref_dets:
            qc = qf.Computer(12)
            v1 = np.zeros(pow(2, 12), dtype=np.complex128)
            v1[i] = 1
            qc.set_coeff_vec(v1)
            refs.append(qc)

        alg = qf.Gibbs_ADAPT(
            mol,
            state_prep_type="computer",
            is_multi_state=True,
            weights=[1 / len(refs)] * len(refs),
            reference=refs,
        )

        U, S, F = alg.run(pool_type="GSD", T=20000, max_depth=2)

        alg.dm_update()

        h = 1e-6

        F = alg.compute_F(alg._tamps)
        dF_numerical = scipy.optimize.approx_fprime(
            alg._tamps, alg.compute_F, epsilon=h
        )

        dF_analytical = alg.compute_dF(alg._tamps)

        assert U == approx(-98.58884089990063, abs = 1e-10)
        assert S == approx(0.039076852094959605, abs = 1e-10)
        assert F == approx(-98.59131588044217, abs = 1e-10)
        assert np.linalg.norm(dF_numerical - dF_analytical) == approx(0, abs = 1e-5)

if __name__ == "__main__":
    test = TestPEPSIADAPTVQE()
    test.test_pepsi_adapt_vqe()
