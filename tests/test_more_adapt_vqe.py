from pytest import approx
from qforte import General_ADAPT
from qforte import system_factory
from qforte import sq_op_to_scipy
from qforte import ritz_eigh
from qforte import cisd_manifold
from qforte import build_refprep
from qforte import Computer


import os
import numpy as np
import scipy


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(THIS_DIR, "lih_cas_dump.json")
if os.path.exists(data_path):
    os.remove(data_path)


class TestMOREADAPTVQE:
    def test_trivial_occupation_ref(self):
        geom = [("Li", (0, 0, 0)), ("H", (0, 0, 1))]
        mol = system_factory(
            system_type="molecule",
            mol_geometry=geom,
            build_type="psi4",
            basis="sto-3g",
            dipole=True,
            num_frozen_docc=1,
            num_frozen_uocc=1,
            symmetry="C2v",
        )

        refs = [mol.hf_reference] + cisd_manifold(mol.hf_reference)
        weights = [2 ** (-i - 1) for i in range(len(refs))]
        weights[-1] += 2 ** (-len(weights))

        alg = General_ADAPT(
            mol,
            references=refs,
            compact_excitations=True,
        )

        H_arr = sq_op_to_scipy(mol.sq_hamiltonian, alg._nqb, Sz=0, N=2).todense()
        dip_x_arr = sq_op_to_scipy(mol.sq_dipole_x, alg._nqb).todense()
        dip_y_arr = sq_op_to_scipy(mol.sq_dipole_y, alg._nqb).todense()
        dip_z_arr = sq_op_to_scipy(mol.sq_dipole_z, alg._nqb).todense()

        E_direct, C_direct = np.linalg.eigh(H_arr)

        alg.run(
            pool_type="GSD",
            opt_thresh=1e-10,
            max_depth=0,
            algorithm="more-adapt-vqe",
            weights=weights,
            pool_ref=refs,
        )

        U = alg.build_Uvqc(amplitudes=alg._tamps)

        E_ritz, C_ritz, ops = ritz_eigh(
            mol.hamiltonian, U, alg._ref, [mol.dipole_x, mol.dipole_y, mol.dipole_z]
        )

        for i in range(len(E_ritz)):
            assert E_ritz[i] == approx(E_direct[i])

        dip_x_ritz, dip_y_ritz, dip_z_ritz = ops

        total_dip_ritz = np.zeros(dip_x_ritz.shape)
        for op in [dip_x_ritz, dip_y_ritz, dip_z_ritz]:
            total_dip_ritz += np.multiply(op.conj(), op).real
        total_dip_ritz = np.sqrt(total_dip_ritz)

        total_dip_direct = np.zeros((len(E_ritz), len(E_ritz)))
        for i in range(len(E_ritz)):
            for op in [dip_x_arr, dip_y_arr, dip_z_arr]:
                sig = op @ C_direct[:, i]
                for j in range(len(E_ritz)):
                    total_dip_direct[i, j] += (
                        (sig.T.conj() @ C_direct[:, j])[0, 0]
                        * (C_direct[:, j].T.conj() @ sig)[0, 0]
                    ).real
        total_dip_direct = np.sqrt(total_dip_direct)

        for i in [0, 1, 2, 8, 9, 15]:
            for j in [0, 1, 2, 8, 9, 15]:
                assert total_dip_direct[i, j] == approx(total_dip_ritz[i, j])

    def test_trivial_circuit_ref(self):
        geom = [("Li", (0, 0, 0)), ("H", (0, 0, 1))]
        mol = system_factory(
            system_type="molecule",
            mol_geometry=geom,
            build_type="psi4",
            basis="sto-3g",
            dipole=True,
            num_frozen_docc=1,
            num_frozen_uocc=1,
            symmetry="C2v",
        )

        refs = [mol.hf_reference] + cisd_manifold(mol.hf_reference)
        refs = [build_refprep(ref) for ref in refs]

        weights = [2 ** (-i - 1) for i in range(len(refs))]
        weights[-1] += 2 ** (-len(weights))

        alg = General_ADAPT(
            mol,
            references=refs,
        )

        H_arr = sq_op_to_scipy(mol.sq_hamiltonian, alg._nqb, Sz=0, N=2).todense()
        dip_x_arr = sq_op_to_scipy(mol.sq_dipole_x, alg._nqb).todense()
        dip_y_arr = sq_op_to_scipy(mol.sq_dipole_y, alg._nqb).todense()
        dip_z_arr = sq_op_to_scipy(mol.sq_dipole_z, alg._nqb).todense()

        E_direct, C_direct = np.linalg.eigh(H_arr)

        alg.run(
            pool_type="GSD",
            opt_thresh=1e-10,
            max_depth=0,
            algorithm="more-adapt-vqe",
            weights=weights,
            pool_ref=refs,
        )

        U = alg.build_Uvqc(amplitudes=alg._tamps)

        E_ritz, C_ritz, ops = ritz_eigh(
            mol.hamiltonian, U, alg._ref, [mol.dipole_x, mol.dipole_y, mol.dipole_z]
        )
        for i in range(len(E_ritz)):
            assert E_ritz[i] == approx(E_direct[i])

        dip_x_ritz, dip_y_ritz, dip_z_ritz = ops

        total_dip_ritz = np.zeros(dip_x_ritz.shape)
        for op in [dip_x_ritz, dip_y_ritz, dip_z_ritz]:
            total_dip_ritz += np.multiply(op.conj(), op).real
        total_dip_ritz = np.sqrt(total_dip_ritz)

        total_dip_direct = np.zeros((len(E_ritz), len(E_ritz)))
        for i in range(len(E_ritz)):
            for op in [dip_x_arr, dip_y_arr, dip_z_arr]:
                sig = op @ C_direct[:, i]
                for j in range(len(E_ritz)):
                    total_dip_direct[i, j] += (
                        (sig.T.conj() @ C_direct[:, j])[0, 0]
                        * (C_direct[:, j].T.conj() @ sig)[0, 0]
                    ).real
        total_dip_direct = np.sqrt(total_dip_direct)

        for i in [0, 1, 2, 8, 9, 15]:
            for j in [0, 1, 2, 8, 9, 15]:
                assert total_dip_direct[i, j] == approx(total_dip_ritz[i, j])

    def test_full_computer_ref(self):
        geom = [("Li", (0, 0, 0)), ("H", (0, 0, 1))]
        mol = system_factory(
            system_type="molecule",
            mol_geometry=geom,
            build_type="psi4",
            basis="sto-3g",
            dipole=True,
            num_frozen_docc=1,
            num_frozen_uocc=1,
            symmetry="C2v",
        )

        refs = [np.zeros(256), np.zeros(256)]
        refs[0][3] = 1.0
        refs[1][12] = 1.0
        computers = [Computer(8), Computer(8)]
        computers[0].set_coeff_vec(refs[0])
        computers[1].set_coeff_vec(refs[1])

        alg = General_ADAPT(
            mol,
            references=computers,
        )

        H_arr = sq_op_to_scipy(mol.sq_hamiltonian, alg._nqb, Sz=0, N=2).todense()
        dip_x_arr = sq_op_to_scipy(mol.sq_dipole_x, alg._nqb).todense()
        dip_y_arr = sq_op_to_scipy(mol.sq_dipole_y, alg._nqb).todense()
        dip_z_arr = sq_op_to_scipy(mol.sq_dipole_z, alg._nqb).todense()

        E_direct, C_direct = np.linalg.eigh(H_arr)

        alg.run(
            pool_type="GSD",
            opt_thresh=1e-10,
            max_depth=100,
            algorithm="more-adapt-vqe",
            weights=[0.6, 0.4],
        )

        U = alg.build_Uvqc(amplitudes=alg._tamps)

        E_ritz, C_ritz, ops = ritz_eigh(
            mol.hamiltonian, U, alg._ref, [mol.dipole_x, mol.dipole_y, mol.dipole_z]
        )

        for i in range(len(E_ritz)):
            assert E_ritz[i] == approx(E_direct[i])

        dip_x_ritz, dip_y_ritz, dip_z_ritz = ops

        total_dip_ritz = np.zeros(dip_x_ritz.shape)
        for op in [dip_x_ritz, dip_y_ritz, dip_z_ritz]:
            total_dip_ritz += np.multiply(op.conj(), op).real
        total_dip_ritz = np.sqrt(total_dip_ritz)

        total_dip_direct = np.zeros((len(E_ritz), len(E_ritz)))
        for i in range(len(E_ritz)):
            for op in [dip_x_arr, dip_y_arr, dip_z_arr]:
                sig = op @ C_direct[:, i]
                for j in range(len(E_ritz)):
                    total_dip_direct[i, j] += (
                        (sig.T.conj() @ C_direct[:, j])[0, 0]
                        * (C_direct[:, j].T.conj() @ sig)[0, 0]
                    ).real
        total_dip_direct = np.sqrt(total_dip_direct)

        for i in [0, 1]:
            for j in [0, 1]:
                assert total_dip_direct[i, j] == approx(
                    total_dip_ritz[i, j], abs=1.0e-7
                )

    def test_gradients(self):
        geom = [("Li", (0, 0, 0)), ("H", (0, 0, 1))]
        mol = system_factory(
            system_type="molecule",
            mol_geometry=geom,
            build_type="psi4",
            basis="sto-3g",
            dipole=True,
            num_frozen_docc=1,
            num_frozen_uocc=1,
            symmetry="C2v",
        )

        alg = General_ADAPT(
            mol,
            references=[
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 0, 0, 1, 1],
                [0, 0, 1, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 0, 1, 1],
            ],
            compact_excitations=True,
            state_prep_type="computer",
        )

        alg.run(
            pool_type="GSD",
            algorithm="more-adapt-vqe",
            max_depth=3,
            weights=[0.25] * 4,
            verbose=True,
        )

        h = 1e-6
        alg._tamps = np.array([0.5, 0.75, 1])

        alg.coupling = True
        alg.compute_F(alg._tamps)
        dF_numerical = scipy.optimize.approx_fprime(
            alg._tamps, alg.compute_F, epsilon=h
        )
        dF_analytical = alg.compute_dF(alg._tamps)[1]
        assert np.linalg.norm(dF_numerical - dF_analytical) == approx(0, abs=1e-6)

        alg.coupling = False
        alg.compute_F(alg._tamps)
        dF_numerical = scipy.optimize.approx_fprime(
            alg._tamps, alg.compute_F, epsilon=h
        )
        dF_analytical = alg.compute_uncoupled_dF(alg._tamps)[1]
        assert np.linalg.norm(dF_numerical - dF_analytical) == approx(0, abs=1e-6)
