from pytest import approx
from qforte import General_ADAPT
from qforte import system_factory
from qforte import sq_op_to_scipy
from qforte import ritz_eigh
from qforte import cisd_manifold
from qforte import build_refprep
from qforte import build_effective_array
from qforte import build_effective_symmetric_operator
from qforte import Computer

import copy
import os
import numpy as np
import scipy
import psi4

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(THIS_DIR, "lih_cas_dump.json")
if os.path.exists(data_path):
    os.remove(data_path)


class TestMOREADAPTVQE:
    def test_LiH_more_adapt_vqe(self):
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
            print_summary_file=False,
            is_multi_state=True,
            references=refs,
            compact_excitations=True,
        )

        H = sq_op_to_scipy(mol.sq_hamiltonian, alg._nqb, Sz=0, N=2).todense()

        w, v = np.linalg.eigh(H)
        full_w = w
        non_degens = [0, 1, 2, 7, 8, 15]
        w = w[non_degens]
        v = v[:, non_degens]

        dip_x_arr = sq_op_to_scipy(mol.sq_dipole_x, alg._nqb).todense()
        dip_y_arr = sq_op_to_scipy(mol.sq_dipole_y, alg._nqb).todense()
        dip_z_arr = sq_op_to_scipy(mol.sq_dipole_z, alg._nqb).todense()

        alg.run(
            pool_type="GSD",
            opt_thresh=1e-10,
            max_depth=0,
            algorithm="more-adapt-vqe",
            weights = weights,
            pool_ref = refs[0]
        )

        U = alg.build_Uvqc(amplitudes=alg._tamps)

        Es, A, ops = ritz_eigh(
             mol.hamiltonian, U, alg._ref, [mol.dipole_x, mol.dipole_y, mol.dipole_z]
        )
        dip_x, dip_y, dip_z = ops


        Es = Es[non_degens]

        for i in range(len(Es)):
            assert Es[i] == approx(w[i], abs=1.0e-10)

        
        total_dip = np.zeros(dip_x.shape)
        for op in [dip_x, dip_y, dip_z]:
            total_dip += np.multiply(op.conj(), op).real
        total_dip = np.sqrt(total_dip)
        total_dip = total_dip[np.ix_(non_degens, non_degens)]

        dip_dir = np.zeros((len(Es), len(Es)))
        for i in range(len(Es)):
            for op in [dip_x_arr, dip_y_arr, dip_z_arr]:
                sig = op @ v[:, i]
                for j in range(len(Es)):
                    dip_dir[i, j] += (
                        (sig.T.conj() @ v[:, j])[0, 0] * (v[:, j].T.conj() @ sig)[0, 0]
                    ).real
        dip_dir = np.sqrt(dip_dir)

        for i in range(len(Es)):
            for j in range(len(Es)):
                assert dip_dir[i, j] - total_dip[i, j] == approx(0.0, abs=1e-10)


        circ_refs = [build_refprep(ref) for ref in refs]

        alg = General_ADAPT(
            mol,
            print_summary_file=False,
            references = circ_refs,
            compact_excitations=True,
        )

        alg.run(
            pool_type="GSD",
            opt_thresh=1e-10,
            max_depth=1,
            algorithm="more-adapt-vqe",
            weights = weights
        )

        U = alg.build_Uvqc(amplitudes=alg._tamps)

        Es, A, ops = ritz_eigh(
            mol.hamiltonian, U, alg._ref, [mol.dipole_x, mol.dipole_y, mol.dipole_z]
        )
        dip_x, dip_y, dip_z = ops
        
        Es = Es[non_degens]

        for i in range(len(Es)):
            print(Es[i])
            print(w[i])
            assert Es[i] == approx(w[i], abs=1.0e-10)


        total_dip = np.zeros(dip_x.shape)
        for op in [dip_x, dip_y, dip_z]:
            total_dip += np.multiply(op.conj(), op).real
        total_dip = np.sqrt(total_dip)
        total_dip = total_dip[np.ix_(non_degens, non_degens)]


        for i in range(len(Es)):
            for j in range(len(Es)):
                assert dip_dir[i, j] - total_dip[i, j] == approx(0.0, abs=1e-10)


        alg = General_ADAPT(
            mol,
            print_summary_file=False,
            is_multi_state=True,
            references = refs,
            compact_excitations=False,
        )

        alg.run(
            pool_type="GSD",
            opt_thresh=1e-10,
            max_depth=1,
            algorithm="more-adapt-vqe",
            weights=weights,
        )
        alg._tamps = []
        alg._tops = []
        U = alg.build_Uvqc(amplitudes=alg._tamps)

        Es, A, ops = ritz_eigh(
            mol.hamiltonian, U, alg._ref, [mol.dipole_x, mol.dipole_y, mol.dipole_z]
        )
        dip_x, dip_y, dip_z = ops

        Es = Es[non_degens]
        for i in range(len(Es)):
            assert Es[i] == approx(w[i], abs=1.0e-10)

        total_dip = np.zeros(dip_x.shape)
        for op in [dip_x, dip_y, dip_z]:
            total_dip += np.multiply(op.conj(), op).real
        total_dip = np.sqrt(total_dip)
        total_dip = total_dip[np.ix_(non_degens, non_degens)]

        for i in range(len(Es)):
            for j in range(len(Es)):
                assert dip_dir[i, j] - total_dip[i, j] == approx(0.0, abs=1e-10)

        refs = [np.zeros((256)) for i in range(2)]
        refs[0][192] = 1.0
        refs[1][48] = 1.0
        computers = [Computer(8), Computer(8)]
        computers[0].set_coeff_vec(refs[0])
        computers[1].set_coeff_vec(refs[1])
        weights = [0.6, 0.4]

        alg = General_ADAPT(
            mol,
            print_summary_file=False,
            is_multi_state=True,
            references=computers,
            compact_excitations=False,
        )

        alg.run(
            pool_type="GSD",
            opt_thresh=1e-10,
            max_depth=30,
            algorithm="more-adapt-vqe",
            weights=weights,
        )

        Uvqc = alg.build_Uvqc(amplitudes=alg._tamps)
        H_eff = build_effective_array(
            mol.hamiltonian, Uvqc, alg._ref
        ).real
        dip_x_eff = build_effective_array(
            mol.dipole_x, Uvqc, alg._ref
        ).real
        dip_y_eff = build_effective_array(
            mol.dipole_y, Uvqc, alg._ref
        ).real
        dip_z_eff = build_effective_array(
            mol.dipole_z, Uvqc, alg._ref
        ).real

        E_more, C_more = np.linalg.eigh(H_eff)
        dip_x_more = C_more.T @ dip_x_eff @ C_more
        dip_y_more = C_more.T @ dip_y_eff @ C_more
        dip_z_more = C_more.T @ dip_z_eff @ C_more

        for i in range(len(E_more)):
            assert E_more[i] == approx(w[i], abs=1.0e-10)

        total_dip = np.zeros(dip_x_more.shape)
        for op in [dip_x_more, dip_y_more, dip_z_more]:
            total_dip += np.multiply(op, op)
        total_dip = np.sqrt(total_dip)

        for i in range(len(E_more)):
            for j in range(len(E_more)):
                assert dip_dir[i, j] - total_dip[i, j] == approx(0.0, abs=2e-7)

        alg = General_ADAPT(
            mol,
            references = computers,
            compact_excitations=True,
            state_prep_type="computer",

        )

        alg.run(
            pool_type="GSD",
            algorithm="more-adapt-vqe",
            max_depth=30,
            weights=[0.6, 0.4],
            verbose=True,
        )
        alg.coupling = True
        alg.compute_F(alg._tamps)
        for i in range(len(E_more)):
            assert E_more[i] == approx(alg.w[i], abs=1.0e-10)

        spaces = [[1, 0, 0, 0], [2, 0, 0, 0], [1, 0, 1, 1]]
        mol = system_factory(
            system_type="molecule",
            mol_geometry=geom,
            build_type="psi4",
            basis="sto-6g",
            dipole=True,
            symmetry="C2v",
            casscf=spaces,
            no_com=True,
            no_reorient=True,
            json_dump=data_path,
        )

        mol = system_factory(build_type="external", filename=data_path)

        occ_refs = [
            [1, 1, 1, 1, 0, 0] + [0] * 6,
            [1, 1, 0, 0, 1, 1] + [0] * 6,
            [1, 1, 0, 1, 1, 0] + [0] * 6,
            [1, 1, 1, 0, 0, 1] + [0] * 6,
        ]

        alg = General_ADAPT(
            mol,
            print_summary_file=False,
            is_multi_state=True,
            references = occ_refs,
            compact_excitations=True,
        )

        H_eff = build_effective_symmetric_operator(
            12, mol.hamiltonian, alg._refprep
        ).real
        E_casscf = np.linalg.eigh(H_eff)[0][0]
        assert E_casscf == approx(-7.873605319132174, 1e-8)

        alg.run(pool_type="GSD", max_depth=3, algorithm="more-adapt-vqe")

        correct_Es = [
            -7.8593451680521662,
            -7.7158474059591828,
            -7.6836465355578119,
            -7.2065322341909379,
        ]

        for i in range(4):
            assert correct_Es[i] == approx(alg._diag_energies[-1][i])

        comp_refs = [Computer(12) for i in range(4)]

        coeff_vec = np.zeros(2**12)
        coeff_vec[int("001111", 2)] = 1
        comp_refs[0].set_coeff_vec(copy.deepcopy(coeff_vec))

        coeff_vec = np.zeros(2**12)
        coeff_vec[int("110011", 2)] = 1
        comp_refs[1].set_coeff_vec(copy.deepcopy(coeff_vec))

        coeff_vec = np.zeros(2**12)
        coeff_vec[int("100111", 2)] = 1
        comp_refs[2].set_coeff_vec(copy.deepcopy(coeff_vec))

        coeff_vec = np.zeros(2**12)
        coeff_vec[int("011011", 2)] = 1
        comp_refs[3].set_coeff_vec(copy.deepcopy(coeff_vec))

        alg = ADAPTVQE(
            mol,
            print_summary_file=False,
            is_multi_state=True,
            references = comp_refs,
            compact_excitations=True,
            state_prep_type="computer",
        )

        alg.run(pool_type="GSD", max_depth=3, weights=[0.25] * 4, algorithm="more-adapt-vqe")
        for i in range(4):
            assert correct_Es[i] == approx(alg._diag_energies[-1][i])

        alg = General_ADAPT(
            mol,
            references=comp_refs,
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

        alg.coupling = True
        alg.compute_F(alg._tamps)
        for i in range(4):
            print(i)
            assert correct_Es[i] == approx(alg.w[i])

        alg._tamps = np.array([0.5, 0.75, 1])

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

        circ_refs = []

        circ_refs.append(build_refprep([1] * 2 + [1, 1, 0, 0] + [0] * 6))
        circ_refs.append(build_refprep([1] * 2 + [0, 0, 1, 1] + [0] * 6))
        circ_refs.append(build_refprep([1] * 2 + [0, 1, 1, 0] + [0] * 6))
        circ_refs.append(build_refprep([1] * 2 + [1, 0, 0, 1] + [0] * 6))

        alg = General_ADAPT(
            mol,
            print_summary_file=False,
            is_multi_state=True,
            references = circ_refs,
            compact_excitations=True,
            state_prep_type="unitary_circ",
        )

        alg.run(pool_type="GSD", max_depth=3, weights=[0.25] * 4, algorithm="more-adapt-vqe")
        for i in range(4):
            assert correct_Es[i] == approx(alg._diag_energies[-1][i])

        psi4.core.clean_options()
