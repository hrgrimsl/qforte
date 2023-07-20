"""
Algorithms to do excited states based on expansions.  (E.g. q-sc-EOM)
"""
import qforte
import numpy as np


def q_sc_eom(H, U_ref, U_manifold, ops_to_compute = []):
    """
    Quantum, self-consistent equation-of-motion method from Asthana et. al.

    H is the JW-transformed Hamiltonian.
    U_ref is the VQE ground state circuit (or some other state not to be included in the manifold)
    U_manifold is a list of unitaries to be enacted on |0> to generate U_vqe|i> for each |i>.

    ops_to_compute is a list of JW-transformed operators that we want an array in the basis of the ground and excited states.
    """
    N_qb = H.num_qubits()
    myQC = qforte.Computer(N_qb)
    myQC.apply_circuit(U_ref)
    E0 = myQC.direct_op_exp_val(H).real  
    Ek, A = ritz_eigh(H, U_manifold, verbose = False)
    print("q-sc-EOM:")
    print("*"*34)
    print(f"State:          Energy (Eh)")
    print(f"    0{E0:35.16f}")
    for i in range(0, len(Ek)):
        print(f"{(i+1):5}{Ek[i]:35.16f}")
    
    op_mats = []
    if len(ops_to_compute) > 0:
        #Add the reference state with coefficient 1.
        n_states = len(Ek) + 1
        A_plus_ref = np.zeros((n_states, n_states), dtype = "complex")
        A_plus_ref[0, 0] = 1.0
        A_plus_ref[1:,1:] = A
        all_Us = [U_ref] + U_manifold
             
        for op in ops_to_compute:
            op_vqe_basis = qforte.build_effective_operator(op, all_Us)
            op_q_sc_eom_basis = A_plus_ref.T.conj()@op_vqe_basis@A_plus_ref
            op_mats.append(op_q_sc_eom_basis.real)

    return [E0, Ek] + op_mats

def ritz_eigh(H, U, verbose = True, ops_to_compute = []):
    """
    Obtains the ritz eigeinvalues of H in the space of {U|i>}

    H is a qubit operator
    U is a list of unitaries
    ops_to_compute is a list of JW-transformed operators that we want an array in the basis of the ground and excited states.
    """
    M = qforte.build_effective_operator(H, U)
    Ek, A = np.linalg.eigh(M)
    E_pre_diag = np.diag(M).real
    if verbose == True:
        print("Ritz Diagonalization:")
        print(f"State:          Energy (Eh)     Post-Diagonalized Energy")
        for i in range(len(Ek)):
            print(f"{i:5}{E_pre_diag[i]:35.16}{Ek[i]:35.16}")

    op_mats = []
    for op in ops_to_compute:
        op_vqe_basis = qforte.build_effective_operator(op, U)
        op_ritz_basis = A.T.conj()@op_vqe_basis@A
        op_mats.append(op_ritz_basis.real)

    return [Ek, A] + op_mats
    
