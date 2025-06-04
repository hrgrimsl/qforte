"""
Tools for building and diagonalizing operators in subspaces obtained by methods like SA-ADAPT-VQE.
"""

import numpy as np
import qforte as qf

def build_effective_symmetric_operator(qb_op, U, refs):
    """
    qb_op is a qubit operator (e.g. a Hamiltonian, S^2, dipole operator)
    U is a circuit, and refs are the Computer refs it acts on.
    A dense np array will be constructed in the space of those basis states.
    """

    dim = len(refs)
    eff_op = np.zeros((dim, dim), dtype="complex")
    vs = []
    for i in range(dim):
        sigma = qf.Computer(refs[i])
        sigma.apply_circuit(U)
        sigma.apply_operator(qb_op)
        sigma.apply_circuit(U.adjoint())
        vs.append(np.array(sigma.get_coeff_vec()))    
    for i in range(dim):
        for j in range(i, dim):
            eff_op[i,j] = eff_op[j,i] = np.array(refs[i].get_coeff_vec()).T.conj()@vs[j] 
    return eff_op 
