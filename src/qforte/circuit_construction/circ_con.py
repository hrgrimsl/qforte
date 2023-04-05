"""
Algorithms for constructing and optimizing optimal circuits for VQE or whatever.
"""

import qforte as qf
import copy
from qforte.utils.transforms import *
from qforte.utils.state_prep import *
from qforte.utils.trotterization import *
import math
import numpy as np
from scipy.optimize import *
from itertools import *


class CircuitOpt():
    """
    This will represent our circuit as we grow it.
    """

    def __init__(self,
                 system,
                 **kwargs):

        self._sys = system
        self._ref = system.hf_reference
        self._Uprep = build_Uprep(self._ref, 'occupation_list')
        self._nqb = len(self._ref)
        self._qb_ham = system.hamiltonian
        self._hf_energy = system.hf_energy
        self._fci_energy = system.fci_energy
        self._Egs = None
        # ansatz is just a list of CNOTs
        self._ansatz = []
        # params of the single qubit gates
        self._params = []

    def best_circuit(self, n_cnots, guess = None):
        if guess is None:
            x0 = np.zeros(3*self._nqb + 6*n_cnots)
        else:
            x0 = guess


        best_E = self._hf_energy
        best_x = None
        best_ansatz = None
        pairs = []
        for ctrl in range(0, self._nqb):
            for targ in range(ctrl + 1, self._nqb):
                if ctrl != targ:
                    pairs.append([ctrl, targ])
        sequences = list(product(pairs, repeat=n_cnots))
        valid_sequences = []
        for i in range(0, len(sequences)):

            valid = True
            # Reject sequences which are the same up to qubit reordering:
            indices = []
            for pair in sequences[i]:
                indices.append(pair[0])
                indices.append(pair[1])
            idx = []
            [idx.append(x) for x in indices if x not in idx]

            if idx != [j for j in range(0, len(idx))]:
                valid = False
            # Reject sequences without the same first CNOT (same up to orbital rotation of reference.
            if sequences[i][0] != pairs[0]:
                valid = False
            for j in range(0, len(sequences[i]) - 1):
                numrep1 = sequences[i][j][0] * \
                    len(sequences[i]) + sequences[i][j][1]
                numrep2 = sequences[i][j+1][0] * \
                    len(sequences[i]) + sequences[i][j+1][1]
                # Delete redundant sequences due to commuting CNOTs
                if set(sequences[i][j]).intersection(set(sequences[i][j+1])) == set([]):
                    if numrep1 > numrep2:
                        valid = False
                if sequences[i][j][0] == sequences[i][j+1][0] or sequences[i][j][1] == sequences[i][j+1][1]:
                    if numrep1 > numrep2:
                        valid = False
                # Delete sequences with consecutive CNOTs on same qubits
                # (Not sure if this is a valid transpilation...)
                if set(sequences[i][j]) == set(sequences[i][j+1]):
                    valid = False

            if valid == True:
                valid_sequences.append(sequences[i])
        print(f"{len(valid_sequences)} valid sequences out of {len(sequences)}.")
        bounds = [(0, 2*math.pi) for i in x0]

        for i in range(0, len(valid_sequences)):
            self._ansatz = valid_sequences[i]
            print("CNOT sequence:")
            print(valid_sequences[i])
            #res = direct(self.energy, bounds, callback = self.cb, locally_biased = False, f_min_rtol = 1e-10)
            res = basinhopping(self.energy, x0, disp = True, T = 100, stepsize = 4*math.pi, niter = 20000)
            #res = dual_annealing(self.energy, bounds, x0 = x0, callback = self.cb)
            # res = minimize(self.energy, x0, method = 'Nelder-Mead', options = {'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000000, 'disp': True})
            print(f"Energy: {res.fun}")
            if res.fun < best_E:
                best_E = res.fun
                best_ansatz = valid_sequences[i]
                best_x = res.x
            if res.fun < self._fci_energy + 1e-9:
                return best_E, best_ansatz, best_x
            print(f"Best energy so far: {best_E}")
            print(f"Error:              {best_E - self._sys.fci_energy}")
        print(f"\nBest Energy: {best_E}")
        print(f"\nBest Sequence: {best_ansatz}")
        return best_E, best_ansatz, best_x

    #def cb(self, params, e, context, convergence = None):
    def cb(self, params, convergence = None):
        print(self.energy(params))

    def energy(self, params):
        qc = qf.Computer(self._nqb)
        qc.apply_circuit(self._Uprep)
        for i in range(0, self._nqb):
            qc.apply_gate(qf.gate('Rz', i, params[3*i+0]))
            qc.apply_gate(qf.gate('Rx', i, params[3*i+1]))
            qc.apply_gate(qf.gate('Rz', i, params[3*i+2]))


        for i in range(0, len(self._ansatz)):
            qc.apply_gate(
                qf.gate('cX', self._ansatz[i][0], self._ansatz[i][1]))
            qc.apply_gate(
                qf.gate('Rz', self._ansatz[i][0], params[3*self._nqb + 6*i + 0]))
            qc.apply_gate(
                qf.gate('Rx', self._ansatz[i][0], params[3*self._nqb + 6*i + 1]))
            qc.apply_gate(
                qf.gate('Rz', self._ansatz[i][0], params[3*self._nqb + 6*i + 2]))
            qc.apply_gate(
                qf.gate('Rz', self._ansatz[i][1], params[3*self._nqb + 6*i + 3]))
            qc.apply_gate(
                qf.gate('Rx', self._ansatz[i][1], params[3*self._nqb + 6*i + 4]))
            qc.apply_gate(
                qf.gate('Rz', self._ansatz[i][1], params[3*self._nqb + 6*i + 5]))


        state = np.array(qc.get_coeff_vec(), dtype="complex_").real
        qc.apply_operator(self._qb_ham)
        hstate = np.array(qc.get_coeff_vec(), dtype="complex_").real
        E = state@hstate

        return E
