"""
Algorithms for constructing and optimizing optimal circuits for VQE or whatever.
"""

import qforte as qf

from qforte.utils.transforms import *
from qforte.utils.state_prep import *
from qforte.utils.trotterization import *

import numpy as np
from scipy.optimize import minimize
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
        self._Egs = None
        #ansatz is just a list of CNOTs
        self._ansatz = []
        #params of the single qubit gates
        self._params = []

    def best_circuit(self, n_cnots):
        x0 = np.zeros(6*n_cnots + 9)
        best_E = self._hf_energy
        best_ansatz = None
        pairs = []
        for ctrl in range(0, self._nqb):
            for targ in range(0, self._nqb):
                if ctrl != targ:
                    pairs.append([ctrl, targ])
        sequences = list(combinations_with_replacement(pairs, n_cnots))
        for i in range(0, len(sequences)):
            self._ansatz = sequences[i]
            print("CNOT sequence:")
            print(sequences[i])
            res = minimize(self.energy, x0, method = 'Nelder-Mead', options = {'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000000, 'disp': True})
            print(f"Energy: {res.fun}")
            if res.fun < best_E:
                best_E = res.fun
                best_ansatz = sequences[i]
            print(f"Best energy so far: {best_E}")
            print(f"Error:              {best_E - self._sys.fci_energy}")
        print(f"\nBest Energy: {best_E}")
        print(f"\nBest Sequence: {best_ansatz}")
        return best_E, best_ansatz, res.x    


    def energy(self, params):
        qc = qf.Computer(self._nqb)
        qc.apply_circuit(self._Uprep)
        for i in range(0, self._nqb):
            qc.apply_gate(qf.gate('Rz', i, params[3*i+0]))
            qc.apply_gate(qf.gate('Ry', i, params[3*i+1]))
            qc.apply_gate(qf.gate('Rx', i, params[3*i+2]))
        for i in range(0, len(self._ansatz)):
            qc.apply_gate(qf.gate('cX', self._ansatz[i][0], self._ansatz[i][1]))

            qc.apply_gate(qf.gate('Rz', self._ansatz[i][0], params[self._nqb + 6*i + 0]))
            qc.apply_gate(qf.gate('Ry', self._ansatz[i][0], params[self._nqb + 6*i + 1]))
            qc.apply_gate(qf.gate('Rx', self._ansatz[i][0], params[self._nqb + 6*i + 2]))
            qc.apply_gate(qf.gate('Rz', self._ansatz[i][1], params[self._nqb + 6*i + 3]))
            qc.apply_gate(qf.gate('Ry', self._ansatz[i][1], params[self._nqb + 6*i + 4]))
            qc.apply_gate(qf.gate('Rx', self._ansatz[i][1], params[self._nqb + 6*i + 5]))
        state = np.array(qc.get_coeff_vec(), dtype = "complex_").real
        qc.apply_operator(self._qb_ham)
        hstate = np.array(qc.get_coeff_vec(), dtype = "complex_").real
        E = state@hstate
        return E




            

            
            
