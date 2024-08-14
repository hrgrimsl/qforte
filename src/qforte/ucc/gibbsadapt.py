"""
Classes for Gibbs State ADAPT-VQE
====================================
"""

import qforte as qf

from qforte.abc.uccvqeabc import UCCVQE

import numpy as np
import scipy
import copy

kb = 3.166811563455546e-06

class Gibbs_ADAPT(UCCVQE):
    def run(self,
            references = None,
            pool_type = "GSD",
            verbose = False,
            T = 0):
                
        self._pool_type = pool_type
        self._compact_excitations = True
        self.fill_pool() 
        
        self._references = references
        self._weights = [1]*len(references)
        self._is_multi_state = True
        self.T = T
        
        if T > 0:
            self.beta = 1/(kb*T)
        else:
            self.beta = None

        self.history = []
        
        self.verbose = verbose
        self._Uprep = references
        #self._Upreps = [qf.build_Uprep(ref, 'occupation_list') for ref in references]
        self.update_C()
        print(self.subspace_C)

    def update_C(self):
        U = self.build_Uvqc()
        if self._state_prep_type == "computer":
            print(U)
            exit()

        
        
    def update_p(self):
        pass

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
                   


        



        
