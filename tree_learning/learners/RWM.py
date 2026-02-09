import random
import numpy as np 

from tree_learning.utils.structure import sampler
from tree_learning.utils.utils import refresh_dpt

from .base import TreeLearner

class RWM(TreeLearner):

    def __init__(self, data, k, epsilon=0.9):
        super(RWM, self).__init__(data=data, k=k)
       
        self.eta = np.sqrt(8*(self.n-2)*np.log(self.n)/self.T)
        self.epsilon = epsilon
        self.epsilon_rwm = self.epsilon / 2
        self.tau = self.epsilon_rwm / (4 * self.k**2)
        self.gamma = self.epsilon_rwm / 4


    def compute_ell_range(self):
        max_ell = int(np.floor(np.log(1/(2*self.tau))/np.log(1+self.gamma)))
        return list(range(max_ell + 1))

    def generate_distributions(self):
        """
        Discretization from Definition 3.6 from https://arxiv.org/pdf/2405.07914#appendix.C
        """
        ell_range = self.compute_ell_range()
        distributions = []
        
        for j in range(self.k): # iterate over all possible alphabet size values
            P_sigma = np.zeros(self.k)
            P_sigma[j] = self.tau
            for i in range(self.k):
                if i != j:
                    #sample an l_i randomly from the range 
                    ell = random.choice(ell_range)
                    P_sigma_copy = P_sigma.copy()
                    P_sigma_copy[i] = self.tau * (1 + self.gamma) ** ell
                    # normalize to ensure sum(P_sigma) = 1
                    P_sigma_copy[j] = 1 - np.sum(P_sigma_copy)
                    # check if the distribution is valid 
                    if all(P_sigma_copy >= 0):
                        distributions.append(P_sigma_copy)
        return distributions

    def precompute_conditional_distributions(self):
        all_distributions = {}
        for parent in range(self.n):
            for child in range(parent+1, self.n):
                all_distributions[(parent, child)] = [
                    self.generate_distributions() for t in range(self.T)
                ]
        return {
            "discretized_probas": all_distributions
        }

    def select_distributions(self, parent, child, precomputed_distributions, t):
        # get precomputed distributions for the specific parent and child
        all_dist = precomputed_distributions[(parent, child)][t]
        # extract data values for the specific data instance
        value_child = int(self.data.iloc[t, child])
        value_parent = int(self.data.iloc[t, parent])
        # initialize distributions with small default probabilities
        distributions = [[1 for _ in range(self.k)] for _ in range(self.k)]
        distributions[value_child][value_parent] = all_dist[value_child][value_parent]
        return distributions

    def learn_weights(self, precomputed):

        precomputed_dpt = {}
        precomputed_distrs = precomputed['discretized_probas']

        # precompute DPT for all parent-child pairs
        for parent in range(self.n):
            for child in range(parent+1, self.n):
                if parent != child:  
                    dpt = np.ones((self.k, self.k, self.T + 1))  #shape (i, j, T+1)
                    for t in range(self.T):
                        # get conditional probabilities
                        cond_proba = self.select_distributions(parent, child, precomputed_distrs, t)
                        # update the DPT
                        for i in range(self.k):
                            for j in range(self.k):
                                dpt[i, j, t + 1] = dpt[i, j, t] * (cond_proba[i][j] ** self.eta)
                    # sum over DPT values for each time step
                    sum_dpt = dpt.sum(axis=(0, 1))
                    precomputed_dpt[(parent, child)] = sum_dpt
        return precomputed_dpt

    def learn_structure(self, w, **kwargs):
        #call sampler from utils here directly 
        r = kwargs.get("r", 0)
        return sampler(w, r=r)

    def update_weight_matrix(self, w, structure, precomputed_dpt, **kwargs):
        for edge in structure:
            i, j = edge[0], edge[1]
            weight = precomputed_dpt[(i, j)][1]
            w[i, j] = weight
            w[j, i] = weight
        #refresh dpt after update (call function from utils)
        precomputed_dpt = refresh_dpt(precomputed_dpt)
        return w

