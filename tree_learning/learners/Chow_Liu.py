import numpy as np
import pandas as pd
import networkx as nx
import random

from sklearn.metrics import mutual_info_score

from tree_learning.utils.structure import kruskal_algo
from .base import TreeLearner

class Chow_Liu(TreeLearner):

    def __init__(self, data, k):
        super().__init__(data=data, k=k)
    
    def precompute_conditional_distributions(self):
        """
        Compute mutual information (mi) values: mi values for all the data steps 
        directly taken using sklearn package 
        """
        mi = {}
        for i in range(self.n):
            for j in range(i+1,self.n):
                labels_i = self.data.iloc[:,i].tolist()
                labels_j = self.data.iloc[:,j].tolist()
                mi[(i,j)] = mutual_info_score(labels_i, labels_j)
        return mi 

    def learn_weights(self, precomputed_mi):
        w = np.zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(i+1,self.n):
                w[i][j] = precomputed_mi[(i,j)]
                w[j][i] = w[i][j] 
        return w

    def learn_structure(self, w):
        structure = kruskal_algo(w)
        return structure 

    def update_weight_matrix(self, w, structure, precomputed, **kwargs):
        # Dummy/no-op implementation since Chow-Liu is an offline algorithm and doesnt need updates 
        return w

    def run_chow_liu(self):
        precomputed_mi = self.precompute_conditional_distributions()
        w = self.learn_weights(precomputed_mi)
        structure = self.learn_structure(w)
        return w, structure 

