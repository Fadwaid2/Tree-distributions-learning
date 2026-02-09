from abc import ABC, abstractmethod


class TreeLearner(ABC):
    """Abstract base class for tree distributions learners."""
    def __init__(self, data, k):
        self.data = data 
        self.n = data.shape[1]
        self.T = data.shape[0]
        self.k = k
        self.current_time = 1 
    
    @abstractmethod 
    def precompute_conditional_distributions(self):
        pass 

    @abstractmethod 
    def learn_weights(self, precomputed):
        """Learn the parameters of the model from the given data."""
        pass

    @abstractmethod 
    def learn_structure(self, w, **kwargs):
        """Learn the structure of the model from the given data."""
        pass
      
    @abstractmethod
    def update_weight_matrix(self, w, structure, precomputed, **kwargs): 
        pass
