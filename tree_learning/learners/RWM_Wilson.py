import numpy as np
import time

from numba import njit

from tree_learning.learners.RWM import (
    _precompute_distributions, _precompute_dpt, _refresh_dpt,
    _precompute_dpt_fast
)
from .base import TreeLearner

from tree_learning.utils.structure import _wilson_method, _wilson_method_fast

######################################################
###### Version 1: RWM_Wilson — for synthetic data
######################################################

class RWM_Wilson(TreeLearner):
    """
    Randomized Weighted Majority with Wilson's algorithm for synthetic data.

    Uses pure Python Wilson's algorithm + same Python-loop DPT as RWM.

    """

    def __init__(self, data, k, epsilon=0.9):
        super(RWM_Wilson, self).__init__(data=data, k=k)
        self.eta = np.sqrt(8 * (self.n - 2) * np.log(self.n) / self.T)
        self.epsilon = epsilon
        self.epsilon_rwm = epsilon / 2
        self.tau = self.epsilon_rwm / (4 * k ** 2)
        self.gamma = self.epsilon_rwm / 4
        self._precomputed_dpt = None

    def precompute_conditional_distributions(self):
        dists = _precompute_distributions(self.data, self.tau, self.gamma, self.k)
        return {"distributions": dists}

    def learn_weights(self, precomputed):
        dpt = _precompute_dpt(
            self.data, precomputed["distributions"], self.eta, self.k
        )
        self._precomputed_dpt = dpt
        return dpt

    def learn_structure(self, w, **kwargs):
        r = kwargs.get("r", 0)
        return _wilson_method(w, r=r)

    def update_weight_matrix(self, w, structure, precomputed_dpt, **kwargs):
        # Update ALL edges (not just tree edges)
        for parent in range(self.n):
            for child in range(parent + 1, self.n):
                arr = self._precomputed_dpt.get((parent, child))
                if arr is not None and arr.size > 1:
                    weight = arr[1]
                    w[parent, child] = weight
                    w[child, parent] = weight
        self._precomputed_dpt = _refresh_dpt(self._precomputed_dpt)
        return w


######################################################
###### Version 2: RWM_WilsonFast — vectorized for real data
######################################################

class RWM_WilsonFast(TreeLearner):
    """
    Vectorized Randomized Weighted Majority with Wilson's algorithm for large real-world datasets.

    Uses numba-compiled Wilson's algorithm with uniform fallback for stuck walks,
    vectorized DPT precomputation, and a pre-stacked DPT matrix for O(1) weight updates.

    Performance vs RWM_Wilson:
      - Numba-compiled Wilson sampler (~10x faster per step)
      - Vectorized DPT precomputation (no T*n^2 Python loops)
      - Pre-stacked (n, n, T+1) DPT matrix: weight update is a single matrix slice
    """

    def __init__(self, data, k, epsilon=0.9):
        super(RWM_WilsonFast, self).__init__(data=data, k=k)
        self.eta = np.sqrt(8 * (self.n - 2) * np.log(self.n) / self.T)
        self.epsilon = epsilon
        self.epsilon_rwm = epsilon / 2
        self.tau = self.epsilon_rwm / (4 * k ** 2)
        self.gamma = self.epsilon_rwm / 4
        self._dpt_matrix = None
        self._data_array = data.values.astype(np.int32)
        self._total_fallbacks = 0
        self._total_walks = 0

    def precompute_conditional_distributions(self):
        sum_dpt_all = _precompute_dpt_fast(
            self._data_array.astype(np.float64),
            self.tau, self.gamma, self.eta, self.k
        )
        # Pre-stack into (n, n, T+1) for O(1) vectorized weight updates
        n, T = self.n, self.T
        dpt_matrix = np.ones((n, n, T + 1), dtype=np.float64)
        for (p, c), arr in sum_dpt_all.items():
            length = min(len(arr), T + 1)
            dpt_matrix[p, c, :length] = arr[:length]
            dpt_matrix[c, p, :length] = arr[:length]
        dpt_matrix[np.arange(n), np.arange(n), :] = 0.0  # zero the diagonal at every t
        self._dpt_matrix = dpt_matrix
        return {"dpt_matrix": dpt_matrix}

    def learn_weights(self, precomputed):
        return precomputed["dpt_matrix"]

    def learn_structure(self, w, **kwargs):
        r = kwargs.get("r", 0)
        edges, fb_count, tw_count = _wilson_method_fast(w, r=r)
        self._total_fallbacks += fb_count
        self._total_walks += tw_count
        return edges

    def update_weight_matrix(self, w, structure, dpt_matrix, **kwargs):
        # Vectorized weight update: single matrix slice at current_time
        t = self.current_time  # 1-indexed, matches sum_dpt_all index
        w_new = self._dpt_matrix[:, :, t].copy()
        np.fill_diagonal(w_new, 0.0)
        return w_new
