import gc
import numpy as np
import random

from numba import jit

from .base import TreeLearner

from tree_learning.utils.structure import sampler

######################################################
###### Version 1 helpers: distribution generation
######################################################

def _compute_ell_range(tau, gamma):
    max_ell = int(np.floor(np.log(1 / (2 * tau)) / np.log(1 + gamma)))
    return list(range(max_ell + 1))


def _generate_distributions(tau, gamma, k, max_retries=100):
    """
    Discretization from Definition 3.6 (arXiv:2405.07914).
    Fix: generates exactly k distributions (one per j), with retry + uniform fallback.
    Original code generated one partial dist per (j, i) pair causing variable-length
    list and IndexError on access.
    """
    ell_range = _compute_ell_range(tau, gamma)
    distributions = []

    for j in range(k):
        found = False
        for _ in range(max_retries):
            P_sigma = np.zeros(k)
            for ii in range(k):
                if ii != j:
                    ell = random.choice(ell_range)
                    P_sigma[ii] = tau * (1 + gamma) ** ell
            P_sigma[j] = 1 - np.sum(P_sigma)
            if np.all(P_sigma >= 0):
                distributions.append(P_sigma)
                found = True
                break
        if not found:
            distributions.append(np.ones(k) / k)

    return distributions


def _precompute_distributions(data, tau, gamma, k):
    T = len(data)
    n = data.shape[1]
    all_distributions = {}
    for parent in range(n):
        for child in range(parent + 1, n):
            all_distributions[(parent, child)] = [
                _generate_distributions(tau, gamma, k) for _ in range(T)
            ]
    return all_distributions


def _select_distributions(data, parent, child, t, precomputed_distributions, k):
    all_dist = precomputed_distributions[(parent, child)][t]
    value_child = int(data.iloc[t, child])
    value_parent = int(data.iloc[t, parent])
    distributions = [[1 for _ in range(k)] for _ in range(k)]
    # Safety check: indices are valid (fix for original IndexError)
    if value_child < len(all_dist) and value_parent < len(all_dist[value_child]):
        distributions[value_child][value_parent] = all_dist[value_child][value_parent]
    return distributions


def _precompute_dpt(data, precomputed_distributions, eta, k):
    T = len(data)
    n = data.shape[1]
    precomputed_dpt = {}
    for parent in range(n):
        for child in range(parent + 1, n):
            dpt = np.ones((k, k, T + 1))
            for t in range(T):
                cond_proba = _select_distributions(data, parent, child, t, precomputed_distributions, k)
                for i in range(k):
                    for j in range(k):
                        dpt[i, j, t + 1] = dpt[i, j, t] * (cond_proba[i][j] ** eta)
            precomputed_dpt[(parent, child)] = dpt.sum(axis=(0, 1))
    return precomputed_dpt


def _refresh_dpt(old_dpt):
    new_dpt = {}
    for key, arr in old_dpt.items():
        new_dpt[key] = arr[1:] if arr.size > 1 else np.array([])
    del old_dpt
    gc.collect()
    return new_dpt


######################################################
###### Version 1: RWM — for synthetic data
######################################################

class RWM(TreeLearner):
    """
    Randomized Weighted Majority for synthetic data.

    Uses the arborescence sampler + Python-loop DPT precomputation.
    """

    def __init__(self, data, k, epsilon=0.9):
        super(RWM, self).__init__(data=data, k=k)
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
        return sampler(w, r=r)

    def update_weight_matrix(self, w, structure, precomputed_dpt, **kwargs):
        # Fix: update ALL edges (not just tree edges)
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
###### Version 2 helper: vectorized DPT
######################################################

def _precompute_dpt_fast(data_array, tau, gamma, eta, k):
    """
    Vectorized DPT precomputation in a single pass per edge.

    For each edge (parent, child):
      1. Batch-generate T distributions using vectorized numpy (no T Python loops)
      2. Extract prob for observed (child_val, parent_val) at each step
      3. Compute DPT in log-space using per-cell cumsum
      4. Returns index-addressable array — no refresh_dpt needed

    Returns:
        sum_dpt_all: dict {(parent, child): array of shape (T+1,)}
            At main loop step t (1-indexed), the weight is sum_dpt_all[(p,c)][t]
    """
    T, n = data_array.shape
    ell_range = np.arange(int(np.floor(np.log(1 / (2 * tau)) / np.log(1 + gamma))) + 1)
    n_ells = len(ell_range)
    if n_ells == 0:
        ell_range = np.array([0])
        n_ells = 1

    power_table = tau * (1 + gamma) ** ell_range

    sum_dpt_all = {}
    n_edges = n * (n - 1) // 2
    print(f"  Precomputing DPT for {n_edges} edges, T={T}, k={k}...")
    report_interval = max(1, n_edges // 10)
    edge_count = 0

    for parent in range(n):
        for child in range(parent + 1, n):
            vals_child = data_array[:, child].astype(np.intp)
            vals_parent = data_array[:, parent].astype(np.intp)

            # Batch generate distributions for all T steps
            ell_indices = np.random.randint(0, n_ells, size=(T, k, k - 1))
            powers = power_table[ell_indices]

            probs = np.zeros((T, k, k), dtype=np.float64)
            for j in range(k):
                col_idx = 0
                for ii in range(k):
                    if ii != j:
                        probs[:, j, ii] = powers[:, j, col_idx]
                        col_idx += 1
                probs[:, j, j] = 1.0 - np.sum(probs[:, j, :], axis=1)

            # Fix invalid distributions -> uniform fallback
            for j in range(k):
                bad = probs[:, j, j] < 0
                if np.any(bad):
                    probs[bad, j, :] = 1.0 / k

            # Extract probability for observed (child_val, parent_val)
            prob_obs = probs[np.arange(T), vals_child, vals_parent]

            # DPT in log-space: per-cell cumsum (no T*k^2 Python loop)
            log_factor = eta * np.log(np.maximum(prob_obs, 1e-300))
            observed_cell = vals_child * k + vals_parent

            log_dpt = np.zeros((k * k, T + 1), dtype=np.float64)
            for c in range(k * k):
                mask = (observed_cell == c)
                contrib = np.where(mask, log_factor, 0.0)
                log_dpt[c, 1:] = np.cumsum(contrib)

            sum_dpt_all[(parent, child)] = np.sum(np.exp(log_dpt), axis=0)

            edge_count += 1
            if edge_count % report_interval == 0:
                print(f"    {edge_count}/{n_edges} edges ({100*edge_count/n_edges:.0f}%)")

    print(f"  DPT precomputation done.")
    return sum_dpt_all


######################################################
###### Version 2: RWMFast — vectorized for real data
######################################################

class RWMFast(TreeLearner):
    """
    Vectorized Randomized Weighted Majority for large real-world datasets.

    Same arborescence sampler (corrected), but DPT precomputation is vectorized:
      - Batch distribution generation (no T*n^2 Python loops)
      - Log-space DPT via cumsum (no T*n^2*k^2 Python loops)
      - Index-based DPT access at step t — no refresh_dpt array copies
      - All C(n,2) edge weight updates
    """

    def __init__(self, data, k, epsilon=0.9):
        super(RWMFast, self).__init__(data=data, k=k)
        self.eta = np.sqrt(8 * (self.n - 2) * np.log(self.n) / self.T)
        self.epsilon = epsilon
        self.epsilon_rwm = epsilon / 2
        self.tau = self.epsilon_rwm / (4 * k ** 2)
        self.gamma = self.epsilon_rwm / 4
        self._triu_pairs = [(p, c) for p in range(self.n) for c in range(p + 1, self.n)]
        self._sum_dpt_all = None
        self._data_array = data.values.astype(np.int32)

    def precompute_conditional_distributions(self):
        sum_dpt_all = _precompute_dpt_fast(
            self._data_array.astype(np.float64),
            self.tau, self.gamma, self.eta, self.k
        )
        self._sum_dpt_all = sum_dpt_all
        return {"sum_dpt_all": sum_dpt_all}

    def learn_weights(self, precomputed):
        return precomputed["sum_dpt_all"]

    def learn_structure(self, w, **kwargs):
        r = kwargs.get("r", 0)
        return sampler(w, r=r)

    def update_weight_matrix(self, w, structure, sum_dpt_all, **kwargs):
        # Index-based access: at current_time t (1-indexed), read sum_dpt[t]
        t = self.current_time
        for parent, child in self._triu_pairs:
            dpt_arr = self._sum_dpt_all[(parent, child)]
            if t < len(dpt_arr):
                weight = dpt_arr[t]
                w[parent, child] = weight
                w[child, parent] = weight
        return w
