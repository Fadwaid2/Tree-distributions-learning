import numpy as np
import random

from tree_learning.utils.structure import kruskal_algo, rounding, rounding_vectorized, kruskal_mst
from .base import TreeLearner


######################################################
###### Shared helpers
######################################################

def _precompute_cumulative_counts(data, m):
    """
    Precompute cumulative occurrence counts for single variables and pairs.

    count_1var[i][u, t] = number of times variable i took value u in x^1,...,x^t
    count_2var[(i,j)][u, v, t] = number of times (i,j) took (u,v) in x^1,...,x^t

    Index t=0 means zero observations (counts are all 0).
    """
    T = len(data)
    n = data.shape[1]

    count_1var = {}
    for i in range(n):
        counts = np.zeros((m, T + 1))
        for t in range(T):
            counts[:, t + 1] = counts[:, t]
            val = int(data.iloc[t, i])
            if 0 <= val < m:
                counts[val, t + 1] += 1
        count_1var[i] = counts

    count_2var = {}
    for i in range(n):
        for j in range(i + 1, n):
            counts = np.zeros((m, m, T + 1))
            for t in range(T):
                counts[:, :, t + 1] = counts[:, :, t]
                val_i = int(data.iloc[t, i])
                val_j = int(data.iloc[t, j])
                if 0 <= val_i < m and 0 <= val_j < m:
                    counts[val_i, val_j, t + 1] += 1
            count_2var[(i, j)] = counts

    return count_1var, count_2var


def _precompute_phi(data, count_1var, count_2var, m):
    """
    Precompute cumulative phi for all edges across all time steps.

    At time tau (1-indexed), phi^tau_ij uses theta^tau which has counts
    from the first tau-1 observations:
        theta_i^tau(u)    = (count_1var[i][u, tau-1] + 1/2) / ((tau-1) + m/2)
        theta_ij^tau(u,v) = (count_2var[(i,j)][u,v, tau-1] + 1/2) / ((tau-1) + m^2/2)

    phi = log(theta_i * theta_j / theta_ij)  (log-loss decomposition)

    Returns cumulative sums: precomputed_phi[(i,j)][t-1] = sum_{tau=1}^{t} phi^tau
    (matching Algorithm 1, Step 10).
    """
    T = len(data)
    n = data.shape[1]
    precomputed_phi = {}

    for i in range(n):
        for j in range(i + 1, n):
            cumulative = 0.0
            cum_list = []
            for t in range(T):
                val_i = int(data.iloc[t, i])
                val_j = int(data.iloc[t, j])

                cnt_i  = count_1var[i][val_i, t]
                cnt_j  = count_1var[j][val_j, t]
                cnt_ij = count_2var[(i, j)][val_i, val_j, t]

                theta_i  = (cnt_i  + 0.5) / (t + m / 2)
                theta_j  = (cnt_j  + 0.5) / (t + m / 2)
                theta_ij = (cnt_ij + 0.5) / (t + m ** 2 / 2)

                ratio = theta_i * theta_j / theta_ij
                phi = np.log(max(ratio, 1e-300))

                cumulative += phi
                cum_list.append(cumulative)

            precomputed_phi[(i, j)] = cum_list

    return precomputed_phi





######################################################
###### Version 1: OFDE — for synthetic data
######################################################

class OFDE(TreeLearner):
    """
    Online Forest Density Estimation for synthetic data.

    Precomputes cumulative counts and cumulative phi for all time steps,
    then runs FPL + Kruskal (ascending/argmin) + iterative swap rounding.
    """

    def __init__(self, data, k):
        super(OFDE, self).__init__(data=data, k=k)
        self.p = np.zeros(self.n * (self.n - 1) // 2)
        self.precomputed_phi = None
        self._w_fpl = None

    def precompute_conditional_distributions(self):
        count_1var, count_2var = _precompute_cumulative_counts(self.data, self.k)
        phi = _precompute_phi(self.data, count_1var, count_2var, self.k)
        self.precomputed_phi = phi
        return {"precomputed_phi": phi}

    def learn_weights(self, precomputed):
        return precomputed["precomputed_phi"]

    def learn_structure(self, w, **kwargs):
        t = self.current_time
        n = self.n

        # Follow Perturbed Leader: update all edges (not just tree edges)
        beta = (1.0 / n) * np.sqrt(2.0 / t)
        w_new = w.copy()
        for i in range(n):
            for j in range(i + 1, n):
                perturbation = random.uniform(0, 1.0 / beta)
                phi_cumulative = self.precomputed_phi[(i, j)][t - 1]
                w_new[i][j] = perturbation + phi_cumulative
                w_new[j][i] = w_new[i][j]
        self._w_fpl = w_new

        # Kruskal (ascending = argmin, correct for OFDE)
        structure = kruskal_algo(w_new)

        # Iterative swap rounding
        alpha = 1.0 / (4.0 * np.sqrt(2.0 * t))
        p_intermediate = self._generate_p_vector(structure, alpha)
        self.p = rounding(p_intermediate)

        return structure

    def update_weight_matrix(self, w, structure, precomputed_phi, **kwargs):
        return self._w_fpl

    def _generate_p_vector(self, f, alpha):
        """p^{t+1/2} = alpha * p^t + (1 - alpha) * f^{t+1/2}"""
        edge_set = set(f)
        f_array = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                f_array.append(1 if (i, j) in edge_set else 0)
        f_array = np.array(f_array)
        return alpha * self.p + (1 - alpha) * f_array


######################################################
###### Version 2: OFDEFast — vectorized for real data
######################################################



class OFDEFast(TreeLearner):
    """
    Vectorized Online Forest Density Estimation for large real-world datasets.

    Incremental phi computation — no O(n^2 * T) precomputation array.
    Memory: O(n*m + n_edges*m^2) vs O(n^2 * T) in OFDE.
    Much faster on large datasets.

    """

    def __init__(self, data, k):
        super(OFDEFast, self).__init__(data=data, k=k)
        n = self.n
        m = self.k
        n_edges = n * (n - 1) // 2

        triu_i, triu_j = np.triu_indices(n, k=1)
        self.triu_i = triu_i.astype(np.intp)
        self.triu_j = triu_j.astype(np.intp)
        self.arange_n = np.arange(n, dtype=np.intp)
        self.arange_edges = np.arange(n_edges, dtype=np.intp)

        self.running_cnt_1var = np.zeros((n, m), dtype=np.float64)
        self.running_cnt_2var = np.zeros((n_edges, m * m), dtype=np.float64)
        self.phi_cumulative = np.zeros(n_edges, dtype=np.float64)
        self.p = np.zeros(n_edges, dtype=np.float64)

        self._w_upper = None
        self._data_array = data.values.astype(np.int32)

    def precompute_conditional_distributions(self):
        # Incremental version needs no precomputation
        return {}

    def learn_weights(self, precomputed):
        return {}

    def learn_structure(self, w, **kwargs):
        t_idx = self.current_time - 1  # 0-indexed
        t = self.current_time          # 1-indexed
        m = self.k
        n = self.n
        triu_i = self.triu_i
        triu_j = self.triu_j

        # Current observation
        obs = self._data_array[t_idx]
        vals_i = obs[triu_i]
        vals_j = obs[triu_j]

        # Counts from first t_idx observations (before current obs)
        cnt_i  = self.running_cnt_1var[triu_i, vals_i]
        cnt_j  = self.running_cnt_1var[triu_j, vals_j]
        cell   = vals_i * m + vals_j
        cnt_ij = self.running_cnt_2var[self.arange_edges, cell]

        # Theta 
        denom_1var = t_idx + m / 2.0
        denom_2var = t_idx + m ** 2 / 2.0
        theta_i  = (cnt_i  + 0.5) / denom_1var
        theta_j  = (cnt_j  + 0.5) / denom_1var
        theta_ij = (cnt_ij + 0.5) / denom_2var

        # Phi = log(theta_i * theta_j / theta_ij) and accumulate
        ratio = theta_i * theta_j / theta_ij
        self.phi_cumulative += np.log(np.maximum(ratio, 1e-300))

        # Update running counts with current observation
        self.running_cnt_1var[self.arange_n, obs] += 1
        self.running_cnt_2var[self.arange_edges, cell] += 1

        # Follow Perturbed Leader (vectorized)
        beta = (1.0 / n) * np.sqrt(2.0 / t)
        perturbation = np.random.uniform(0, 1.0 / beta, size=len(triu_i))
        w_upper = perturbation + self.phi_cumulative
        self._w_upper = w_upper

        # Kruskal MST (ascending = argmin, correct for OFDE)
        f = kruskal_mst(w_upper, triu_i, triu_j, n)

        # Iterative swap rounding
        alpha = 1.0 / (4.0 * np.sqrt(2.0 * t))
        f_array = f[triu_i, triu_j]
        p_intermediate = alpha * self.p + (1.0 - alpha) * f_array
        self.p = rounding_vectorized(p_intermediate)

        # Return edge list (compatible with run.py: log_likelihood, shd)
        rows, cols = np.where(f > 0)
        structure = [(int(r), int(c)) for r, c in zip(rows, cols) if r < c]
        return structure

    def update_weight_matrix(self, w, structure, precomputed, **kwargs):
        n = self.n
        w_new = np.zeros((n, n), dtype=np.float64)
        if self._w_upper is not None:
            w_new[self.triu_i, self.triu_j] = self._w_upper
            w_new[self.triu_j, self.triu_i] = self._w_upper
        return w_new
