import numpy as np
import random

from numba import jit, njit

#########################################
# using numba jit for faster running time 
#########################################
@jit(nopython=True)
def _indegree_matrix(w):
    n = w.shape[0]
    D_in = np.zeros((n, n))
    weight_vertex = np.sum(w, axis=1)
    for i in range(n):
        D_in[i][i] = weight_vertex[i]
    return D_in


@jit(nopython=True)
def _lapl_1(w):
    return _indegree_matrix(w) - w


@jit(nopython=True)
def _remove_r(arr, r):
    return arr[np.arange(arr.shape[0]) != r][:, np.arange(arr.shape[1]) != r]


@jit(nopython=True)
def _lapl_1_r(w, r):
    return _remove_r(_lapl_1(w), r)


@jit(nopython=True)
def _delete(w, e):
    i, j = e
    w_copy = np.copy(w)
    w_copy[i, j] = 0
    w_copy[j, i] = 0
    return w_copy


@jit(nopython=True)
def _insert_good_edge(u, v):
    return (u, v) if u < v else (v, u)


@jit(nopython=True)
def _initialize_cm(w):
    n = w.shape[0]
    cm = {}
    for u in range(n):
        for v in range(u + 1, n):
            cm[(u, v)] = (u, v)
    return cm


@jit(nopython=True)
def _get_cm_orig_vertices(w_after, w_before, e, prev_cm, epsilon=1e-1):
    i, j = e
    cm = {}
    for u in range(w_after.shape[0]):
        for v in range(u + 1, w_after.shape[1]):
            if u == j - 1:
                first, second = _insert_good_edge(i, v + 1)
                w1 = w_before[first, second]
                first, second = _insert_good_edge(j, v + 1)
                w2 = w_before[first, second]
                rand = np.random.uniform(0, 1)
                cm[(u, v)] = _insert_good_edge(i, v + 1) if rand < w1 / (w1 + w2 + epsilon) else _insert_good_edge(j, v + 1)
            elif v == j - 1:
                first, second = _insert_good_edge(i, u + 1)
                w1 = w_before[first, second]
                first, second = _insert_good_edge(j, u + 1)
                w2 = w_before[first, second]
                rand = np.random.uniform(0, 1)
                cm[(u, v)] = _insert_good_edge(i, u + 1) if rand < w1 / (w1 + w2 + epsilon) else _insert_good_edge(j, u + 1)
            else:
                cm[(u, v)] = _insert_good_edge(u + 1, v + 1)
    for key, val in cm.items():
        for key_prev, val_prev in prev_cm.items():
            if val == key_prev:
                cm[key] = val_prev
    return cm


def _get_contraction_dicts_undirected(w, e):
    n = w.shape[0]
    i, j = e
    sum_w_i = {k: 0 for k in range(n)}
    sum_w_j = {k: 0 for k in range(n)}
    for child in range(n):
        if child != j and w[i][child] != 0:
            sum_w_i[child] += w[i][child]
        if child != i and w[j][child] != 0:
            sum_w_j[child] += w[j][child]
    sum_w = {k: sum_w_j.get(k, 0) + sum_w_i.get(k, 0) for k in set(sum_w_j) & set(sum_w_i)}
    return sum_w, sum_w_i, sum_w_j


def _contraction_original_weight(w, e):
    """
    Contract edge e in weight matrix w.
    Fix: renamed loop variable from i to idx to avoid shadowing the edge vertex i.
    """
    n = w.shape[0]
    i, j = e

    sum_w, sum_w_i, sum_w_j = _get_contraction_dicts_undirected(w, e)
    sum_w_no_i_j = sum_w.copy()
    for key in [i, j]:
        sum_w_no_i_j.pop(key)

    index_contracted_v = j - 1
    key_mapping = {old_key: new_key for new_key, old_key in enumerate(sorted(sum_w_no_i_j.keys()))}
    items = list(key_mapping.items())
    items.insert(index_contracted_v, (j, index_contracted_v))
    # Fix: renamed loop variable from i to idx
    for idx in range(index_contracted_v + 1, len(items)):
        items[idx] = (items[idx][0], items[idx][1] + 1)

    old_to_new_indices = dict(items)
    sum_w_no_i_j_new_keys = {old_to_new_indices[old_key]: value for old_key, value in sum_w_no_i_j.items()}

    contracted_w = np.copy(w)
    contracted_w = np.delete(contracted_w, i, 0)
    contracted_w = np.delete(contracted_w, i, 1)
    contracted_w = np.delete(contracted_w, j - 1, 0)
    contracted_w = np.delete(contracted_w, j - 1, 1)

    new_size = contracted_w.shape[0] + 1
    new_contracted_w = np.zeros((new_size, new_size))

    for col in range(new_size):
        col_in_w_0 = next(key for key, value in old_to_new_indices.items() if value == col)
        for row in range(new_size):
            row_in_w_0 = next(key for key, value in old_to_new_indices.items() if value == row)
            if row != index_contracted_v and col != index_contracted_v:
                new_contracted_w[row][col] = w[row_in_w_0][col_in_w_0]
            elif row == index_contracted_v:
                if col in sum_w_no_i_j_new_keys:
                    new_contracted_w[row][col] = sum_w_no_i_j_new_keys[col]
            elif col == index_contracted_v:
                if row in sum_w_no_i_j_new_keys:
                    new_contracted_w[row][col] = sum_w_no_i_j_new_keys[row]

    return new_contracted_w


def _safe_logdet(A, epsilon=1e-1):
    sign, logdet = np.linalg.slogdet(A)
    if sign == 0 or not np.isfinite(logdet):
        A_reg = A + epsilon * np.eye(A.shape[0])
        sign, logdet = np.linalg.slogdet(A_reg)
    return logdet


def sampler(w, r=0, epsilon=1e-1):
    """
    Algorithm 8 from arXiv:2405.07914: SamplingArborescence.
    Fix: uses w.shape[0] for bounds after contractions (not a fixed original n).
    """
    w = w.astype(np.float64)
    n_original = w.shape[0]
    edges = np.zeros((n_original - 1, 2), dtype=np.int64)
    edge_count = 0

    CM_previous = _initialize_cm(w)
    i = 1

    while np.any(w[r] != 0):
        # Fix: use current matrix size after contractions
        current_n = w.shape[0]
        if i >= current_n:
            i = 1

        e = (r, i)
        deleted_adj = _delete(w, e)

        remaining_edges = np.count_nonzero(w[r])
        if remaining_edges == 1:
            adj_before = w.copy()
            w = _contraction_original_weight(w, e)
            CM_current = _get_cm_orig_vertices(w, adj_before, e, CM_previous)
            edges[edge_count] = CM_previous[e]
            edge_count += 1
            CM_previous = CM_current
            i = 1
            continue

        arr1 = _lapl_1_r(deleted_adj, r=r)
        arr2 = _lapl_1_r(w, r=r)

        log_p_e = _safe_logdet(arr1, epsilon) - _safe_logdet(arr2, epsilon)
        log_u = np.log(np.random.uniform(0, 1))

        if log_u <= log_p_e:
            w = _delete(w, e)
            i += 1
            if np.all(w[r] == 0):
                print("Root isolated — cannot continue.")
        else:
            adj_before = w.copy()
            w = _contraction_original_weight(w, e)
            CM_current = _get_cm_orig_vertices(w, adj_before, e, CM_previous)
            edges[edge_count] = CM_previous[e]
            edge_count += 1
            CM_previous = CM_current
            i = 1

        if edge_count == n_original - 1:
            break

    return edges


################################################################
#### For RWM-Wilson method
################################################################
######################################################
###### Version 1: Wilson's algorithm — pure Python
######################################################

def _wilson_method(weight, r=0):
    """
    Sample a random spanning tree using Wilson's algorithm (loop-erased random walks).
    Pure Python / numpy implementation for synthetic data.
    """
    weight = weight.astype(np.float32)
    n = weight.shape[0]
    T_edges = set()
    in_tree = {r}

    def random_walk(v):
        path = [v]
        while v not in in_tree:
            neighbors = np.arange(n)[np.arange(n) != v]
            weights = weight[v, neighbors]
            total = np.sum(weights)
            if total <= 0 or np.isnan(total):
                probs = np.ones_like(weights) / len(weights)
            else:
                probs = weights / total
            v = int(np.random.choice(neighbors, p=probs))
            if v in path:
                idx = path.index(v)
                path = path[:idx + 1]
            else:
                path.append(v)
        return path

    for v in range(n):
        if v not in in_tree:
            path = random_walk(v)
            for u, nxt in zip(path[:-1], path[1:]):
                T_edges.add((min(u, nxt), max(u, nxt)))
                in_tree.add(u)
                in_tree.add(nxt)

    return list(T_edges)


######################################################
###### Version 2: Wilson's algorithm — numba + fallback
######################################################

@njit
def _wilson_sample(weight, r, max_steps_per_walk):
    """
    Numba-compiled Wilson's algorithm core.
    If a walk exceeds max_steps_per_walk, switches to uniform
    neighbor selection for the remainder of that walk.
    Returns (edges, fallback_count, total_walks).
    """
    n = weight.shape[0]
    in_tree = np.zeros(n, dtype=np.bool_)
    in_tree[r] = True

    pos = np.full(n, -1, dtype=np.int64)
    path = np.empty(n, dtype=np.int64)
    edges = np.empty((n - 1, 2), dtype=np.int64)
    edge_count = 0
    fallback_count = 0
    total_walks = 0

    cumw = np.empty(n, dtype=np.float64)

    for start in range(n):
        if in_tree[start]:
            continue

        path_len = 0
        v = start
        pos[v] = 0
        path[0] = v
        path_len = 1
        steps = 0
        hit_fallback = False
        total_walks += 1

        while not in_tree[v]:
            use_uniform = (steps >= max_steps_per_walk)
            if use_uniform and not hit_fallback:
                hit_fallback = True
                fallback_count += 1

            if use_uniform:
                u = np.random.randint(0, n - 1)
                if u >= v:
                    u += 1
            else:
                s = 0.0
                for j in range(n):
                    if j != v:
                        s += weight[v, j]

                if s <= 0.0:
                    u = np.random.randint(0, n - 1)
                    if u >= v:
                        u += 1
                else:
                    cumw[0] = weight[v, 0] if 0 != v else 0.0
                    for j in range(1, n):
                        cumw[j] = cumw[j - 1] + (weight[v, j] if j != v else 0.0)
                    r_val = np.random.random() * s
                    u = 0
                    for j in range(n):
                        if cumw[j] > r_val:
                            u = j
                            break
                    else:
                        u = n - 1

            v = u
            steps += 1

            if pos[v] != -1:
                cut = pos[v] + 1
                for idx in range(cut, path_len):
                    pos[path[idx]] = -1
                path_len = cut
            else:
                pos[v] = path_len
                path[path_len] = v
                path_len += 1

        for idx in range(path_len - 1):
            a = path[idx]
            b = path[idx + 1]
            if a < b:
                edges[edge_count, 0] = a
                edges[edge_count, 1] = b
            else:
                edges[edge_count, 0] = b
                edges[edge_count, 1] = a
            edge_count += 1
            in_tree[a] = True
        in_tree[path[path_len - 1]] = True

        for idx in range(path_len):
            pos[path[idx]] = -1

    return edges[:edge_count], fallback_count, total_walks


def _wilson_method_fast(weight, r=0, max_steps_factor=None):
    """
    Sample a random spanning tree using Wilson's algorithm.
    Thin wrapper around the numba-compiled core.
    Falls back to uniform sampling for walks that exceed max_steps_factor * n steps.
    Returns (edges_list, fallback_count, total_walks).
    """
    weight = np.ascontiguousarray(weight, dtype=np.float64)
    n = weight.shape[0]
    max_steps = int(n * max_steps_factor) if max_steps_factor is not None else 2 * n * n
    edges_arr, fallback_count, total_walks = _wilson_sample(weight, r, max_steps)
    edges_list = [(int(edges_arr[i, 0]), int(edges_arr[i, 1])) for i in range(edges_arr.shape[0])]
    return edges_list, int(fallback_count), int(total_walks)

# Maximum-weight spanning tree (MWST) algorithm : kruskal's algorithm 
## SOURCE : https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/
## Citation : This code is contributed by Neelam Yadav
## Citation : Improved by James Graça-Jones

# Class for union and find operations in kruskal's algorithm 
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    
    def union(self, u, v):
    # Merge Sets: If the roots are different, the subsets are merged. The subset with the smaller rank is attached to the root of the subset with the larger rank. If both ranks are equal, one root becomes the parent of the other, and its rank is incremented.
    
        root_u = self.find(u)
        root_v = self.find(v)
        
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1
                
def kruskal_algo(w, reverse=False):
    n = w.shape[0]
    list_edges = []
    edges = []
    adjacency_matrix = np.zeros((n,n))
    e = 0 #to index the edges in the result

    for i in range(n):
        for j in range(i+1,n):
            if w[i][j]!=0:
                edges.append((w[i][j] , i, j ))
    edges.sort(reverse=reverse)  # ascending for OFDE (argmin), descending for Chow-Liu (argmax)
    
    uf = UnionFind(n)
    
    for weight, i, j in edges:
        if uf.find(i) != uf.find(j):
            uf.union(i, j)
            list_edges.append((min(i,j), max(i,j)))
            adjacency_matrix[i][j]=1
            adjacency_matrix[j][i]=1
    
    return list_edges


class _UnionFindFast:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        while self.parent[u] != u:
            self.parent[u] = self.parent[self.parent[u]]
            u = self.parent[u]
        return u

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1
            return True
        return False


def kruskal_mst(w_upper, triu_i, triu_j, n):
    """Minimum spanning tree from upper-triangle edge weights (returns adjacency matrix)."""
    order = np.argsort(w_upper)
    adjacency = np.zeros((n, n), dtype=np.float64)
    uf = _UnionFindFast(n)
    count = 0
    for idx in order:
        i, j = int(triu_i[idx]), int(triu_j[idx])
        if uf.union(i, j):
            adjacency[i, j] = 1
            adjacency[j, i] = 1
            count += 1
            if count == n - 1:
                break
    return adjacency


def rounding_vectorized(x):
    """Vectorized iterative SWAP_1 rounding onto spanning-tree matroid polytope."""
    x = x.copy()
    n = len(x)
    max_iterations = n * 2
    for _ in range(max_iterations):
        fractional = np.where((x > 1e-10) & (x < 1 - 1e-10))[0]
        if len(fractional) < 2:
            break
        choice = np.random.choice(len(fractional), size=2, replace=False)
        i, j = fractional[choice[0]], fractional[choice[1]]
        if x[i] + x[j] <= 1:
            total = x[i] + x[j]
            if total < 1e-12:
                continue
            if np.random.rand() < x[i] / total:
                x[i], x[j] = total, 0.0
            else:
                x[i], x[j] = 0.0, total
        else:
            denom = 2 - x[i] - x[j]
            if denom < 1e-12:
                if np.random.rand() < 0.5:
                    x[i], x[j] = 1.0, x[i] + x[j] - 1
                else:
                    x[i], x[j] = x[i] + x[j] - 1, 1.0
            else:
                if np.random.rand() < (1 - x[j]) / denom:
                    x[i], x[j] = 1.0, x[i] + x[j] - 1
                else:
                    x[i], x[j] = x[i] + x[j] - 1, 1.0
    return x

def rounding(x):
    """
    Iterative SWAP_1 rounding to project onto spanning-tree matroid polytope.
    Iterates until all components are 0 or 1.
    """
    x = x.copy()
    n = len(x)
    max_iterations = n * 2
    for _ in range(max_iterations):
        fractional = [idx for idx in range(n) if 1e-10 < x[idx] < 1 - 1e-10]
        if len(fractional) < 2:
            break
        i, j = random.sample(fractional, 2)
        if x[i] + x[j] <= 1:
            total = x[i] + x[j]
            if total < 1e-12:
                continue
            if np.random.rand() < x[i] / total:
                x[i], x[j] = total, 0.0
            else:
                x[i], x[j] = 0.0, total
        else:
            denom = 2 - x[i] - x[j]
            if denom < 1e-12:
                if np.random.rand() < 0.5:
                    x[i], x[j] = 1.0, x[i] + x[j] - 1
                else:
                    x[i], x[j] = x[i] + x[j] - 1, 1.0
            else:
                if np.random.rand() < (1 - x[j]) / denom:
                    x[i], x[j] = 1.0, x[i] + x[j] - 1
                else:
                    x[i], x[j] = x[i] + x[j] - 1, 1.0
    return x
