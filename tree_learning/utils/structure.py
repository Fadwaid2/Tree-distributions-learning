import numpy as np
import random

from numba import jit

#########################################
# using numba jit for faster running time 
#########################################
@jit(nopython=True)
def insert_good_edge(u, v):
    """Ensure edge is always stored as (min, max)."""
    return (u, v) if u < v else (v, u)


@jit(nopython=True)
def initialize_cm(w):
    """Initial contraction mapping: each edge maps to itself."""
    n = w.shape[0]
    cm = {}
    for u in range(n):
        for v in range(u + 1, n):
            cm[(u, v)] = (u, v)
    return cm

@jit(nopython=True)
def indegree_matrix(w):
    n = w.shape[0]
    D_in = np.zeros((n, n))
    # create a list of neighbors and compute vertex weights
    weight_vertex = np.sum(w, axis=1)
    for i in range(n):
        D_in[i][i] = weight_vertex[i]
    return D_in

@jit(nopython=True)
def Lapl_1(w):
    D = indegree_matrix(w)
    L_1 = D - w
    return L_1

@jit(nopython=True)
def Lapl_1_r(w, r):
    l_1 = Lapl_1(w)
    return remove_r(l_1, r)

@jit(nopython=True)
def remove_r(arr, r):
    return arr[np.arange(arr.shape[0]) != r][:, np.arange(arr.shape[1]) != r]

@jit(nopython=True)
def delete(w, e):
    n = w.shape[0]
    i, j = e
    delete_e_w = np.copy(w)
    delete_e_w[i, j] = 0
    delete_e_w[j, i] = 0
    return delete_e_w

def safe_logdet(A, epsilon=1e-1):
    """Compute log|det(A)| safely with regularization.
       Note: -working in log abs det for more stability 
             -removed sign check-up, otherwise it crashes. 
    """
    sign, logdet = np.linalg.slogdet(A)
    if sign == 0 or not np.isfinite(logdet):
        A_reg = A + epsilon * np.eye(A.shape[0])
        sign, logdet = np.linalg.slogdet(A_reg)
    #if sign <= 0:
    #    raise ValueError("Matrix is not positive definite or graph disconnected.")
    return logdet


def get_contraction_dicts_undirected(w, e):
    """
        Support function to contraction function.     
    """
    n=w.shape[0]
    i, j = e

    sum_w_i = {k: 0 for k in range(n)}
    sum_w_j = {k: 0 for k in range(n)}

    for child in range(n):
        if child != j and w[i][child] != 0:
            sum_w_i[child] +=w[i][child]
        if child != i and w[j][child] != 0:
            sum_w_j[child] +=w[j][child]


    sum_w={k: sum_w_j.get(k, 0) + sum_w_i.get(k, 0) for k in set(sum_w_j) & set(sum_w_i)}

    return sum_w, sum_w_i, sum_w_j

def contraction_original_weight(w, e):
    n = w.shape[0]
    i, j = e #get parent and child of that edge / i is always 0  

    #get the children of the contracted edge vertices 
    sum_w, sum_w_i, sum_w_j = get_contraction_dicts_undirected(w, e)

    sum_w_no_i_j = sum_w.copy()
    for key in [i, j]:
        sum_w_no_i_j.pop(key)

    #get the ordering of the new weight matrix 
    index_contracted_v = j-1 # i is always 0 

    key_mapping = {old_key: new_key for new_key, old_key in enumerate(sorted(sum_w_no_i_j.keys()))}
    items = list(key_mapping.items())

    items.insert(index_contracted_v, (j, index_contracted_v))
    for i in range(index_contracted_v + 1, len(items)):
        items[i] = (items[i][0], items[i][1] + 1)

    old_to_new_indices = dict(items) # this now has the ordering of the vertices dict{old index : new index}

    sum_w_no_i_j_new_keys = {old_to_new_indices[old_key]: value for old_key, value in sum_w_no_i_j.items()}

    #deleting i and j columns and rows 
    contracted_w = np.copy(w)

    contracted_w = np.delete(contracted_w, i, 0)
    contracted_w = np.delete(contracted_w, i, 1)
    contracted_w = np.delete(contracted_w, j-1, 0)
    contracted_w = np.delete(contracted_w, j-1, 1)

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
            elif col == index_contracted_v :
                if row in sum_w_no_i_j_new_keys:
                    new_contracted_w[row][col] = sum_w_no_i_j_new_keys[row]

    return new_contracted_w


@jit(nopython=True)
def get_cm_orig_vertices(w_after, w_before, e, prev_cm, epsilon=1e-1):
    """
    Update contraction mapping after edge contraction.
    """
    i, j = e
    cm = {}

    for u in range(w_after.shape[0]):
        for v in range(u + 1, w_after.shape[1]):
            if u == j - 1:
                first, second = insert_good_edge(i, v + 1)
                w1 = w_before[first, second]
                first, second = insert_good_edge(j, v + 1)
                w2 = w_before[first, second]
                rand = np.random.uniform(0, 1)
                cm[(u, v)] = insert_good_edge(i, v + 1) if rand < w1 / (w1 + w2 + epsilon) else insert_good_edge(j, v + 1)
            elif v == j - 1:
                first, second = insert_good_edge(i, u + 1)
                w1 = w_before[first, second]
                first, second = insert_good_edge(j, u + 1)
                w2 = w_before[first, second]
                rand = np.random.uniform(0, 1)
                cm[(u, v)] = insert_good_edge(i, u + 1) if rand < w1 / (w1 + w2 + epsilon) else insert_good_edge(j, u + 1)
            else:
                cm[(u, v)] = insert_good_edge(u + 1, v + 1)

    # Replace with previous CM values if mapped
    for key, val in cm.items():
        for key_prev, val_prev in prev_cm.items():
            if val == key_prev:
                cm[key] = val_prev
    return cm


#############################
# Sampling a spanning tree 
#############################
@jit(nopython=True)
def sampler(w, r=0, epsilon=1e-1):
    """
    Implements Algorithm 8 from https://arxiv.org/abs/2405.07914: SamplingArborescence.
    Returns list of edges in the sampled arborescence.
    """
    w = w.astype(np.float64)
    n = w.shape[0]
    edges = np.zeros((n - 1, 2), dtype=np.int64)
    edge_count = 0

    CM_previous = initialize_cm(w)
    i = 1

    # main loop: while root has outgoing edges
    while np.any(w[r] != 0):
        if i >= n: #to avoid endless loops 
            i = 1
        
        e = (r, i)
        deleted_adj = delete(w, e)
        
        # Preventive: this bit stops the deletion of the last edge, so only contraction allowed; otherwise all edges are deleted in some cases
        remaining_edges = np.count_nonzero(w[r])
        if remaining_edges == 1:
            #print(f"Only one edge left from root {r} ({e}) — must contract and not delete")
            # Forcing contraction instead of deletion
            adj_before = w.copy()
            w = contraction_original_weight(w, e)
            CM_current = get_cm_orig_vertices(w, adj_before, e, CM_previous)

            edges[edge_count] = CM_previous[e]
            edge_count += 1
            CM_previous = CM_current

            i = 1
            continue   

        arr1 = Lapl_1_r(deleted_adj, r=r)
        arr2 = Lapl_1_r(w, r=r)
        
        log_p_e = safe_logdet(arr1, epsilon) - safe_logdet(arr2, epsilon)
        log_u = np.log(np.random.uniform(0, 1))

        if log_u <= log_p_e:
            # DELETE edge
            w = delete(w, e)
            i += 1
            if np.all(w[r] == 0):
                print("Root isolated — cannot continue.") #shouldn't happen with the preventive bit above 
                #raise RuntimeError("Root isolated — cannot continue.")
        else:
            # CONTRACT edge
            adj_before = w
            w = contraction_original_weight(w, e)
            CM_current = get_cm_orig_vertices(w, adj_before, e, CM_previous)
            edges[edge_count] = CM_previous[e]
            edge_count += 1
            CM_previous = CM_current
            i = 1
        if edge_count == n - 1:
            break

    return edges

################################################################
#### For RWM-Wilson method
################################################################
def wilson_algorithm(weight, r=0, seed=None):
    """
    Sample a random spanning tree using Wilson's algorithm (Loop erased random walks)
    on a complete weighted graph.
    """

    if seed is not None:
        np.random.seed(seed)

    weight = weight.astype(np.float64)
    n = weight.shape[0]

    in_tree = np.zeros(n, dtype=bool)
    in_tree[r] = True

    pos = np.full(n, -1, dtype=int)
    T_edges = []

    for start in range(n):
        if in_tree[start]:
            continue

        path = []
        v = start

        # record start immediately
        pos[v] = 0
        path.append(v)

        while not in_tree[v]:
            w = weight[v].copy()
            w[v] = 0.0
            s = w.sum()

            if s <= 0 or not np.isfinite(s):
                u = np.random.randint(n - 1)
                if u >= v:
                    u += 1
            else:
                u = np.random.choice(n, p=w / s)

            v = u

            if pos[v] != -1:
                # erase loop
                path = path[:pos[v] + 1]
            else:
                pos[v] = len(path)
                path.append(v)

        # add edges along the path (excluding final in-tree vertex)
        for u, nxt in zip(path[:-1], path[1:]):
            #T_edges.append((u, nxt))
            if u < nxt:
                T_edges.append((u, nxt))
            else:
                T_edges.append((nxt, u))
            in_tree[u] = True

        # mark last vertex
        in_tree[path[-1]] = True

        # reset positions
        for u in path:
            pos[u] = -1

    return T_edges

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
                
def kruskal_algo(w): 
    n = w.shape[0]
    list_edges = []
    edges = []
    adjacency_matrix = np.zeros((n,n))
    e = 0 #to index the edges in the result
    
    for i in range(n):
        for j in range(i+1,n):
            if w[i][j]!=0:
                edges.append((w[i][j] , i, j ))
    edges.sort() #sorting edges by non decreasing weight
    
    uf = UnionFind(n)
    
    for weight, i, j in edges:
        if uf.find(i) != uf.find(j):
            uf.union(i, j)
            list_edges.append((min(i,j), max(i,j)))
            adjacency_matrix[i][j]=1
            adjacency_matrix[j][i]=1
    
    return list_edges



