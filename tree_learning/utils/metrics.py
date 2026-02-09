import baycomp  
import numpy as np

from .utils import sanitize_edges

def bayesian_test(data1, data2, name1, name2):
    """
       Performs a Bayesian signed rank test between two methods results-Log likelihood (LL)- 

        Parameters: 
            data1, data2 (numpy array with LL values obtained over different datasets): average Log-likelihood of two methods.  
                    e.g data1 from OFDE and data2 from RWM 
            name1, name2 : names of methods to show in histogram 
        Returns 
            Histogram 
    """
    fig = baycomp.SignedRankTest.plot_histogram(data1, data2, names=(name1, name2))
    fig.savefig(f'histogram_{name1}_{name2}.png', bbox_inches='tight')
    return fig 

def log_likelihood(test_data, cpt, edge_list):
    """
    Computes the log-likelihood of test data given a graph.

    Parameters:
        test_data (pd.DataFrame): Test dataset
        cpt (dict): Conditional Probability Tables
        edge_list (list): List of edges (parent, child).

    Returns:
        float: The negative log-likelihood.
    """
    log_likelihood = 0
    T = len(test_data) 
    for t in range(T): 
        row = test_data.iloc[t]  
        for edge in edge_list:
            parent, child = edge[0],edge[1]
            parent_value = row[parent]  
            node_value = row[child]  
            prob = cpt[(parent, child)].at[node_value, parent_value]
            log_likelihood += np.log(max(prob, 1e-10))
    return log_likelihood/len(test_data)

def shd(l_true, l_pred):
    """
        Takes two lists : 
            l_true representing edges in true spanning tree 
            l_pred with edges from predicted spanning tree
        Returns : 
            Structural Hamming Distance (SHD) 
    """
    l_true_set = set(l_true)
    l_pred_set = set(sanitize_edges(l_pred))

    # missing edges from true graph 
    missing_edges = l_true_set - l_pred_set
    # extra edges (predicted but are not in true graph)
    extra_edges = l_pred_set - l_true_set

    shd = len(missing_edges) + len(extra_edges)
    return shd
