import json
import ast
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from pathlib import Path

from tree_learning.learners.Chow_Liu import Chow_Liu
from tree_learning.learners.RWM import RWM
from tree_learning.learners.OFDE import OFDE
from tree_learning.utils.data import generate_synthetic_data, conditional_distributions_set
from tree_learning.utils.metrics import *

def main(args): 
    # Either generate synthetic data or directly load available dataset  
    if args.synthetic:
        print(f"Generating synthetic data with {args.n} nodes, {args.T} samples and alphabet size {args.k}")
        train_data, test_data, graph = generate_synthetic_data(args.n, args.T, args.k, args.seed, args.tree, args.noise)
    else:
        print(f"Reading train data from {args.train_data}")
        train_data = pd.read_csv(args.train_data, index_col=0)
        print(f"Reading test data from {args.test_data}")
        test_data = pd.read_csv(args.test_data, index_col=0)
        print(f'Reading true graph list of edges from {args.true_graph}')
        with open(args.true_graph, "r") as file:
            graph = ast.literal_eval(file.read().strip())  # Convert string to list of tuples 
        

    # Initialize the learning algorithm 
    if args.method == 'RWM':
        learner = RWM(data=train_data, k=args.k, epsilon=args.epsilon)
    elif args.method == 'OFDE':
        learner = OFDE(data=train_data, k=args.k)
    elif args.method == 'Chow-Liu':
        # Chow-Liu is an offline method so get results direclty here 
        learner = Chow_Liu(data=train_data, k=args.k)
        cl_weight, cl_structure = learner.run_chow_liu() 
        results = {
            'log-likelihood': log_likelihood(test_data, conditional_distributions_set(train_data, args.k), cl_structure),
            'shd': shd(graph, cl_structure)
        }
    else:
        raise ValueError("Unsupported method. Choose from: RWM, OFDE, Chow-Liu.")


    if args.method in ['RWM', 'OFDE']:
        # Initialize weight matrix 
        w = np.ones((args.n, args.n), dtype=np.float64)
        np.fill_diagonal(w, 0)

        # Precompute conditional distributions 
        precomputed = learner.precompute_conditional_distributions()
        # Compute weights 
        precomputed_weights = learner.learn_weights(precomputed) 

        # Online Learning loop over T samples  
        for t in range(1, args.T+1):
            # track current time step in OL algorithms 
            learner.current_time = t 
            structure = learner.learn_structure(w)
            w = learner.update_weight_matrix(w, structure, precomputed_weights)


        # evaluate 
        results = {'log-likelihood': log_likelihood(test_data, conditional_distributions_set(train_data, args.k), structure),
                   'shd': shd(graph, structure)
           }

    # uncomment to call comparison between two different methods data1 and data2 are two log-likelihoods arrays from methods 1 and 2 across different datasets  
    #comparison_2methods_results = {'bayesian-test': bayesian_test(data1, data2, name1, name2)}


    args.output_folder.mkdir(exist_ok=True)
    with open(args.output_folder / 'arguments.json', 'w') as f:
        json.dump(vars(args), f, default=str)
    with open(args.output_folder / 'results.json', 'w') as f:
        json.dump(results, f, default=list)


if __name__ == '__main__':
     
    parser = ArgumentParser(description='Learning Tree-structured distributions')

    parser.add_argument('--train_data', type=Path, help='Path to the training dataset (CSV)')
    parser.add_argument('--test_data', type=Path, help='Path to the testing dataset (CSV)')
    parser.add_argument('--true_graph', type=Path, help='Path to the true graph list of edges (.txt file)')
    parser.add_argument('--synthetic', type=bool, help='Flag to generate synthetic data instead of loading')
    parser.add_argument('--output_folder', type=Path, required=True, help='Directory to save results and arguments')
    parser.add_argument('--n', type=int, required=True, help='Number of nodes (variables) in the distribution')
    parser.add_argument('--T', type=int, default=10, help='Number of time steps for online learning')
    parser.add_argument('--k', type=int, default=2, help='Alphabet size (values taken by the variables)')
    parser.add_argument('--tree', type=bool, help='Flag to generate synthetic data from tree')
    parser.add_argument('--seed', type=int, default=22, help='Random seed for synthetic data')
    parser.add_argument('--noise', type=float, default=0.5, help='Noise for synthetic data')

    # Algorithms 
    algorithm = parser.add_argument_group('Method')
    algorithm.add_argument('--method', type=str, choices=['Chow-Liu', 'RWM', 'OFDE'], required=True, help='Algorithm to learn the tree distribution')
    algorithm.add_argument('--epsilon', type=float, default=0.9, help='Epsilon value for RWM algorithm')

    args = parser.parse_args()

    main(args)
