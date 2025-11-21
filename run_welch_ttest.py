#!/usr/bin/env python

"""
Simple Welch t-test Omnibenchmark module.

Loads:
  --data.matrix         numeric matrix (samples × features)
  --data.true_labels    1D cluster/group labels (one per sample)

Outputs:
  A CSV-like .gz matrix:
      header: 
      rows:   
"""

import argparse
import os, sys
import numpy as np
from itertools import combinations
import pandas as pd
from scipy.stats import ttest_ind


# VALID_METHODS = ['birch', 'kmeans', 'spectral', 'gm']


def load_labels(data_file):
    data = np.loadtxt(data_file, ndmin=1)
    if data.ndim != 1:
        raise ValueError("Invalid data structure: true_labels must be 1D")
    return data.astype(int)


def load_dataset(data_file):
    data = np.loadtxt(data_file, ndmin=2)
    if data.ndim != 2:
        raise ValueError("Invalid data structure: data.matrix must be 2D")
    return data

def welch_ttest_df(X, labels):
    """
    X: 2D numpy array, shape (n_samples, n_features)
    labels: 1D array of 0/1 group labels, length = n_samples
    
    Returns:
        pandas DataFrame with t-statistics and p-values for each column
    """

    X = np.asarray(X)
    labels = np.asarray(labels)
    
    results = []

    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        
        # Split into groups
        group0 = col[labels == 0]
        group1 = col[labels == 1]
        
        # Run Welch's t-test (unequal variance)
        t_stat, p_val = ttest_ind(group0, group1, equal_var=False)
        
        results.append({'feature': f'col_{col_idx}', 't_stat': t_stat, 'p_value': p_val})

    df = pd.DataFrame(results)
    return df

# def run_welch(data, labels):


def main():
    parser = argparse.ArgumentParser(description='Welch t-test benchmark runner')

    parser.add_argument('--data.matrix', type=str,
                        help='gz-compressed textfile containing the comma-separated data to be clustered.', required = True)
    parser.add_argument('--data.true_labels', type=str,
                        help='gz-compressed textfile with the true labels; used to select a range of ks.', required = True) 
    parser.add_argument('--output_dir', type=str,
                        help='output directory to store data files.')
    # parser.add_argument('--name', type=str, help='name of the dataset', default='clustbench')
    # parser.add_argument('--method', type=str,
    #                     help='sklearn method',
    #                     required = True)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    labels = load_labels(getattr(args, 'data.true_labels'))
    X = load_dataset(getattr(args, 'data.matrix'))

    results = welch_ttest_df(X, labels)

    results.to_csv(os.path.join(args.output_dir, "results_welch_ttest.csv"))

if __name__ == "__main__":
    main()