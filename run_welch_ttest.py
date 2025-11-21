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


# VALID_METHODS = no methods


def load_dataset(data_file):
    data = (
    pd.read_csv(data_file, index_col=0)
    .set_index('Name')
    .select_dtypes(include=['number']) # Select only proteomics data
    .T #transform
)
    if data.ndim != 2:
        raise ValueError("Invalid data structure: data.matrix must be 2D")
    return data

def get_labels(data):
    # Make sure there is something in the index
    if len(data.index) == 0:
        raise ValueError("Data index is empty.")

    labels = [0 if ".N" in name else 1 for name in data.index]

    # Warn if there are only one class (case or control) in the data.
    if len(set(labels)) < 2:
        warnings.warn("Labels contain only one class.", UserWarning)
    
    print(f'Found {labels.count(0)} controls and {labels.count(1)} cases in the data.')

    return labels

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