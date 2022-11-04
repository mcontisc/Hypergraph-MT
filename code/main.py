""" Code to extract overlapping communities in hypergraphs. """

from argparse import ArgumentParser
import numpy as np
import yaml
import os
import time

import tools as tl

import sys
sys.path.append('HyMT')
import HyMT as hymt


def main():
    p = ArgumentParser()
    p.add_argument('-f', '--in_folder', type=str, default='../data/input/')  # path of the input
    p.add_argument('-d', '--dataset', type=str, default='workplace')  # dataset to analyse
    p.add_argument('-K', '--K', type=int, default=5)  # number of communities
    p.add_argument('-v', '--verbose', type=int, default=1)  # flag to print details
    p.add_argument('-b', '--baselines', type=int, default=1)  # flag to run the baselines
    p.add_argument('-D', '--maxD', type=int, default=None)  # threshold for the highest degree (size hyperedge) to keep
    p.add_argument('-U', '--constraintU', type=int, default=0)  # flag to normalize u such that every row sums to 1
    args = p.parse_args()

    verbose = bool(args.verbose)

    '''
    Import data
    '''
    dataset = args.dataset
    filename = args.in_folder + dataset + '.npz'
    data = np.load(filename, allow_pickle=True)
    A = data['A']  # array of length E, containing the weights of the hyperedges
    B = data['B']  # incidence matrix of dimension NxE
    hye = data['hyperedges']  # array of length E, containing the sets of hyperedges (as tuples)

    hyL = [len(e) for e in hye]
    if args.maxD:  # keep only hyperedges with 2 <= degree <= maxD
        hyL2 = [eid for eid, d in enumerate(hyL) if 2 <= d <= args.maxD]
    else:
        hyL2 = [eid for eid, _ in enumerate(hyL)]

    N = B.shape[0]  # number of nodes

    '''
    Model parameters
    '''
    K = args.K
    constraintU = bool(args.constraintU)

    # setting to run the algorithm
    with open(f'setting_{dataset}.yaml') as f:
        conf_inf = yaml.load(f, Loader=yaml.FullLoader)
    conf_inf['constraintU'] = constraintU

    if conf_inf['out_inference']:
        # folder to store the inferred parameters
        if not os.path.exists(conf_inf['out_folder']):
            os.makedirs(conf_inf['out_folder'])

        # output file name
        if args.maxD:
            init_end_file = conf_inf['end_file'] + f'_D{args.maxD}' + f'_cU{constraintU}'
        else:
            init_end_file = conf_inf['end_file'] + f'_cU{constraintU}'

    '''
    Run models
    '''

    """ Run Hypergraph-MT """
    time_HyMT = time.time()
    if verbose:
        print('### Run Hypergraph-MT ###')
    if conf_inf['out_inference']:
        conf_inf['end_file'] = init_end_file + '_HyMT'
    model = hymt.model.HyMT()
    _ = model.fit(A[hyL2], hye[hyL2], B[:, hyL2], K=K, **conf_inf)
    if verbose:
        print(f'\nTime elapsed: {np.round(time.time() - time_HyMT, 2)} seconds.')

    if bool(args.baselines):
        """ Baseline1: Run the model on the graph obtained by clique expansions (Graph-MT) """
        time_GrMT = time.time()
        if verbose:
            print('\n### Run Graph-MT ###')
        if conf_inf['out_inference']:
            conf_inf['end_file'] = init_end_file + '_GrMT'
        A2, hye2, B2 = tl.extract_input_pairwise(A[hyL2], hye[hyL2], N)  # get the graph by clique expansions
        model2 = hymt.model.HyMT()
        _ = model2.fit(A2, hye2, B2, K=K, **conf_inf)
        if verbose:
            print(f'\nTime elapsed: {np.round(time.time() - time_GrMT, 2)} seconds.')

        """ Baseline2: Run the model on the graph given by the subset of pairwise interactions (Pairs-MT) """
        time_PaMT = time.time()
        if verbose:
            print('\n### Run Pairs-MT ###')
        if conf_inf['out_inference']:
            conf_inf['end_file'] = init_end_file + '_PaMT'
        mask_pairs = np.array([False if len(e) != 2 else True for e in hye])  # keep only the subset of pairs
        if sum(mask_pairs) > 0:
            model3 = hymt.model.HyMT()
            _ = model3.fit(A[mask_pairs], hye[mask_pairs], B[:, mask_pairs], K=K, **conf_inf)
            if verbose:
                print(f'\nTime elapsed: {np.round(time.time() - time_PaMT, 2)} seconds.')


if __name__ == '__main__':
    main()
