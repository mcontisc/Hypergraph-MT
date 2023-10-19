"""
    Code to perform k-fold cross-validation for the hyperedge prediction task.
"""

from argparse import ArgumentParser
import numpy as np
import yaml
import os
import csv
import time

import tools as tl
import cv_functions as cvfun

import sys
sys.path.append('HyMT')
import HyMT as hymt

algos = ['HyMT', 'GrMT', 'HyMTpairs', 'GrMTpairs', 'PaMT']


def main_cv():
    p = ArgumentParser()
    p.add_argument('-f', '--in_folder', type=str, default='../data/input/')  # path of the input
    p.add_argument('-d', '--dataset', type=str, default='workplace')  # dataset to analyse
    p.add_argument('-K', '--K', type=int, default=5)  # number of communities
    p.add_argument('-F', '--NFold', type=int, default=5)  # number of fold to perform cross-validation
    p.add_argument('-S', '--NComparisons', type=int, default=1e3)  # number of comparisons in the AUC computation
    p.add_argument('-m', '--out_mask', type=bool, default=False)  # flag to output the masks
    p.add_argument('-s', '--out_results', type=bool, default=True)  # flag to output the results in a csv file
    p.add_argument('-o', '--out_folder_cv', type=str, default='../data/output/5-fold_cv/')  # path to store cv outputs
    p.add_argument('-v', '--verbose', type=int, default=1)  # flag to print details
    p.add_argument('-b', '--baselines', type=int, default=1)  # flag to run the baselines
    p.add_argument('-D', '--maxD', type=int, default=None)  # threshold for the highest degree (size hyper-edge)
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

    '''
    Auxiliary variables to extract test/train split
    '''
    AD = A[hyL2]
    BD = B[:, hyL2]
    hyeD = hye[hyL2]

    E = AD.shape[0]
    N = BD.shape[0]

    '''
    Cross validation parameters
    '''
    NFold = args.NFold
    n_comparisons = int(args.NComparisons)
    out_mask = args.out_mask
    out_results = args.out_results
    out_folder = args.out_folder_cv
    if out_results:
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

    '''
    Model parameters
    '''
    K = args.K

    # setting to run the algorithm
    with open(f'setting_{dataset}.yaml') as f:
        conf_inf = yaml.load(f, Loader=yaml.FullLoader)
    constraintU = bool(conf_inf['constraintU'])

    if conf_inf['out_inference']:
        # folder to store the inferred parameters
        if not os.path.exists(conf_inf['out_folder']):
            os.makedirs(conf_inf['out_folder'])

        # output file name
        if args.maxD:
            init_end_file = conf_inf['end_file'] + f'_D{args.maxD}' + f'_cU{constraintU}'
        else:
            init_end_file = conf_inf['end_file'] + f'_cU{constraintU}'

    prng = np.random.RandomState(seed=conf_inf['seed'])  # set seed random number generator
    rseed = prng.randint(1000)

    # save the results
    if out_results:
        cols = ['K', 'fold', 'seed0', 'rseed']
        for l in ['auc_train_', 'auc_test_']:
            for a in algos:
                cols.append(l + a)

        if args.maxD:
            out_file = out_folder + dataset + f'_{args.maxD}' + f'_cU{constraintU}' + '.csv'
        else:
            out_file = out_folder + dataset + f'_cU{constraintU}' + '.csv'

        if not os.path.isfile(out_file):  # write header
            with open(out_file, 'w') as outfile:
                wrtr = csv.writer(outfile, delimiter=',', quotechar='"')
                wrtr.writerow(cols)
        outfile = open(out_file, 'a')
        wrtr = csv.writer(outfile, delimiter=',', quotechar='"')
        if verbose:
            print(f'Results will be saved in: {out_file}')

    results = {'K': K, 'seed0': conf_inf['seed']}

    '''
    Cycle over folds
    '''
    time_start = time.time()

    indices = cvfun.shuffle_indices(E, rng=prng)

    for fold in range(NFold):

        results['fold'], results['rseed'] = fold, rseed

        rseed += prng.randint(500)

        if verbose:
            print('\nFOLD ', fold)

        """ Set up training and test datasets """
        mask_train = cvfun.extract_mask_kfold(indices, fold=fold, NFold=NFold, out_mask=out_mask, dataset=dataset)
        mask_test = np.logical_not(mask_train)
        mask_pairs = np.array([0 if len(e) != 2 else 1 for e in hyeD])
        mask_pairs_train = np.logical_and(mask_pairs, mask_train)
        mask_pairs_test = np.logical_and(mask_pairs, mask_test)

        """ Run Hypergraph-MT """
        if verbose:
            print('### Run Hypergraph-MT ###')
        if conf_inf['out_inference']:
            conf_inf['end_file'] = init_end_file + '_HyMT_f' + str(fold)
        model = hymt.model.HyMT()
        u, w, maxL = model.fit(AD[mask_train], hyeD[mask_train], BD[:, mask_train], K=K, **conf_inf)

        if bool(args.baselines):
            """ Baseline1: Run the model on the graph obtained by clique expansions (Graph-MT) """
            if verbose:
                print('\n### Run Graph-MT ###')
            if conf_inf['out_inference']:
                conf_inf['end_file'] = init_end_file + '_GrMT_f' + str(fold)
            A2, hye2, B2 = tl.extract_input_pairwise(AD[mask_train], hyeD[mask_train], N)
            model2 = hymt.model.HyMT()
            u2, w2, maxL2 = model2.fit(A2, hye2, B2, K=K, **conf_inf)

            if sum(mask_pairs_train) > 0:
                """ Baseline2: Run the model on the graph given by the subset of pairwise interactions (Pairs-MT) """
                if verbose:
                    print('- - - Run baseline only pairs - - -')
                if conf_inf['out_inference']:
                    conf_inf['end_file'] = init_end_file + '_PaMT_f' + str(fold)
                model3 = hymt.model.HyMT()
                u3, w3, maxL3 = model3.fit(AD[mask_pairs_train], hyeD[mask_pairs_train], BD[:, mask_pairs_train], K=K,
                                           **conf_inf)

        """ Output performance results """
        # all hyperedges
        results['auc_train_HyMT'] = cvfun.calculate_AUC(hyeD, u, w, N, mask=mask_train, pairwise=False,
                                                        n_comparisons=n_comparisons, rseed=rseed)
        results['auc_test_HyMT'] = cvfun.calculate_AUC(hyeD, u, w, N, mask=mask_test, pairwise=False,
                                                       n_comparisons=n_comparisons, rseed=rseed)

        if bool(args.baselines):
            results['auc_train_GrMT'] = cvfun.calculate_AUC(hyeD, u2, w2, N, mask=mask_train, pairwise=True,
                                                            n_comparisons=n_comparisons, rseed=rseed)
            results['auc_test_GrMT'] = cvfun.calculate_AUC(hyeD, u2, w2, N, mask=mask_test, pairwise=True,
                                                           n_comparisons=n_comparisons, rseed=rseed)
        else:
            results['auc_train_GrMTpairs'] = results['auc_test_GrMTpairs'] = None
            results['auc_train_PaMT'] = results['auc_test_PaMT'] = None

        # only pairwise
        if sum(mask_pairs_train) > 0:
            results['auc_train_HyMTpairs'] = cvfun.calculate_AUC(hyeD, u, w, N, mask=mask_pairs_train, pairwise=False,
                                                                 n_comparisons=n_comparisons, rseed=rseed)
        else:
            results['auc_train_HyMTpairs'] = None
        if sum(mask_pairs_test) > 0:
            results['auc_test_HyMTpairs'] = cvfun.calculate_AUC(hyeD, u, w, N, mask=mask_pairs_test, pairwise=False,
                                                                n_comparisons=n_comparisons, rseed=rseed)
        else:
            results['auc_test_HyMTpairs'] = None
        if bool(args.baselines):
            if sum(mask_pairs_train) > 0:
                results['auc_train_GrMTpairs'] = cvfun.calculate_AUC(hyeD, u2, w2, N, mask=mask_pairs_train,
                                                                     pairwise=True, n_comparisons=n_comparisons,
                                                                     rseed=rseed)
                results['auc_train_PaMT'] = cvfun.calculate_AUC(hyeD, u3, w3, N, mask=mask_pairs_train, pairwise=True,
                                                                n_comparisons=n_comparisons, rseed=rseed)
            else:
                results['auc_train_GrMTpairs'] = results['auc_train_PaMT'] = None
            if sum(mask_pairs_test) > 0:
                results['auc_test_GrMTpairs'] = cvfun.calculate_AUC(hyeD, u2, w2, N, mask=mask_pairs_test,
                                                                    pairwise=True, n_comparisons=n_comparisons,
                                                                    rseed=rseed)
            else:
                results['auc_test_GrMTpairs'] = None
            if np.logical_and(sum(mask_pairs_train) > 0, sum(mask_pairs_test) > 0):
                results['auc_test_PaMT'] = cvfun.calculate_AUC(hyeD, u3, w3, N, mask=mask_pairs_test, pairwise=True,
                                                               n_comparisons=n_comparisons, rseed=rseed)
            else:
                results['auc_test_PaMT'] = None

        if out_results:
            wrtr.writerow([results[c] for c in cols])
            outfile.flush()

    if verbose:
        print(f'\nTime elapsed: {np.round(time.time() - time_start, 2)} seconds.')


if __name__ == '__main__':
    main_cv()
