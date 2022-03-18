"""
    Functions used in the k-fold cross-validation procedure.
"""

import numpy as np
from itertools import combinations
import os
import pickle


def sample_zero_n(hye, oneAUCG, N, rseed):
    """
        Sample zero entries.

        INPUT
        -------
        hye : ndarray
              Array of length E, containing the sets of hyperedges (as tuples).
        oneAUCG : list
                  List containing samples of the existing hyperedges.
        N : int
            Number of nodes.
        rseed : int
                Random seed.

        OUTPUT
        -------
        sample_zero : list
                      List containing non-existing hyperedges.
    """

    rng = np.random.RandomState(rseed)
    sample_zero = np.zeros_like(oneAUCG)
    nonzero_or_sampled = set(hye)
    for eid, e in enumerate(oneAUCG):
        d = len(e)  # of the same degree so we have balanced sets
        found_nnz = False
        while not found_nnz:
            t = tuple([rng.randint(0, N) for _ in range(d)])
            if t not in nonzero_or_sampled:
                nonzero_or_sampled.add(t)
                found_nnz = True
                sample_zero[eid] = t

    return sample_zero


def prob_greater_zero_Poisson(lmbd):
    """
        P(A>0) for a Poisson with mean lmbd.
    """
    return 1. - np.exp(-lmbd)


def calculate_AUC(hye, u, w, N, mask=None, n_comparisons=1000, rseed=10, proba=False, pairwise=False):
    """
        Return the AUC of the hyperedge prediction. It represents the probability that a randomly chosen missing
        connection (true positive) is given a higher score by our method than a randomly chosen set of
        unconnected vertices (true negative).

        INPUT
        -------
        hye : ndarray
              Array containing the sets of hyperedges (as tuples).
        u : ndarray
            Membership matrix.
        w : ndarray
            Affinity matrix.
        N : int
            Number of nodes.
        mask : ndarray
               Mask for selecting the held out set.
        n_comparisons : int
                        Number of comparisons to consider in the computation of AUC.
        rseed : int
                Random seed.
        proba : bool
                Flag to call prob_greater_zero_Poisson function.
        pairwise : bool
                   Flag to call the pairwise computations.

        OUTPUT
        -------
        auc : float
              AUC value.
    """

    rng = np.random.RandomState(rseed)
    if mask is None:
        oneAUCG = rng.choice(hye, n_comparisons, replace=True)
    else:
        oneAUCG = rng.choice(hye[mask], n_comparisons, replace=True)
    zeroAUCG = sample_zero_n(hye, oneAUCG, N, rseed)

    if pairwise:  # network model with only pairwise interactions
        proba = True
        assert w.shape[0] == 1

    R1 = []
    for el in oneAUCG:
        if not pairwise:
            # to check whether in the test there is an hyperedge whose D is greater than the max D found in the train
            if (len(el) - 2) >= (w.shape[0]):
                M = 0
            else:
                M = (np.prod(u[np.array(el)], axis=0) * w[len(el) - 2]).sum()
            if not proba:
                R1.append(M)
            else:
                R1.append(prob_greater_zero_Poisson(M))
        else:
            pairs = combinations(el, 2)
            M = np.array([(np.prod(u[np.array(e)], axis=0) * w[0]).sum() for e in pairs])
            R1.append(np.prod(prob_greater_zero_Poisson(M)))

    R1 = np.array(R1)

    R0 = []
    for el in zeroAUCG:
        assert el not in set(hye)
        if not pairwise:
            if (len(el) - 2) >= (w.shape[0]):
                M = 0
            else:
                M = (np.prod(u[np.array(el)], axis=0) * w[len(el) - 2]).sum()
            if not proba:
                R0.append(M)
            else:
                R0.append(prob_greater_zero_Poisson(M))
        else:
            pairs = combinations(el, 2)
            M = np.array([(np.prod(u[np.array(e)], axis=0) * w[0]).sum() for e in pairs])
            R0.append(np.prod(prob_greater_zero_Poisson(M)))

    R0 = np.array(R0)

    assert R0.shape[0] == n_comparisons
    assert R1.shape[0] == n_comparisons

    n1 = (R1 > R0).sum()
    n_tie = (R1 == R0).sum()
    auc = (n1 + 0.5 * n_tie) / float(n_comparisons)

    return auc


def shuffle_indices(n_samples, rng):
    """
        Shuffle the indices of the hyperedges.

        INPUT
        -------
        n_samples : int
                    Number of hyperedges.
        rng : RandomState
              Container for the Mersenne Twister pseudo-random number generator.

        OUTPUT
        -------
        indices : ndarray
                  Shuffled array with the indices of the hyperedges.
    """

    indices = np.arange(n_samples)
    rng.shuffle(indices)

    return indices


def extract_mask_kfold(indices, fold=0, NFold=5, out_mask=False, dataset=''):
    """
        Return the mask for selecting the held out set.

        INPUT
        -------
        indices : ndarray
                  Shuffled array with the indices of the hyperedges.
        fold : int
               Current fold.
        NFold : int
                Number of k-fold.
        out_mask : bool
                   Flag to save the masks.
        dataset : str
                  Dataset name.

        OUTPUT
        -------
        mask : ndarray
               Mask for selecting the held out set.
    """

    n_samples = indices.shape[0]
    mask = np.ones(n_samples, dtype=bool)
    test = indices[fold * (n_samples // NFold):(fold + 1) * (n_samples // NFold)]
    mask[test] = 0

    if out_mask:
        out_folder_mask = '../data/input/CV_masks/'
        if not os.path.exists(out_folder_mask):
            os.makedirs(out_folder_mask)
        outmask = out_folder_mask + f'mask_{dataset}_f{fold}.pkl'
        print('Mask saved in ', outmask)
        with open(outmask, 'wb') as f:
            pickle.dump(np.where(mask > 0), f)

    return mask
