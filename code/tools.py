"""
    Functions for handling the data.
"""

import numpy as np
import networkx as nx
from itertools import combinations


def HyG2Net(A, hye):
    """
        Return a networkx graph, i.e., a network with only pairwise interactions.

        INPUT
        -------
        A: ndarray
           Array of length E, containing the weights of the hyperedges.
        hye : ndarray
              Array of length E, containing the sets of hyperedges (as tuples).

        OUTPUT
        -------
        G : Graph
            NetworkX Graph.
    """

    G = nx.Graph()
    if isinstance(hye, tuple):  # individual edge
        size = len(hye)
        comb = combinations(hye, 2)
        for e in list(comb):
            if e in G.edges():
                G.edges[e]['weight'] += A
            else:
                G.add_edge(*e, weight=A, d=size)
    else:
        for eid, e in enumerate(hye):
            size = len(e)
            if size == 2:
                if e in G.edges():
                    G.edges[e]['weight'] += A[eid]
                else:
                    G.add_edge(*e, weight=A[eid], d=2)
            elif size > 2:
                comb = combinations(e, 2)
                for e1 in list(comb):
                    if e1 in G.edges():
                        G.edges[e1]['weight'] += A[eid]
                    else:
                        G.add_edge(*e1, weight=A[eid], d=size)

    return G


def extract_input_pairwise(A, hye, N):
    """
        Return the network with only pairwise interactions, i.e., a clique expansion of a hypergraph.

        INPUT
        -------
        A: ndarray
           Array of length E, containing the weights of the hyperedges.
        hye : ndarray
              Array of length E, containing the sets of hyperedges (as tuples).
        N : int
            Number of nodes.

        OUTPUT
        -------
        A2: ndarray
            Array containing the weights of the edges of the clique expansion network.
        hye2 : ndarray
               Array containing the sets of the edges (as tuples) of the clique expansion network.
        B2 : ndarray
             Incidence matrix of the pairwise clique expansion network.
    """

    Gtmp = HyG2Net(A, hye)

    hye2 = [(e[0], e[1]) for e in list(Gtmp.edges())]
    A2 = np.array([d['weight'] for u, v, d in list(Gtmp.edges(data=True))]).astype('int')

    B2 = np.zeros((N, A2.shape[0])).astype('int')
    for eid, e in enumerate(hye2):
        B2[np.array(e), eid] = A2[eid]

    return A2, hye2, B2


def normalize_nonzero_membership(u):
    """
        Given a matrix, it returns the same matrix normalized by row.

        INPUT
        -------
        u: ndarray
           Numpy matrix.

        OUTPUT
        -------
        The matrix normalized by row.
    """

    den1 = u.sum(axis=1, keepdims=True)
    nzz = den1 == 0.
    den1[nzz] = 1.

    return u / den1


def CalculatePermutation(U_infer, U0):
    """
        Permuting a membership matrix so that the groups from the two partitions correspond.
        U0 has dimension N x K, and it is the reference membership matrix.

        INPUT
        -------
        U_infer: ndarray
                 Inferred membership matrix.
        U0: ndarray
           Reference membership matrix.

        OUTPUT
        -------
        P : ndarray
            Permutation matrix.
    """

    N, RANK = U0.shape
    M = np.dot(np.transpose(U_infer), U0) / float(N)  # dim = RANK x RANK
    rows = np.zeros(RANK)
    columns = np.zeros(RANK)
    P = np.zeros((RANK, RANK))  # Permutation matrix
    for t in range(RANK):
        # Find the max element in the remaining sub-matrix,
        # the one with rows and columns removed from previous iterations
        max_entry = 0.
        c_index = 0
        r_index = 0
        for i in range(RANK):
            if columns[i] == 0:
                for j in range(RANK):
                    if rows[j] == 0:
                        if M[j, i] > max_entry:
                            max_entry = M[j, i]
                            c_index = i
                            r_index = j
        if max_entry > 0:
            P[r_index, c_index] = 1
            columns[c_index] = 1
            rows[r_index] = 1
    if (np.sum(P, axis=1) == 0).any():
        row = np.where(np.sum(P, axis=1) == 0)[0]
        if (np.sum(P, axis=0) == 0).any():
            col = np.where(np.sum(P, axis=0) == 0)[0]
            # print('P before:', P)
            P[row, col] = 1
            # print('P after:', P)
    return P


def cosine_similarity(U_infer, U0):
    """
        Compute the cosine similarity between ground-truth communities and detected communities.

        Parameters
        ----------
        U_infer : ndarray
                  Inferred membership matrix (detected communities), row-normalized.
        U0 : ndarray
             Ground-truth membership matrix (ground-truth communities), row-normalized.

        Returns
        -------
        RES : float
              Cosine similarity value.
    """

    P = CalculatePermutation(U_infer, U0)
    U_infer_tmp = np.dot(U_infer, P)  # permute inferred matrix
    U0_tmp = U0.copy()
    N, K = U0.shape
    cosine_sim = 0.
    norm_inf = np.linalg.norm(U_infer_tmp, axis=1)
    norm0 = np.linalg.norm(U0_tmp, axis=1)
    for i in range(N):
        if norm_inf[i] > 0.:
            U_infer_tmp[i, :] = U_infer_tmp[i, :] / norm_inf[i]
        if norm0[i] > 0.:
            U0_tmp[i, :] = U0_tmp[i, :] / norm0[i]

    for k in range(K):
        cosine_sim += np.dot(np.transpose(U_infer_tmp[:, k]), U0[:, k])

    return np.round(cosine_sim / float(N), 4)
