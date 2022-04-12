"""
    Class for visualizing and handling communities
"""

from __future__ import print_function
import time
import numpy as np
import pandas as pd
import tools as tl
import networkx as nx
from os.path import isfile
from collections import OrderedDict
from itertools import combinations


class vizCM:
    def __init__(self, A, hye, U, verbose=False, threshold=0.1, gt_labels=None, spring_layout=None,isolates=None):

        self.hyL = [len(e) for eid, e in enumerate(hye)]
        self.G = HyG2Net(A, hye)
        if isolates is not None: self.G.add_nodes_from(isolates)

        self.nodes = list(self.G.nodes())
        self.edges = list(self.G.edges())
        self.N = self.G.number_of_nodes()
        self.E = self.G.number_of_edges()

        self.nodeName2Id = {}
        self.nodeIdName = {}
        for c, n in enumerate(self.nodes):
            self.nodeName2Id[n] = c
            self.nodeIdName[c] = n

        self.U = {a: normalize_nonzero_membership(U[a]) for a in U.keys()}
        self.K = self.U['HyMT'].shape[1]

        if gt_labels is not None:
            self.gt_labels2U_gt(gt_labels)
        # print(self.U['gt'].shape)

        self.U = permute_membership(self.U, ref_layer='gt')

        self.groups, self.community_H, self.community_S = self.extract_community(self.U, threshold=threshold,
                                                                                 verbose=verbose)
       
        if gt_labels is not None:
            self.nodelist = sort_nodelist_by_community(self.community_H['gt'])
        else:
            self.nodelist = sort_nodelist_by_community(self.community_H['HyMT'])

        self.bridges = self.extract_bridges(self.U)
        self.pos = self.assign_position(delta=0.5, spring_layout=spring_layout)
        self.degree, self.node_color = self.assign_color()

    def extract_community(self, U, threshold=0.1, verbose=False):

        algos = list(U.keys())

        '''
        Hard membership
        '''
        groups = {}
        for l in algos:
            groups[l] = extract_hard_membership(U[l])
        '''
        Soft membership
        '''
        community_H, community_S = {}, {}
        for l in algos:
            community_H[l], community_S[l] = {}, {}
            community_H[l] = extract_communities_hard(groups[l], verbose=verbose)
            community_S[l] = extract_communities_soft(U[l], threshold=threshold, verbose=verbose)

        return groups, community_H, community_S

    def extract_bridges(self, u, threshold=0.1):

        bridges = {}

        algos = list(u.keys())
        for l in algos:
            bridges[l] = extract_bridges_routine(u[l], threshold=threshold)

        return bridges

    def assign_pos_based_on_color(self, nodes, groups, delta=0.5):
        C = max(np.unique(groups)) +1
        pos_groups = nx.circular_layout(nx.cycle_graph(C))
        pos = {}
        for c, i in enumerate(nodes):
            r = np.random.rand() * 2 * np.math.pi
            radius = np.random.rand()
            pos[i] = pos_groups[groups[self.nodeName2Id[i]]] + delta * radius * np.array(
                [np.math.cos(r), np.math.sin(r)])
        return pos

    def assign_position(self, delta=0.5, spring_layout=False):
        algos = list(self.groups.keys())
        pos = {}
        for a in algos:
            if spring_layout:
                pos[a] = nx.spring_layout(self.G)
            else:
                pos[a] = self.assign_pos_based_on_color(self.nodes, self.groups[a], delta=delta)
        return pos

    def assign_color(self):
        algos = list(self.groups.keys())
        # print(algos)
        degree = {n: self.G.degree[n] for n in self.nodes}
        node_color = {}
        for a in algos:
            node_color[a] = {}
            for c, i in enumerate(self.nodes):
                node_color[a][i] = self.groups[a][c]
        return degree, node_color

    def set_node_attributes(self, A, algo='HyMT'):
        nx.set_node_attributes(A, self.pos[algo], "pos_" + algo)
        nx.set_node_attributes(A, self.node_color[algo], "node_color_" + algo)
        nx.set_node_attributes(A, self.degree, "node_size_degree")
        return A

    def gt_labels2U_gt(self, groups):
        self.U['gt'] = np.zeros((self.N, self.K))
        for i in range(self.U['gt'].shape[0]):
            self.U['gt'][i][groups[i]] = 1


def HyG2Net(A, hye):
    G = nx.Graph()
    if isinstance(hye, tuple):  # individual edge
        l = len(hye)
        comb = combinations(hye, 2)
        for e in list(comb):
            if e in G:
                G.edges[e]['weight'] += A
            else:
                G.add_edge(*e, weight=A, d=l)
    else:
        for eid, e in enumerate(hye):
            l = len(e)
            if l == 2:
                if e in G:
                    G.edges[e]['weight'] += A[eid]
                else:
                    G.add_edge(*e, weight=A[eid], d=2)
            elif l > 2:
                comb = combinations(e, 2)
                for e1 in list(comb):
                    if e1 in G:
                        G.edges[e1]['weight'] += A[eid]
                    else:
                        G.add_edge(*e1, weight=A[eid], d=l)
    return G


def extract_hard_membership(U):
    K = U.shape[1]
    groups = U.argmax(axis=1)

    den = U.sum(axis=1)
    nzz = den == 0.
    groups[nzz] = K
    return groups


def extract_bridges_routine(U, threshold=0.1):
    bridges = {}
    for i in range(U.shape[0]):
        k = (U[i] > threshold).sum()
        bridges[i] = k
    return bridges


def extract_bridge_properties(i, cm, U, threshold=0.1):
    groups = np.where(U[i] > threshold)[0]
    wedge_sizes = U[i][groups]
    wedge_colors = [cm(c) for c in groups]
    return wedge_sizes, wedge_colors


def extract_communities_soft(U, threshold=0.1, verbose=False):
    N, C = U.shape
    community = {}
    for k in range(C):
        community[k] = np.where(U[:, k] > threshold)[0]
        if verbose:
            print(k, community[k].shape[0])
    if verbose:
        print()

    return community


def extract_communities_hard(groups_u, verbose=False):
    """
    Uses only the max entry, stored in groups_v
    """
    community = {}
    for k in np.unique(groups_u):
        community[k] = np.where(groups_u == k)[0]
        if verbose:
            print(k, community[k].shape[0])
    if verbose:
        print()

    return community


def sort_nodelist_by_community(community):
    nodelist = []
    for k in community.keys():
        nodelist.extend(community[k])
    nodelist = list(OrderedDict.fromkeys(nodelist))  # removes duplicates while preserving the order
    return nodelist


def print_cc(A_aggr, threshold=10):
    cc = nx.weakly_connected_components(A_aggr)
    for i, c in enumerate(sorted(cc, key=len, reverse=True)):
        print(i, len(c))
        if i > threshold:
            break


def permute_membership(U, ref_layer='gt'):
    algos = list(U.keys())
    K_ref = U[ref_layer].shape[1]
    for a in algos:
        if a != ref_layer:
            K_a = U[a].shape[1]
            # print(a,K_a,K_ref)
            if K_a == K_ref:
                P = CalculatePermutation(U[a], U[ref_layer])
                U[a] = np.dot(U[a], P)  # Permute inferred matrix
            else:
                if K_a < K_ref:
                    U1 = np.zeros_like(U[ref_layer])
                    U1[:, :K_a] = U[a].copy()
                    P = CalculatePermutation(U1, U[ref_layer])
                    U[a] = np.dot(U1, P)  # Permute inferred matrix
                    U[a] = U[a][:, ~np.all(U[a] == 0, axis=0)]
                    assert U[a].shape[1] <= K_a
                else:
                    U1 = np.zeros_like(U[a])
                    U1[:, :K_ref] = U[ref_layer].copy()
                    P = CalculatePermutation(U[a], U1)
                    U[a] = np.dot(U[a], P)  # Permute inferred matrix
    return U


def CalculatePermutation(U_infer, U0):
    """
    Permuting the overlap matrix so that the groups from the two partitions correspond
    U0 has dimension NxK, reference membership
    """
    N, RANK = U0.shape
    M = np.dot(np.transpose(U_infer), U0) / float(N)  # dim=RANKxRANK
    rows = np.zeros(RANK)
    columns = np.zeros(RANK)
    P = np.zeros((RANK, RANK))  # Permutation matrix
    for t in range(RANK):
        # Find the max element in the remaining submatrix,
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
    if (np.sum(P,axis=1) == 0).any():
        row = np.where(np.sum(P,axis=1) == 0)[0]
        if (np.sum(P,axis=0) == 0).any():
            col = np.where(np.sum(P,axis=0) == 0)[0]
            # print('P before:',P)
            P[row,col] = 1
            # print('P after:',P)
    return P


def normalize_nonzero_membership(U):
    """
        Given a matrix, it returns the same matrix normalized by row.

        Parameters
        ----------
        U: ndarray
           Numpy Matrix.

        Returns
        -------
        The matrix normalized by row.
    """

    den1 = U.sum(axis=1, keepdims=True)
    nzz = den1 == 0.
    den1[nzz] = 1.

    return U / den1
