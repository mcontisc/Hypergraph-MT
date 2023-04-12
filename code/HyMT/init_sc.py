"""
    Implementation of Hypergraph Spectral Clustering:
    D.Zhou, J.Huang, B.Schölkopf, Learning with hypergraphs: Clustering, classification, and embedding.
    Advances in neural information processing systems 19 (2006).
"""

import numpy as np
from typing import Tuple
from sklearn.cluster import KMeans


class HySC:
    def __init__(
        self,
        rseed: int = 0,
        inf: int = 1e40,
        N_real: int = 10,
        verbose: bool = True,
        out_inference: bool = False,
        out_folder: str = "../data/output/",
        end_file: str = "_sc.dat",
    ) -> None:

        self.rseed = rseed  # random seed for the initialization
        self.inf = inf  # initial value of the log-likelihood
        self.N_real = N_real  # number of iterations with different random initialization
        self.verbose = verbose  # flag to print details

        self.out_inference = out_inference  # flag for storing the inferred parameters
        self.out_folder = out_folder  # path for storing the output
        self.end_file = end_file  # output file suffix

    def fit(self, B: np.array, K: int, weighted_L: bool = False) -> np.array:
        """
        Performing community detection on hypergraphs with spectral clustering.

        Parameters
        ----------
        B : ndarray
            Incidence matrix of dimension NxE.
        K : int
            Number of communities.
        weighted_L : bool
                     Flag to use the weighted Laplacian.
        """

        self.K = K
        self.N, self.E = B.shape

        self.it = 0

        """
        Pre-process data
        """
        self.B = np.copy(B)
        self.binary_incidence()
        self.extract_degrees()
        self.D = max(self.edge_degree)  # maximum observed hyperedge degree
        self.isolates = list(np.where(self.node_degree == 0)[0])
        self.non_isolates = list(np.where(self.node_degree > 0)[0])

        self.extract_laplacian(weighted=weighted_L)

        """
        INFERENCE
        """
        e_vals, e_vecs = self.extract_eigenvectors(self.L, self.K)
        self.u = self.apply_kmeans(e_vecs.real, seed=self.rseed)

        if self.out_inference:
            self.output_results()

        return self.u

    def binary_incidence(self) -> None:
        """
        Binarize the incidence matrix of dimension NxE.
        """

        self.B_binary = np.copy(self.B)
        self.B_binary[self.B_binary > 1] = 1

    def extract_degrees(self) -> None:
        """
        Extract node and hyperedge degree sequences.
        """

        self.node_degree = np.sum(self.B_binary, axis=1)
        self.edge_degree = np.sum(self.B_binary, axis=0)
        self.node_degree_weighted = np.sum(self.B, axis=1)
        self.edge_degree_weighted = np.sum(self.B, axis=0)

    def extract_laplacian(self, weighted: bool = False) -> None:
        """
        Extract the Laplacian associated to the hypergaph.
        """

        invDE = np.diag(1.0 / self.edge_degree)
        invDV2 = np.diag(np.sqrt(1.0 / self.node_degree))
        # set to zero for isolated nodes
        invDV2 = np.diag(np.where(invDV2, 0, invDV2))
        if weighted:
            HT = self.B.T
            self.L = np.eye(self.N) - invDV2 @ self.B @ invDE @ HT @ invDV2
        else:
            HT = self.B_binary.T
            self.L = np.eye(self.N) - invDV2 @ self.B_binary @ invDE @ HT @ invDV2

    def extract_eigenvectors(self, L: np.array, K: int) -> Tuple[np.array, np.array]:
        """
        Extract eigenvalues and eigenvectors of the hypergraph Laplacian.
        """

        e_vals, e_vecs = np.linalg.eig(L[self.non_isolates][:, self.non_isolates])
        sorted_indices = np.argsort(e_vals)

        return e_vals[sorted_indices[:K]], e_vecs[:, sorted_indices[1:K]]

    def apply_kmeans(self, X: np.array, seed: int = 10) -> np.array:
        """
        Apply K-means algorithm to the eigenvectors of the hypergraph Laplacian.
        """

        y_pred = KMeans(
            n_clusters=self.K, random_state=seed, n_init=self.N_real
        ).fit_predict(X)

        X_pred = np.zeros((self.N, self.K))
        for idx, i in enumerate(self.non_isolates):
            X_pred[i, y_pred[idx]] = 1

        return X_pred

    def output_results(self) -> None:
        """
        Function to output the results.
        """

        outfile = self.out_folder + "theta" + self.end_file
        np.savez_compressed(outfile + ".npz", u=self.u)
        print(f'\nInferred parameters saved in: {outfile + ".npz"}')
        print('To load: theta=np.load(filename), then e.g. theta["u"]')
