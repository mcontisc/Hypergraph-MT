"""
    Class definition of Hypergraph-MT, the algorithm to perform community detection in hypergraphs.
"""

from __future__ import print_function
import time
from termcolor import colored
import numpy as np
from scipy.special import comb
from scipy.optimize import root

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

EPS = 1e-3  # epsilon to fix numerical issues


# noinspection PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyBroadException
class HyMT:
    def __init__(self, rseed=0, inf=1e40, err_max=1e-3, err=0.01, N_real=1, tolerance=0.1, decision=1, max_iter=500,
                 fix_communities=False, fix_w=False, verbose=True, initialization=0, constraintU=True,
                 out_inference=False, out_folder='../data/output/', end_file='.dat', gammaU=0, gammaW=0,
                 plot_loglik=False, in_parameters='../data/input/theta.npz'):

        self.rseed = rseed  # random seed for the initialization
        self.inf = inf  # initial value of the log-likelihood
        self.err_max = err_max  # minimum value for the parameters
        self.err = err  # noise for the initialization
        self.N_real = N_real  # number of iterations with different random initialization
        self.tolerance = tolerance  # tolerance parameter for convergence
        self.decision = decision  # convergence parameter
        self.max_iter = max_iter  # maximum number of iteration steps before aborting

        if initialization not in {0, 1}:  # indicator for choosing how to initialize the parameters
            raise ValueError('The initialization parameter can be either 0 or 1. It is used as an indicator to '
                             'initialize the membership matrix u and the affinity matrix w. If it is 0, they will be '
                             'generated randomly; if it is 1 both u and w will be initialized through an input file.')
        self.initialization = initialization
        self.in_parameters = in_parameters  # path of the input files for u and w (when initialization is 1)
        self.fix_communities = fix_communities  # flag to keep the communities fixed
        self.fix_w = fix_w  # flag to keep the affinity matrix fixed
        if self.fix_w:
            if not self.initialization:
                raise ValueError('If fix_w=True, the initialization has to be 1.')
        if self.fix_communities:
            if not self.initialization:
                raise ValueError('If fix_communities=True, the initialization has to be 1.')

        self.gammaU = gammaU  # constant to regularize the communities
        self.gammaW = gammaW  # constant to regularize the affinity matrix
        self.constraintU = constraintU  # if True, then u is normalized such that every row sums to 1

        self.verbose = verbose  # flag to print details

        self.out_inference = out_inference  # flag to store the inferred parameters
        self.out_folder = out_folder  # path to store the output
        self.end_file = end_file  # output file suffix

        self.plot_loglik = plot_loglik  # flag to plot the log-likelihood

    # noinspection PyUnusedLocal
    def fit(self, data, hyperEdges, B, K=2):
        """
            Performing community detection on hypergraphs with a mixed-membership probabilistic model.

            INPUT
            -------
            data : ndarray
                   Array of length E, containing the weights of the hyperedges.
            hyperEdges : ndarray
                         Array of length E, containing the sets of hyperedges (as tuples).
            B : ndarray
                Incidence matrix of dimension NxE.
            K : int
                Number of communities.

            OUTPUT
            -------
            u_f : ndarray
                  Membership matrix.
            w_f : ndarray
                  Affinity matrix.
            maxL : float
                   Maximum log-likelihood value.
        """

        self.K = K
        self.N, self.E = B.shape
        self.it = 0
        best_loglik_values = []
        conv, best_r = None, None

        '''
        Pre-process data
        '''
        self.data = np.copy(data)
        self.B = np.copy(B)
        self.hyperEdges = np.copy(hyperEdges)  # same order as the columns of matrix B
        self.Bsubs = extract_indicesB(B)  # list of length N containing the indices of non-zero hyperedges for each node
        self.D = max([len(e) for e in hyperEdges])  # maximum observed hyperedge degree
        self.HyD2eId, self.HyeId2D = extract_indicesHy(self.hyperEdges)
        self.isolates = [i for i in range(self.N) if len(self.Bsubs[i]) == 0]  # isolated nodes
        self.non_isolates = [i for i in range(self.N) if len(self.Bsubs[i]) > 0]  # non-isolated nodes

        '''
        INFERENCE
        '''
        self.maxL = - self.inf  # initialization of the maximum log-likelihood
        self.rng = np.random.RandomState(self.rseed)

        for r in range(self.N_real):

            self._initialize(rng=self.rng, hyperEdges=hyperEdges)  # initialize u, w
            self._update_old_variables()
            self._initialize_psiOmega()  # initialize psiOmega and psiBarOmega
            self._random_initial_update_U()  # randomize u and update psiOmega and psiBarOmega
            self._update_rho()
            self._initialize_rho_subnz()

            # convergence local variables
            coincide, it = 0, 0
            convergence = False

            if self.verbose:
                print(f'Updating realization {r} ...', end='')

            loglik_values = []
            time_start = time.time()
            loglik = -self.inf

            while not convergence and it < self.max_iter:
                # noinspection PyUnusedLocal
                delta_u, delta_w = self._update_em()

                it, loglik, coincide, convergence = self._check_for_convergence(it, loglik, coincide, convergence)
                loglik_values.append(loglik)

            if self.verbose:
                print('done!')
                print(f'Nreal = {r+1} - Log-likelihood = {loglik} - iterations = {it} - '
                      f'time = {np.round(time.time() - time_start, 2)} seconds')

            if self.maxL < loglik:
                self._update_optimal_parameters()
                self.maxL = loglik
                self.final_it = it
                conv = convergence
                best_loglik_values = list(loglik_values)
                best_r = r

            self.rseed += 1

            # end cycle over realizations
        print(f'\nBest real = {best_r} - maxL = {self.maxL} - best iterations = {self.final_it}')

        if np.logical_and(self.final_it == self.max_iter, not conv):
            # convergence not reaches
            try:
                print(colored('Solution failed to converge in {0} EM steps!'.format(self.max_iter), 'blue'))
            except:
                print('Solution failed to converge in {0} EM steps!'.format(self.max_iter))

        if self.plot_loglik:
            plot_L(best_loglik_values, int_ticks=True)

        if self.out_inference:
            self._output_results()

        return self.u_f, self.w_f, self.maxL

    def _initialize(self, rng=None, hyperEdges=None):
        """
            Initialization of the parameters u and w.

            INPUT
            -------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
            hyperEdges : ndarray
                         Array of length E, containing the sets of hyperedges (as tuples).
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)

        if self.initialization == 0:
            if self.verbose:
                print('u and w are initialized randomly')
            self._randomize_w(rng=rng, hyperEdges=hyperEdges)
            self._randomize_u(rng=rng)

        elif self.initialization == 1:
            if self.verbose:
                print(f'u and w initialized using the input files: {self.in_parameters}')
            theta = np.load(self.in_parameters, allow_pickle=True)
            self._initialize_u(theta['u'])
            self._initialize_w(theta['w'])

    def _randomize_w(self, rng, hyperEdges=None):
        """
            Initialize affinity matrix w.

            INPUT
            -------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
            hyperEdges : ndarray
                         Array of length E, containing the sets of hyperedges (as tuples).
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)

        self.w = rng.random_sample((self.D - 1, self.K))
        if hyperEdges is not None:
            ds = np.array(list(set(np.arange(self.D - 1)).difference(set([len(e) - 2 for e in hyperEdges]))))
            if len(ds) > 0:
                print('setting certain d in w to zero:', ds)
                self.w[ds] = 0.

    def _randomize_u(self, rng=None):
        """
            Initialize membership matrix u.

            INPUT
            -------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)

        uk = rng.random_sample(self.K)
        self.u = np.tile(uk, [self.N, 1])

        row_sums = self.u.sum(axis=1)
        self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

    def _initialize_u(self, u0):
        """
            Initialize membership matrix u from file.

            INPUT
            -------
            u0 : ndarray
                 Membership matrix from file.
        """

        if u0.shape[0] != self.N:
            raise ValueError(f'u.shape[0] is different that the initialized one: {self.N} vs {u0.shape[0]}!')
        self.u = u0.copy()
        max_entry = np.max(u0)
        self.u += max_entry * self.err * np.random.random_sample(self.u.shape)

    def _initialize_w(self, w0):
        """
            Initialize affinity matrix w from file.

            INPUT
            -------
            w0 : ndarray
                 Affinity matrix from file.
        """

        if w0.shape[0] != self.D - 1:
            raise ValueError(f'w.shape[0] is different that the initialized one: {self.D - 1} vs {w0.shape[0]}!')
        self.w = w0.copy()
        max_entry = np.max(w0)
        self.w += max_entry * self.err * np.random.random_sample(self.w.shape)

    def _initialize_psiOmega(self, u0=None):
        """
            Initialize psi matrices psiOmega and psiBarOmega, i.e., psi(0)(Omega^d, k) and psi(0)(BarOmega^d, k).
            They have dimension DxK, and the first row refers to degree=1.

            INPUT
            -------
            u0 : ndarray
                 Array of length K used to initialize the matrix psiOmega.
        """

        self.psiBarOmega = np.zeros((self.D, self.K))

        if u0 is None:
            u0 = np.mean(self.u[self.non_isolates], axis=0)

        self.psiOmega = np.zeros((self.D, self.K))
        for k in range(self.K):
            Nk = np.count_nonzero(self.u[:, k])
            self.psiOmega[0, k] = np.sum(self.u[:, k])
            for d in range(1, self.D):
                self.psiOmega[d, k] = np.power(u0[k], d + 1) * comb(Nk, d + 1)

    def _random_initial_update_U(self, rng=None):
        """
            Random initialize membership matrix u, and update psiOmega and psiBarOmega matrices.

            INPUT
            -------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)

        for i in range(self.N):
            self._update_psiBarOmega(i)

            if i not in self.isolates:
                self.u[i] = rng.random_sample(self.K)
                self.u[i] /= self.u[i].sum()
                low_values_indices = self.u[i] < self.err_max  # values are too low
                self.u[i][low_values_indices] = 0.  # and set to 0.
            else:
                self.u[i] = np.zeros(self.K)

            self._update_psiOmega(i)

            self.u_old[i] = np.copy(self.u[i])

    def _update_old_variables(self):
        """
            Update values of the parameters in the previous iteration.
        """

        self.u_old = np.copy(self.u)
        self.w_old = np.copy(self.w)

    def _initialize_rho_subnz(self):
        """
            Initialize an array full of True, with the same dimension as rho i.e., ExK.
            It is useful to take track of the non-zero entries of rho.
        """

        self.rho_subsnz = np.ones((self.E, self.K)).astype('bool')

    def _update_rho(self, edgelist=None, mask=None):
        """
            Update the rho matrix that represents the variational distribution used in the EM routine.

            INPUT
            -------
            edgelist : list
                       List of hyperedges to update.
            mask : ndarray
                   Mask to hide the entries that do not need to be updated.
        """

        if edgelist is not None:
            for eid in edgelist:
                self.rho[eid] = self.w[self.HyeId2D[eid] - 2] * np.prod(self.u[np.array(self.hyperEdges[eid])], axis=0)
                Z = self.rho[eid].sum()
                if Z > 0:
                    self.rho[eid] /= Z
        elif mask is not None:
            for eid, k in np.stack(np.where(mask > 0)).T:
                self.rho[eid, k] = self.w[self.HyeId2D[eid] - 2, k] * np.prod(self.u[np.array(self.hyperEdges[eid]), k])

            Z = np.sum(self.rho, axis=1)
            Z[Z < self.err_max] = 1
            self.rho /= Z[:, np.newaxis]
        else:
            self.rho = np.zeros((self.E, self.K))
            for eid in range(self.E):
                self.rho[eid] = self.w[self.HyeId2D[eid] - 2] * np.prod(self.u[np.array(self.hyperEdges[eid])], axis=0)
                Z = self.rho[eid].sum()
                if Z > 0:
                    self.rho[eid] /= Z

    def _update_psiOmega(self, i, ks=None):
        """
            Update psiOmega matrix.

            INPUT
            -------
            i : int
                Row index.
            ks : int
                 Column index.
        """

        if ks is None:
            self.psiOmega[0] = self.psiOmega[0] + (self.u[i] - self.u_old[i])
            assert np.allclose(self.psiOmega[0], self.u.sum(axis=0))
            for d in range(1, self.D):
                self.psiOmega[d] = self.psiOmega[d] + (self.u[i] - self.u_old[i]) * self.psiBarOmega[d - 1]
        else:
            self.psiOmega[0][ks] = self.psiOmega[0][ks] + (self.u[i][ks] - self.u_old[i][ks])
            assert np.allclose(self.psiOmega[0], self.u.sum(axis=0))
            for d in range(1, self.D):
                self.psiOmega[d][ks] = self.psiOmega[d][ks] + \
                                       (self.u[i][ks] - self.u_old[i][ks]) * self.psiBarOmega[d - 1][ks]
                tmpMask = self.psiOmega[d] < 0
                if np.sum(tmpMask) > 0:
                    if (abs(self.psiOmega[d][tmpMask]) < EPS).all():
                        self.psiOmega[d][tmpMask] = np.zeros_like(self.psiOmega[d][tmpMask])

        # for debug
        if self.it == 0:
            if np.sum(self.psiOmega < 0) > 0:
                tmpMask = self.psiOmega < 0
                tmpMask2 = np.logical_and(tmpMask, abs(self.psiOmega) < EPS)
                self.psiOmega[tmpMask2] = abs(self.psiOmega[tmpMask2])
                tmpMask = self.psiOmega < 0
                if tmpMask.sum() > 0:
                    print('psiOmega', self.psiOmega[tmpMask])
                    print(np.where(tmpMask))

                self.it += 1
                print('i=', i, self.Bsubs[i])

    def _update_psiBarOmega(self, i, ks=None):
        """
            Update psiBarOmega matrix.

            INPUT
            -------
            i : int
                Row index.
            ks : int
                 Column index.

            OUTPUT
            -------
            success : bool
                      Flag to check whether the matrix psiBarOmega has all non-negative entries.
        """

        success = True
        if ks is None:
            self.psiBarOmega[0] = self.psiOmega[0] - self.u[i]
            for d in range(1, self.D):
                self.psiBarOmega[d] = self.psiOmega[d] - self.u[i] * self.psiBarOmega[d - 1]
        else:
            self.psiBarOmega[0][ks] = self.psiOmega[0][ks] - self.u[i][ks]
            for d in range(1, self.D):
                self.psiBarOmega[d][ks] = self.psiOmega[d][ks] - self.u[i][ks] * self.psiBarOmega[d - 1][ks]

        tmpMask = self.psiBarOmega < 0
        if np.sum(tmpMask) > 0:
            if (abs(self.psiBarOmega[tmpMask]) < EPS).all():
                self.psiBarOmega[tmpMask] = np.zeros_like(self.psiBarOmega[tmpMask])
        tmpMask = self.psiBarOmega < 0
        if np.sum(tmpMask) > 0:
            success = False

        return success

    def _update_em(self):
        """
            Expectation-Maximization routine.
        """

        if not self.fix_w:
            d_w = self._update_W()
        else:
            d_w = 0
        self._update_rho()

        if not self.fix_communities:
            d_u = self._update_U()
        else:
            d_u = 0
        self._update_rho()

        return d_u, d_w

    def _update_U(self):
        """
            Update membership matrix u. It is parallel for all the k of a node, but sequential in nodes (with random
            permutation of the nodes).
        """

        perm = self.rng.permutation(range(self.N))
        for i in perm:
            ks = np.where(self.u[i] > self.err_max)[0]
            if ks.shape[0] != 0:
                success = self._update_psiBarOmega(i, ks=ks)
                if success:
                    u_tmp = np.einsum('I,Ik->k', self.B[i][self.Bsubs[i]], self.rho[self.Bsubs[i]][:, ks])
                    u_tmp_den = self.gammaU + np.sum(self.w[:, ks] * self.psiBarOmega[:-1, ks], axis=0)  # sum over d

                    if self.constraintU:
                        low_values_indices = (u_tmp / u_tmp_den) < self.err_max
                        u_tmp[low_values_indices] = 0

                        lambda_i = enforce_constraintU(u_tmp, u_tmp_den)
                        self.u[i, ks] = u_tmp / (lambda_i + u_tmp_den)
                    else:
                        self.u[i, ks] = u_tmp / u_tmp_den

                    tmpMask = self.u[i, ks] < 0
                    if np.sum(tmpMask) > 0:
                        if abs(self.u[i, ks][tmpMask]).any() > 1e-01:
                            self.u[i, ks] = abs(self.u[i, ks])
                        else:
                            print('WARNING!', i, self.u[i])

                    low_values_indices = self.u[i] < self.err_max  # values are too low
                    self.u[i][low_values_indices] = 0  # and set to 0.

                    high_values_indices = self.u[i] > 1e2  # values are too high
                    self.u[i][high_values_indices] = 1e2  # and set to 100.

                    self._update_psiOmega(i, ks=ks)

            self.u_old[i] = np.copy(self.u[i])

        dist_u = np.amax(abs(self.u - self.u_old))

        return dist_u

    def _update_W(self):
        """
            Update affinity matrix w.
        """

        for d in range(self.D - 1):
            self.w[d] = np.einsum('I,Ik->k', self.data[self.HyD2eId[d]], self.rho[self.HyD2eId[d]])
            Z = self.gammaW + self.psiOmega[d + 1]

            non_zeros = Z > 0
            self.w[d, non_zeros] /= Z[non_zeros]

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _check_for_convergence(self, it, loglik, coincide, convergence):
        """
            Check for convergence by using the log-likelihood.

            INPUT
            -------
            it : int
                 Number of iteration.
            loglik : float
                     Log-likelihood value.
            coincide : int
                       Number of time the update of the log-likelihood respects the tolerance.
            convergence : bool
                          Flag for convergence.

            OUTPUT
            -------
            it : int
                 Number of iteration.
            loglik : float
                     Log-likelihood value.
            coincide : int
                       Number of time the update of the log-likelihood respects the tolerance.
            convergence : bool
                          Flag for convergence.
        """

        if it % 1 == 0:
            old_L = loglik
            loglik = self.__Likelihood()
            if abs(loglik - old_L) < self.tolerance:
                coincide += 1
            else:
                coincide = 0
        if coincide > self.decision:
            convergence = True
        it += 1

        return it, loglik, coincide, convergence

    def __Likelihood(self, eps=1e-300):
        """
            Compute the log-likelihood of the data.

            INPUT
            -------
            eps : int
                  Random noise.

            OUTPUT
            -------
            ll : float
                 Log-likelihood value.
        """

        ll = 0.
        ll -= self.gammaW * np.sum(self.w) + self.gammaU * np.sum(self.u)

        for d in range(self.D - 1):
            wpsiOmega = self.w[d] * self.psiOmega[d + 1]  # K-dim
            ll -= np.sum(wpsiOmega)
            for eid in self.HyD2eId[d]:
                tmp = self.w[d] * np.prod(self.u[np.array(self.hyperEdges[eid])], axis=0)
                tmp = max(0, np.sum(tmp))
                ll += self.data[eid] * np.log(tmp + eps)

        if np.isnan(ll):
            print("Log-likelihood is NaN!!")
            return - self.inf
        else:
            return ll

    def _update_optimal_parameters(self):
        """
            Update values of the parameters after convergence.
        """

        self.u_f = np.copy(self.u)
        self.w_f = np.copy(self.w)

    def _output_results(self):
        """
            Function to output the results.
        """

        outfile = self.out_folder + 'theta' + self.end_file
        np.savez_compressed(outfile + '.npz', u=self.u_f, w=self.w_f, max_it=self.final_it,
                            maxL=self.maxL, non_isolates=self.non_isolates)
        print(f'\nInferred parameters saved in: {outfile + ".npz"}')
        print('To load: theta=np.load(filename), then e.g. theta["u"]')


def func_lagrange_multiplier(lambda_i, num, den):
    """
        Return the objective function to find the lagrangian multiplier to enforce the constraint on the matrix u.
    """

    f = num / (lambda_i + den)

    return np.sum(f) - 1


def extract_indicesB(B):
    """"
        List of length N containing the indices of non-zero hyperedges for every node.

        INPUT
        -------
        B : ndarray
            Incidence matrix of dimension NxE.

        OUTPUT
        -------
        Bsubs : list
                List of length N containing the indices of non-zero hyperedges for every node.
    """

    N = B.shape[0]
    Bsubs = [np.where(B[i] > 0)[0] for i in range(N)]

    return Bsubs


def extract_indicesHy(hyperEdges):
    """"
        Lists containing information about the hyperedges.

        INPUT
        -------
        hyperEdges : ndarray
                     Array of length E, containing the sets of hyperedges (as tuples).

        OUTPUT
        -------
        HyD2eId : list
                  List of list containing the indices of hyperedges with a given degree.
        HyeId2D : list
                  List of length E, containing the degree of each hyperedge.
    """

    HyeId2D = np.array([len(e) for e in hyperEdges])
    HyD2eId = [list(np.where(HyeId2D == d)[0]) for d in np.arange(2, np.max(HyeId2D + 1))]

    return HyD2eId, HyeId2D


def enforce_constraintU(num, den):
    """
        Return the lagrangian multiplier to enforce the constraint on the matrix u.

        INPUT
        -------
        num : float
              Numerator of the update of the membership matrix u.
        den : float
              Numerator of the update of the membership matrix u.

        OUTPUT
        -------
        lambda_i : float
                   Lagrangian multiplier.
    """

    lambda_i_test = root(func_lagrange_multiplier, x0=np.array([0.1]), args=(num, den))
    lambda_i = lambda_i_test.x

    return lambda_i


def plot_L(values, indices=None, k_i=5, figsize=(4, 3), int_ticks=False, xlab='Iterations'):
    """
        Function to plot the log-likelihood.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if indices is None:
        ax.plot(values[k_i:])
    else:
        ax.plot(indices[k_i:], values[k_i:])
    ax.set_xlabel(xlab)
    ax.set_ylabel('Log-likelihood values')
    if int_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()

    plt.tight_layout()
    plt.show()
