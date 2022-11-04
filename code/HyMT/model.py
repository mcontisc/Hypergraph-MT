"""
    Class definition of Hypergraph-MT, the algorithm to perform inference and community detection in hypergraphs.
"""

from __future__ import print_function
from typing import Union, Tuple, List
import time
import os
from termcolor import colored
import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.optimize import root

import time_glob as gl

from .log import setup_logging
from .init_sc import HySC

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DEFAULT_ERR = 0.001
DEFAULT_ERR_MAX = 1e-5
DEFAULT_NUM_REALIZATIONS = 15
DEFAULT_MAX_ITER = 500
DEFAULT_LOGLIK_STEPS = 1
DEFAULT_CONVERGENCE_TOL = 0.1
DEFAULT_DECISION = 15
DEFAULT_PLOT_LOGLIK = True
DEFAULT_VERBOSE = True  # flag to print details
DEFAULT_SEED = 10
DEFAULT_INF = 1e10
DEFAULT_EPS = 1e-20


class HyMT:
    def __init__(
        self,
        err: float = DEFAULT_ERR,  # noise to initialize u or w around input values
        err_max: float = DEFAULT_ERR_MAX,  # minimum value for the parameters
        num_realizations: int = DEFAULT_NUM_REALIZATIONS,  # number of iterations with different random initialization
        max_iter: int = DEFAULT_MAX_ITER,  # maximum number of iteration steps before aborting
        loglik_steps: int = DEFAULT_LOGLIK_STEPS,  # after how many step to update the log-likelihood value
        convergence_tol: float = DEFAULT_CONVERGENCE_TOL,  # tolerance parameter for convergence
        decision: int = DEFAULT_DECISION,  # convergence parameter
        verbose: bool = DEFAULT_VERBOSE,  # flag to print details
    ) -> None:

        self.err_max = err_max
        self.err = err
        self.num_realizations = num_realizations
        self.max_iter = max_iter
        self.loglik_steps = loglik_steps
        self.convergence_tol = convergence_tol
        self.decision = decision
        self.verbose = verbose

        self.logger = setup_logging("hymt.model.HyMT", verbose)

    def __check_fit_params(
        self,
        data: np.ndarray,
        hyperEdges: np.ndarray,
        B: np.ndarray,
        K: int,
        seed: Union[int, None] = None,
        constraintU: bool = False,
        **extra_params,
    ) -> None:
        """
        Pre-processing data and initialization parameters for the fit function.
        """

        self._set_seed(seed)

        self.data = data
        self.hyperEdges = hyperEdges  # same order as the columns of matrix B
        self.B = B
        self.B_binarize = binarize_B(self.B)

        self.K = K
        self.D = max([len(e) for e in self.hyperEdges])  # maximum observed degree
        self.N, self.E = self.B.shape
        if (len(self.data) != self.E) or (len(self.hyperEdges) != self.E):
            raise ValueError(
                f"Wrong dimension in data given input! "
                f"data.shape = {data.shape}, hyperEdges.shape = {hyperEdges.shape}, B.shape = {B.shape}"
            )

        self.Bsubs = extract_indicesB(B)
        self.HyD2eId, self.HyeId2D = extract_indicesHy(self.hyperEdges)
        self.isolates = [
            i for i in range(self.N) if len(self.Bsubs[i]) == 0
        ]  # isolated nodes
        self.non_isolates = [
            i for i in range(self.N) if len(self.Bsubs[i]) > 0
        ]  # non-isolated nodes

        self.constraintU = (
            constraintU  # if True, then u is normalized such that every row sums to 1
        )
        available_extra_params = [
            "fix_communities",  # flag to keep the communities fixed
            "fix_w",  # flag to keep the affinity matrix fixed
            "gammaU",  # constant to regularize the communities
            "gammaW",  # constant to regularize the affinity matrix
            "initialize_u",  # whether to initialize u from input
            "initialize_w",  # whether to initialize w from input
            "out_inference",  # flag to store the inferred parameters
            "out_folder",  # path to store the output
            "end_file",  # output file suffix
            "plot_loglik",  # flag to plot the log-likelihood
        ]
        for extra_param in extra_params:
            if extra_param not in available_extra_params:
                msg = "Ignoring unrecognised parameter %s." % extra_param
                self.logger.warn(msg)

        if "fix_communities" in extra_params:
            self.fix_communities = extra_params["fix_communities"]
        else:
            self.fix_communities = False

        if "fix_w" in extra_params:
            self.fix_w = extra_params["fix_w"]
        else:
            self.fix_w = False

        if "gammaU" in extra_params:
            self.gammaU = extra_params["gammaU"]
        else:
            self.gammaU = 0

        if "gammaW" in extra_params:
            self.gammaW = extra_params["gammaW"]
        else:
            self.gammaW = 0

        u0 = None
        if "initialize_u" in extra_params:
            if extra_params["initialize_u"] is not None:
                u0 = self._initialize_u0_from_input(extra_params["initialize_u"])
        self.u0 = u0

        w0 = None
        if "initialize_w" in extra_params:
            if extra_params["initialize_w"] is not None:
                w0 = self._initialize_w0_from_input(extra_params["initialize_w"])
        self.w0 = w0

        if "out_inference" in extra_params:
            self.out_inference = extra_params["out_inference"]
        else:
            self.out_inference = True

        if "out_folder" in extra_params:
            self.out_folder = extra_params["out_folder"]
        else:
            self.out_folder = "../data/output/"

        if "end_file" in extra_params:
            self.end_file = extra_params["end_file"]
        else:
            self.end_file = ""

        if "plot_loglik" in extra_params:
            self.plot_loglik = extra_params["plot_loglik"]
        else:
            self.plot_loglik = DEFAULT_PLOT_LOGLIK

    def _set_seed(self, seed: Union[int, None]) -> None:
        """
        Set the container for the Mersenne Twister pseudo-random number generator.
        """

        if seed is None:
            seed = DEFAULT_SEED
        self.seed = seed
        self.prng = np.random.RandomState(seed)

    def _initialize_u0_from_input(self, input_val: Union[np.array, str]) -> np.array:

        if isinstance(input_val, np.ndarray):
            if input_val.shape == (self.N, self.K):
                u0 = self.u0_from_nparray(input_val)
            else:
                msg = f"u0 must have shape {(self.N, self.K)}. In input was given {input_val.shape}!"
                self.logger.error(msg)
                raise ValueError(msg)
        elif isinstance(input_val, str):
            if input_val == "spectral":  # Hypergraph Spectral Clustering
                u0 = calculate_u_SC(self.B, self.K, seed=self.seed)
            else:
                if os.path.exists(input_val):
                    u0 = self.u0_from_file(input_val)
                else:
                    msg = f"{input_val} does not exist!"
                    self.logger.error(msg)
                    raise ValueError(msg)
        return u0

    def _initialize_w0_from_input(self, input_val: Union[np.array, str]) -> np.array:

        if isinstance(input_val, np.ndarray):
            if input_val.shape == (self.D - 1, self.K):
                w0 = self.w0_from_nparray(input_val)
            else:
                msg = f"w0 must have shape {(self.D - 1, self.K)}. In input was given {input_val.shape}!"
                self.logger.error(msg)
                raise ValueError(msg)
        elif isinstance(input_val, str):
            if os.path.exists(input_val):
                w0 = self.w0_from_file(input_val)
            else:
                msg = f"{input_val} does not exist!"
                self.logger.error(msg)
                raise ValueError(msg)
        return w0

    def u0_from_nparray(self, input_val: np.array) -> np.array:

        return input_val

    def u0_from_file(self, filename: str) -> np.array:

        theta = np.load(filename)
        return theta["u"]

    def w0_from_nparray(self, input_val: np.array) -> np.array:

        return input_val

    def w0_from_file(self, filename: str) -> np.array:

        theta = np.load(filename)
        return theta["w"]

    @gl.timeit("fit")
    def fit(
        self,
        data: np.ndarray,
        hyperEdges: np.ndarray,
        B: np.ndarray,
        K: int,
        seed: int = None,
        constraintU: bool = False,
        **extra_params,
    ) -> Tuple[np.array, np.array, float]:
        """
        Performing community detection on hypergraphs with a mixed-membership probabilistic model.

        Parameters
        ----------
        data : ndarray
               Array of length E, containing the weights of the hyperedges.
        hyperEdges : ndarray
                     Array of length E, containing the sets of hyperedges (as tuples).
        B : ndarray
            Incidence matrix of dimension NxE.
        K : int
            Number of communities.
        seed : int
               Random seed.
        constraintU : bool
                      If True, then u is normalized such that every row sums to 1.
        **extra_params
                    Additional keyword arguments handed to __check_fit_params to handle u and w.

        Returns
        -------
        u_f : ndarray
              Membership matrix.
        w_f : ndarray
              Affinity matrix.
        maxL : float
               Maximum log-likelihood value.
        """

        self.logger.debug("Checking user parameters passed to the HyMT.fit()")

        self.__check_fit_params(
            data=data,
            hyperEdges=hyperEdges,
            B=B,
            K=K,
            seed=seed,
            constraintU=constraintU,
            **extra_params,
        )

        """
        INFERENCE
        """
        self.it = 0
        conv = None
        best_loglik_values = []
        train_info = (
            []
        )  # keep track of log-likelihood values, running time, and other training info

        self.maxL = -DEFAULT_INF  # initialization of the maximum log-likelihood

        for r in range(self.num_realizations):

            self._initialize_psiOmega()  # initialize psiOmega and psiBarOmega

            if r == 0:  # initialize around the baseline
                self._initialize_uw0_realization(
                    hyperEdges=self.hyperEdges, baseline=True
                )
            else:
                self._initialize_uw0_realization(
                    hyperEdges=self.hyperEdges, baseline=False
                )

            self._random_initial_update_U()  # randomize u and update psiOmega and psiBarOmega
            self._update_rho()

            # convergence local variables
            coincide, it = 0, 0
            convergence = False

            if self.verbose:
                print(f"Updating realization {r} ...", end="\n")

            loglik_values = []
            loglik = -DEFAULT_INF

            while not convergence and it < self.max_iter:

                time_start = time.time()
                delta_u, delta_w = self._update_em()

                it, loglik, coincide, convergence = self._check_for_convergence(
                    it, loglik, coincide, convergence
                )

                runtime = time.time() - time_start
                if it % self.loglik_steps == 0:
                    train_info.append((r, self.seed, it, loglik, runtime, convergence))

                loglik_values.append(loglik)

            self.logger.debug(f"N_real={r}--num it={it}--Loglikelihood:{loglik}")

            if self.maxL < loglik:
                self._update_optimal_parameters()
                self.maxL = loglik
                self.final_it = it
                conv = convergence
                best_loglik_values = list(loglik_values)

            self._set_seed(self.seed + self.prng.randint(1, 1e6))

        # end cycle over realizations

        cols = [
            "realization",
            "seed",
            "iter",
            "loglik",
            "runtime",
            "reached_convergence",
        ]
        self.train_info = pd.DataFrame(train_info, columns=cols)

        if np.logical_and(self.final_it == self.max_iter, not conv):
            # convergence not reaches
            try:
                print(
                    colored(
                        "Solution failed to converge in {0} EM steps!".format(
                            self.max_iter
                        ),
                        "blue",
                    )
                )
            except:
                print(
                    "Solution failed to converge in {0} EM steps!".format(self.max_iter)
                )

        if self.plot_loglik:
            plot_L(best_loglik_values, int_ticks=True)

        if self.out_inference:
            self._output_results()

        return self.u_f, self.w_f, self.maxL

    def _set_dummy_u0(self) -> None:
        """
        Initial dummy u0 to compute psiOmega in closed-form at step t==0.
        """

        uk = self.prng.random_sample(self.K)
        self.u0_dummy = np.tile(uk, [self.N, 1])

        row_sums = self.u0_dummy.sum(axis=1)
        self.u0_dummy[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

    def _initialize_uw0_realization(
        self, hyperEdges: np.array, baseline: bool = False
    ) -> None:
        """
        Initialization of the parameters u and w.

        Parameters
        ----------
        hyperEdges : ndarray
                     Array of length E, containing the sets of hyper-edges (as tuples).
        baseline : bool
                   Flag to initialize u from baseline.
        """

        if baseline:  # TODO: add other baselines
            if self.verbose:
                print("u0 is initialized from baseline")
            self.u0_current_real_t0 = calculate_u_SC(self.B, self.K, seed=self.seed)
            max_entry = np.max(self.u0_current_real_t0)
            self.u0_current_real_t0 += (
                max_entry
                * self.err
                * self.prng.random_sample(self.u0_current_real_t0.shape)
            )
        else:
            if self.u0 is None:
                if self.verbose:
                    print("u0 is initialized randomly")
                self.u0_current_real_t0 = self._randomize_u0()
            else:
                if self.verbose:
                    print(f"u0 is initialized from input")
                self.u0_current_real_t0 = self._initialize_u(self.u0)

        if self.w0 is None:
            if self.verbose:
                print("w0 is initialized randomly")
            self.w = self._randomize_w0(hyperEdges=hyperEdges)
        else:
            if self.verbose:
                print(f"w0 is initialized from input")
            self.w = self._initialize_w(self.w0)
        self.w_old = np.copy(self.w)

    def _randomize_u0(self) -> np.array:
        """
        Initialize membership matrix u.
        """

        u0 = self.prng.random_sample((self.N, self.K))

        row_sums = u0.sum(axis=1)
        u0[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        return u0

    def _randomize_w0(self, hyperEdges: Union[np.array, None] = None) -> np.array:
        """
        Initialize affinity matrix w.

        Parameters
        ----------
        hyperEdges : ndarray
                     Array of length E, containing the sets of hyper-edges (as tuples).
        """

        w0 = self.prng.random_sample((self.D - 1, self.K))

        if hyperEdges is not None:
            ds = np.array(
                list(
                    set(np.arange(self.D - 1)).difference(
                        set([len(e) - 2 for e in hyperEdges])
                    )
                )
            )
            if len(ds) > 0:
                print("setting certain d in w to zero:", ds)
                w0 = 0.0

        return w0

    def _initialize_u(self, u0: np.array) -> np.array:
        """
        Initialize membership matrix u from input.
        """

        max_entry = np.max(u0)
        return u0 + max_entry * self.err * self.prng.random_sample(u0.shape)

    def _initialize_w(self, w0: np.array) -> np.array:
        """
        Initialize affinity matrix w from input.
        """

        max_entry = np.max(w0)
        return w0 + max_entry * self.err * self.prng.random_sample(w0.shape)

    def _initialize_psiOmega(self) -> None:
        """
        Initialize psi matrices psiOmega and psiBarOmega, i.e., psi(0)(Omega^d, k) and psi(0)(BarOmega^d, k).
        They have dimension DxK, and the first row refers to degree=1.
        """

        self.psiBarOmega = np.zeros((self.D, self.K))
        self.psiOmega = np.zeros((self.D, self.K))

        self._set_dummy_u0()
        u0 = np.copy(self.u0_dummy)
        u0_mean = np.mean(self.u0_dummy[self.non_isolates], axis=0)

        for k in range(self.K):
            Nk = np.count_nonzero(u0[:, k])
            self.psiOmega[0, k] = np.sum(u0[:, k])
            for d in range(1, self.D):
                self.psiOmega[d, k] = np.power(u0_mean[k], d + 1) * comb(Nk, d + 1)

    def _random_initial_update_U(self):
        """
        Initialize membership matrix u, and update PsiOmega and PsiBarOmega matrices.
        """

        self.u = np.copy(self.u0_dummy)
        self.u_old = np.copy(self.u0_dummy)

        for i in range(self.N):
            self._update_psiBarOmega(i)

            if i not in self.isolates:
                self.u[i] = self.u0_current_real_t0[i]
                self.u[i] /= self.u[i].sum()
                low_values_indices = self.u[i] < self.err_max  # values are too low
                self.u[i][low_values_indices] = 0.0  # and set to 0.
            else:
                self.u[i] = np.zeros(self.K)

            self._update_psiOmega(i)

            self.u_old[i] = np.copy(self.u[i])

    @gl.timeit("psiBarOmega")
    def _update_psiBarOmega(self, i: int, ks: Union[int, None] = None) -> bool:
        """
        Update psiBarOmega matrix.

        Parameters
        ----------
        i : int
            Row index.
        ks : int
             Column index.

        Returns
        -------
        success : bool
                  Flag to check whether the matrix psiBarOmega has all non-negative entries.
        """

        success = True
        if ks is None:
            self.psiBarOmega[0] = self.psiOmega[0] - self.u[i]
            for d in range(1, self.D):
                self.psiBarOmega[d] = (
                    self.psiOmega[d] - self.u[i] * self.psiBarOmega[d - 1]
                )
        else:
            self.psiBarOmega[0][ks] = self.psiOmega[0][ks] - self.u[i][ks]
            for d in range(1, self.D):
                self.psiBarOmega[d][ks] = (
                    self.psiOmega[d][ks] - self.u[i][ks] * self.psiBarOmega[d - 1][ks]
                )

        tmpMask = self.psiBarOmega < 0
        if np.sum(tmpMask) > 0:
            if (abs(self.psiBarOmega[tmpMask]) < 1e-3).all():
                self.psiBarOmega[tmpMask] = np.zeros_like(self.psiBarOmega[tmpMask])
        tmpMask = self.psiBarOmega < 0
        if np.sum(tmpMask) > 0:
            success = False

        return success

    @gl.timeit("psiOmega")
    def _update_psiOmega(self, i: int, ks: Union[int, None] = None) -> None:
        """
        Update psiOmega matrix.

        Parameters
        ----------
        i : int
            Row index.
        ks : int
             Column index.
        """

        if ks is None:
            self.psiOmega[0] = self.psiOmega[0] + (self.u[i] - self.u_old[i])
            assert np.allclose(self.psiOmega[0], self.u.sum(axis=0))
            for d in range(1, self.D):
                self.psiOmega[d] = (
                    self.psiOmega[d]
                    + (self.u[i] - self.u_old[i]) * self.psiBarOmega[d - 1]
                )
        else:
            self.psiOmega[0][ks] = self.psiOmega[0][ks] + (
                self.u[i][ks] - self.u_old[i][ks]
            )
            assert np.allclose(self.psiOmega[0], self.u.sum(axis=0))
            for d in range(1, self.D):
                self.psiOmega[d][ks] = (
                    self.psiOmega[d][ks]
                    + (self.u[i][ks] - self.u_old[i][ks]) * self.psiBarOmega[d - 1][ks]
                )

        tmpMask = self.psiOmega[d] < 0
        if np.sum(tmpMask) > 0:
            if (abs(self.psiOmega[d][tmpMask]) < 1e-3).all():
                self.psiOmega[d][tmpMask] = np.zeros_like(self.psiOmega[d][tmpMask])

        if self.it == 0:
            if np.sum(self.psiOmega < 0) > 0:
                tmpMask = self.psiOmega < 0
                tmpMask2 = np.logical_and(tmpMask, abs(self.psiOmega) < 1e-3)
                self.psiOmega[tmpMask2] = abs(self.psiOmega[tmpMask2])
                tmpMask = self.psiOmega < 0
                if tmpMask.sum() > 0:
                    print("psiOmega", self.psiOmega[tmpMask])
                    print(np.where(tmpMask == True))

                self.it += 1
                print("i=", i, self.Bsubs[i])

    @gl.timeit("rho")
    def _update_rho(self) -> None:
        """
        Update the rho matrix that represents the variational distribution used in the EM routine.
        """

        self.rho = self.w[self.HyeId2D - 2] * np.exp(
            np.dot(self.B_binarize.T, np.log(self.u + DEFAULT_EPS))
        )
        row_sums = np.sum(self.rho, axis=1)
        self.rho[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

    @gl.timeit("update_em")
    def _update_em(self) -> Tuple[float, float]:
        """
        Expectation-Maximization routine.
        """

        if not self.fix_w:
            d_w = self._update_W()
            self._update_rho()
        else:
            d_w = 0

        if not self.fix_communities:
            d_u = self._update_U()
            self._update_rho()
        else:
            d_u = 0

        return d_u, d_w

    @gl.timeit_cum("update_U")
    def _update_U(self) -> float:
        """
        Update membership matrix u. It is parallel for all the k of a node, but sequential in nodes (with random
        permutation of the nodes).
        """

        perm = self.prng.permutation(range(self.N))
        for i in perm:
            ks = np.where(self.u[i] > self.err_max)[0]
            if ks.shape[0] != 0:
                success = self._update_psiBarOmega(i, ks=ks)
                if success:
                    u_tmp = np.einsum(
                        "I,Ik->k",
                        self.B[i][self.Bsubs[i]],
                        self.rho[self.Bsubs[i]][:, ks],
                    )
                    u_tmp_den = self.gammaU + np.sum(
                        self.w[:, ks] * self.psiBarOmega[:-1, ks], axis=0
                    )  # sum over d

                    if self.constraintU:
                        low_values_indices = (u_tmp / u_tmp_den) < self.err_max
                        u_tmp[low_values_indices] = 0

                        lambda_i = self.enforce_constraintU(u_tmp, u_tmp_den)
                        self.u[i, ks] = u_tmp / (lambda_i + u_tmp_den)
                    else:
                        self.u[i, ks] = u_tmp / u_tmp_den

                    self.check_u(i, ks)

                    low_values_indices = self.u[i] < self.err_max  # values are too low
                    self.u[i][low_values_indices] = 0  # and set to 0.

                    high_values_indices = self.u[i] > 1e2  # values are too high
                    self.u[i][high_values_indices] = 1e2  # and set to 100.

                    self._update_psiOmega(i, ks=ks)

            self.u_old[i] = np.copy(self.u[i])

        dist_u = np.amax(abs(self.u - self.u_old))

        return dist_u

    @staticmethod
    def enforce_constraintU(num: np.array, den: float) -> float:
        """
        Return the lagrangian multiplier to enforce the constraint on the matrix u.

        Parameters
        ----------
        num : ndarray
              Numerator of the update of the membership matrix u.
        den : float
              Numerator of the update of the membership matrix u.

        Returns
        -------
        lambda_i : float
                   Lagrangian multiplier.
        """

        lambda_i_test = root(
            func_lagrange_multiplier, x0=np.array([0.1]), args=(num, den)
        )
        lambda_i = lambda_i_test.x

        return lambda_i

    def check_u(self, i: int, ks: Union[int, None]) -> None:
        """
        Check the updated value of u[i]
        """

        tmpMask = self.u[i, ks] < 0
        if np.sum(tmpMask) > 0:
            if abs(self.u[i, ks][tmpMask]).any() > 1e-01:
                self.u[i, ks] = abs(self.u[i, ks])
            else:
                print("WARNING!", i, self.u[i])

    @gl.timeit_cum("update_W")
    def _update_W(self,) -> float:
        """
        Update affinity matrix w.
        """

        for d in range(self.D - 1):
            self.w[d] = np.einsum(
                "I,Ik->k", self.data[self.HyD2eId[d]], self.rho[self.HyD2eId[d]]
            )
            Z = self.gammaW + self.psiOmega[d + 1]

            non_zeros = Z > 0
            self.w[d, non_zeros] /= Z[non_zeros]

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _check_for_convergence(
        self, it: int, loglik: float, coincide: int, convergence: bool
    ) -> Tuple[int, float, int, bool]:
        """
        Check for convergence by using the log-likelihood.

        Parameters
        ----------
        it : int
             Number of iteration.
        loglik : float
                 Log-likelihood value.
        coincide : int
                   Number of time the update of the log-likelihood respects the tolerance.
        convergence : bool
                      Flag for convergence.

        Returns
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

        if it % self.loglik_steps == 0:
            old_L = loglik
            loglik = self.__Likelihood()
            if abs(loglik - old_L) < self.convergence_tol:
                coincide += 1
            else:
                coincide = 0
        if coincide > self.decision:
            convergence = True
        it += 1

        return it, loglik, coincide, convergence

    @gl.timeit("Likelihood")
    def __Likelihood(self, EPS: int = 1e-300) -> float:
        """
        Compute the log-likelihood of the data.

        Parameters
        ----------
        EPS : int
              Random noise.

        Returns
        -------
        ll : float
            Log-likelihood value.
        """

        ll = -self.gammaW * np.sum(self.w) - self.gammaU * np.sum(self.u)

        tmp = self.w[self.HyeId2D - 2] * np.exp(
            np.dot(self.B_binarize.T, np.log(self.u + DEFAULT_EPS))
        )
        ll += np.sum(self.data * np.log(np.sum(tmp, axis=1) + EPS))

        ll -= np.sum(
            self.w[np.arange(self.D - 1)] * self.psiOmega[np.arange(1, self.D)]
        )

        if np.isnan(ll):
            print("Log-likelihood is NaN!!")
            return -DEFAULT_INF
        else:
            return ll

    def _update_optimal_parameters(self) -> None:
        """
        Update values of the parameters after convergence.
        """

        self.u_f = np.copy(self.u)
        self.w_f = np.copy(self.w)

    def _output_results(self) -> None:
        """
        Function to output the results.
        """

        outfile = self.out_folder + "theta" + self.end_file
        np.savez_compressed(
            outfile + ".npz",
            u=self.u_f,
            w=self.w_f,
            max_it=self.final_it,
            maxL=self.maxL,
            non_isolates=self.non_isolates,
        )
        print(f'\nInferred parameters saved in: {outfile + ".npz"}')
        print('To load: theta=np.load(filename), then e.g. theta["u"]')


def binarize_B(B: np.array) -> np.array:
    """
    Binarize the incidence matrix of dimension NxE.
    """

    B_binarize = np.zeros(B.shape).astype(bool)
    B_binarize[B > 0] = 1

    return B_binarize


def extract_indicesB(B: np.array) -> List[np.array]:
    """"
    Extract the list of length N containing the indices of non-zero hyperedges for every node.
    """

    N = B.shape[0]
    Bsubs = [np.where(B[i] > 0)[0] for i in range(N)]

    return Bsubs


def extract_indicesHy(hyperEdges: np.array) -> Tuple[List[List[int]], np.array]:
    """"
    Lists containing information about the hyperedges.

    Parameters
    ----------
    hyperEdges : ndarray
                 Array of length E, containing the sets of hyperedges (as tuples).

    Returns
    -------
    HyD2eId : list
              List of list containing the indices of hyperedges with a given degree.
    HyeId2D : ndarray
              Array of length E, containing the degree of each hyperedge.
    """

    HyeId2D = np.array([len(e) for e in hyperEdges])
    HyD2eId = [
        list(np.where(HyeId2D == d)[0]) for d in np.arange(2, np.max(HyeId2D + 1))
    ]

    return HyD2eId, HyeId2D


def calculate_u_SC(B: np.array, K: int, seed: int) -> np.array:
    """
    Calculate the membership with the Hypergraph Spectral Clustering.

    Parameters
    ----------
    B : ndarray
        Incidence matrix of dimension NxE.
    K : int
        Number of communities.
    seed : int
           Random seed.

    Returns
    -------
    Membership matrix.
    """

    sc_model = HySC(rseed=seed)
    return sc_model.fit(B, K=K)


def func_lagrange_multiplier(lambda_i: float, num: np.array, den: float) -> float:
    """
    Return the objective function to find the lagrangian multiplier to enforce the constraint on the matrix u.
    """

    f = num / (lambda_i + den)
    return np.sum(f) - 1


def plot_L(
    values: List[float],
    indices: Union[List[int], None] = None,
    k_i: int = 1,
    figsize: Tuple[int, int] = (4, 3),
    int_ticks: bool = False,
    xlab: str = "Iterations",
) -> None:
    """
    Function to plot the log-likelihood.
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if indices is None:
        ax.plot(values[k_i:])
    else:
        ax.plot(indices[k_i:], values[k_i:])
    ax.set_xlabel(xlab)
    ax.set_ylabel("Log-likelihood values")
    if int_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()

    plt.tight_layout()
    plt.show()
