"""Module for the FPBoost class."""

import numbers
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils._param_validation import Interval, StrOptions
from sksurv.base import SurvivalAnalysisMixin
from sksurv.functions import StepFunction
from sksurv.util import check_array_survival
from torch.autograd import Variable

__all__ = ["FPBoost"]


class FPBoost(SurvivalAnalysisMixin, BaseEstimator):
    r"""Gradient boosting for survival data based on the composition of fully parametric
    distributions. The model is trained by minimizing the negative log-likelihood with an
    optional ElasticNet regularization term. The model is an ensemble of base learners, where each
    base learner is either a Weibull or a log-logistic distribution.

    References
    ----------
    Archetti, A., Lomurno, E., Piccinotti, D. & Matteucci, M. FPBoost: Fully Parametric Gradient 
        Boosting for Survival Analysis. arXiv preprint arXiv:2409.13363 (2024). 
        https://arxiv.org/abs/2409.13363

    Parameters
    ----------
    weibull_heads : int, default=2
        Number of Weibull heads in the ensemble.
    loglogistic_heads : int, default=2
        Number of log-logistic heads in the ensemble.
    n_estimators : int, default=100
        Number of boosting iterations to perform. During each iteration, a base
        learner is trained to minimize the negative log-likelihood of the ensemble predictions
        for each parameter of the fully parametric distributions.
    max_depth : int, default=1
        Maximum depth of the individual trees in the ensemble.
    learning_rate : float, default=0.1
        Shrinks the contribution of each base learner.
    alpha : float, default=0.0
        Strength of the ElasticNet regularization. The penalty term is a combination of
        L1 and L2 regularization. A value of `alpha=0` corresponds to no regularization.
    l1_ratio : float, default=0.5
        Ratio of L1 regularization in the ElasticNet penalty. A value of `l1_ratio=1`
        corresponds to L1 regularization, `l1_ratio=0` corresponds to L2 regularization.
    uniform_heads : bool, default=False
        If `True`, the weights of the heads are fixed to be uniform. Otherwise,
        the weights are learned by the model.
    heads_activation : {'relu', 'sigmoid'}, default='relu'
        Activation function for the weights of the heads. If 'relu', the weights
        are constrained to be non-negative. If 'sigmoid', the weights are constrained to be
        in the range `[0, 1]`.
    random_state : int, optional
        The seed of the pseudo random number generator to use when training the model.
    """

    _parameter_constraints = {
        "weibull_heads": [Interval(numbers.Integral, 0, None, closed="left")],
        "loglogistic_heads": [Interval(numbers.Integral, 0, None, closed="left")],
        "n_estimators": [Interval(numbers.Integral, 1, None, closed="left")],
        "max_depth": [Interval(numbers.Integral, 1, None, closed="left")],
        "learning_rate": [Interval(numbers.Real, 0, None, closed="neither")],
        "alpha": [Interval(numbers.Real, 0, None, closed="left")],
        "l1_ratio": [Interval(numbers.Real, 0, 1, closed="both")],
        "uniform_heads": [bool],
        "heads_activation": [StrOptions({"relu", "sigmoid"})],
    }

    def __init__(
        self,
        weibull_heads: int = 2,
        loglogistic_heads: int = 2,
        n_estimators: int = 100,
        max_depth: int = 1,
        learning_rate: float = 0.1,
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        uniform_heads: bool = False,
        heads_activation: str = "relu",
        random_state: Optional[int] = None,
    ):
        self.weibull_heads = weibull_heads
        self.loglogistic_heads = loglogistic_heads

        self.heads = weibull_heads + loglogistic_heads
        if self.heads == 0:
            self.weibull_heads = 1
            self.heads = 1

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.uniform_heads = uniform_heads
        self.heads_activation = heads_activation
        self.random_state = random_state

        self._base_timeline = np.linspace(0, 1, 100)

    def _init_state(self):
        seed = np.random.default_rng(self.random_state)
        self.init_eta_ = seed.random(self.heads) + 0.5
        self.eta_heads_ = [[] for _ in range(self.heads)]
        self.init_k_ = seed.random(self.heads) * 2
        self.k_heads_ = [[] for _ in range(self.heads)]
        self.init_w_ = seed.random(self.heads)
        self.w_heads_ = [[] for _ in range(self.heads)]

    def _predict_etas(self, X):
        output = np.zeros((len(X), self.heads)) + self.init_eta_.reshape((1, -1))
        for i, regs in enumerate(self.eta_heads_):
            if len(regs) == 0:
                continue
            preds = np.concatenate([reg.predict(X).reshape((-1, 1)) for reg in regs], axis=1)
            output[:, i] += self.learning_rate * np.sum(preds, axis=1)
        return output

    def _predict_ks(self, X):
        output = np.ones((len(X), self.heads)) * self.init_k_.reshape((1, -1))
        for i, regs in enumerate(self.k_heads_):
            if len(regs) == 0:
                continue
            preds = np.concatenate([reg.predict(X).reshape((-1, 1)) for reg in regs], axis=1)
            output[:, i] += self.learning_rate * np.sum(preds, axis=1)
        return output

    def _predict_ws(self, X):
        if self.uniform_heads:
            return np.ones((len(X), self.heads)) / self.heads
        output = np.ones((len(X), self.heads)) * self.init_w_.reshape((1, -1))
        for i, regs in enumerate(self.w_heads_):
            if len(regs) == 0:
                continue
            preds = np.concatenate([reg.predict(X).reshape((-1, 1)) for reg in regs], axis=1)
            output[:, i] += self.learning_rate * np.sum(preds, axis=1)
        return output

    def _predict_params(self, X):
        etas = self._predict_etas(X).reshape((-1, self.heads, 1))
        ks = self._predict_ks(X).reshape((-1, self.heads, 1))
        ws = self._predict_ws(X).reshape((-1, self.heads, 1))
        return np.concatenate([etas, ks, ws], -1)

    def _weibull_hazard(self, eta, k, times):
        return k * eta * times ** (k - 1)

    def _weibull_cum_hazard(self, eta, k, times):
        return eta * times**k

    def _loglogistic_hazard(self, eta, k, times):
        return eta * k * times ** (k - 1) / (1 + eta * times**k)

    def _loglogistic_cum_hazard(self, eta, k, times):
        if torch.is_tensor(times):
            return torch.log1p(eta * times**k)
        return np.log1p(eta * times**k)

    def _get_neg_grads(self, params, events, times):
        params_torch = Variable(torch.tensor(params).float(), requires_grad=True)

        etas = F.relu(params_torch[:, :, 0])
        ks = F.relu(params_torch[:, :, 1])
        if self.heads_activation == "relu":
            ws = F.relu(params_torch[:, :, 2])
        else:
            ws = F.softmax(params_torch[:, :, 2], dim=1)

        hazard = torch.zeros(len(times))
        cum_hazard = torch.zeros(len(times))

        if self.weibull_heads > 0:
            weibull_hazard = self._weibull_hazard(
                etas[:, : self.weibull_heads], ks[:, : self.weibull_heads], times
            )
            weibull_cum_hazard = self._weibull_cum_hazard(
                etas[:, : self.weibull_heads], ks[:, : self.weibull_heads], times
            )
            hazard += (weibull_hazard * ws[:, : self.weibull_heads]).sum(dim=1)
            cum_hazard += (weibull_cum_hazard * ws[:, : self.weibull_heads]).sum(dim=1)

        if self.loglogistic_heads > 0:
            loglogistic_hazard = self._loglogistic_hazard(
                etas[:, self.weibull_heads :], ks[:, self.weibull_heads :], times
            )
            loglogistic_cum_hazard = self._loglogistic_cum_hazard(
                etas[:, self.weibull_heads :], ks[:, self.weibull_heads :], times
            )
            hazard += (loglogistic_hazard * ws[:, self.weibull_heads :]).sum(dim=1)
            cum_hazard += (loglogistic_cum_hazard * ws[:, self.weibull_heads :]).sum(dim=1)

        log_likelihood = (events * torch.log(hazard) - cum_hazard).mean()
        l1_reg = torch.abs(params_torch).mean()
        l2_reg = (params_torch**2).mean()
        elastic_net_reg = self.l1_ratio * l1_reg + (1 - self.l1_ratio) * l2_reg
        loss = -log_likelihood + self.alpha * elastic_net_reg

        loss.backward()
        grad = params_torch.grad.numpy()
        grad[np.isnan(grad)] = 0.0
        return -(grad / np.abs(grad).max())

    def _fit_base_learner(self, X, y) -> DecisionTreeRegressor:
        reg = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
        reg.fit(X, y)
        return reg

    def _fit(self, X, events, times) -> None:
        events = torch.tensor(events.copy()).float().reshape((-1,))
        times = torch.tensor(times.copy()).float().reshape((-1, 1))

        for _ in range(self.n_estimators):
            params = self._predict_params(X)

            neg_grads = self._get_neg_grads(params, events, times)
            eta_grads = neg_grads[:, :, 0]
            k_grads = neg_grads[:, :, 1]
            w_grads = neg_grads[:, :, 2]

            for i in range(self.heads):
                self.eta_heads_[i].append(self._fit_base_learner(X, eta_grads[:, i]))
                self.k_heads_[i].append(self._fit_base_learner(X, k_grads[:, i]))
                if not self.uniform_heads:
                    self.w_heads_[i].append(self._fit_base_learner(X, w_grads[:, i]))

    def fit(self, X, y) -> "FPBoost":
        """Fit the model to the training data.

        Parameters
        ----------
        X : np.array
            Input data of shape `(n_samples, n_features)`.
        y : np.array
            Structured array of shape `(n_samples,)` containing the `event` and `time` fields.

        Returns
        -------
        FPBoost
            The fitted model.
        """
        self._validate_params()
        X = self._validate_data(X)
        events, times = check_array_survival(X, y)
        self.max_time_ = times.max()
        times = times / self.max_time_
        self.unique_times_ = np.unique(times)
        self._init_state()
        self._fit(X, events, times)
        return self

    def predict(self, X):
        """Predict the negative mean time to event for the input data.

        Parameters
        ----------
        X : np.array
            Input data of shape `(n_samples, n_features)`.

        Returns
        -------
        np.array
            The predicted negative mean time to event.
        """
        X = self._validate_data(X, reset=False)
        cum_hazard = self._predict_cumulative_hazard(X, self._base_timeline)
        survival = np.exp(-cum_hazard)
        mean_time = survival.sum(axis=1) / len(self._base_timeline)
        return -mean_time

    def _predict_cumulative_hazard(self, X, times):
        check_is_fitted(self, "unique_times_")

        params = torch.tensor(self._predict_params(X)).float()

        etas = F.relu(params[:, :, 0]).numpy().reshape((-1, self.heads, 1))
        ks = F.relu(params[:, :, 1]).numpy().reshape((-1, self.heads, 1))
        if self.heads_activation == "relu":
            ws = F.relu(params[:, :, 2]).numpy().reshape((-1, self.heads, 1))
        else:
            ws = F.softmax(params[:, :, 2], dim=1).numpy().reshape((-1, self.heads, 1))

        cum_hazard = np.zeros((len(X), len(times)))

        if self.weibull_heads > 0:
            weibull_cum_hazard = self._weibull_cum_hazard(
                etas[:, : self.weibull_heads], ks[:, : self.weibull_heads], times
            )
            cum_hazard += (weibull_cum_hazard * ws[:, : self.weibull_heads]).sum(axis=1)

        if self.loglogistic_heads > 0:
            loglogistic_cum_hazard = self._loglogistic_cum_hazard(
                etas[:, self.weibull_heads :], ks[:, self.weibull_heads :], times
            )
            cum_hazard += (loglogistic_cum_hazard * ws[:, self.weibull_heads :]).sum(axis=1)
        return cum_hazard

    def predict_cumulative_hazard_function(self, X, return_array=False):
        """Predict the cumulative hazard function for the input data.

        Parameters
        ----------
        X : np.array
            Input data of shape `(n_samples, n_features)`.
        return_array : bool, default=False
            If `True`, the output is a numpy array. Otherwise, the output is a list
            of `StepFunction` objects, which can be called to evaluate the cumulative hazard
            function at specific times.

        Returns
        -------
        Union[list[StepFunction], np.array]
            The predicted cumulative hazard function.
        """
        times = self.unique_times_ if return_array else self._base_timeline
        cum_hazard = self._predict_cumulative_hazard(X, times)
        if return_array:
            return cum_hazard
        times = self.max_time_ * times
        return np.array([StepFunction(times, cum_hazard[i]) for i in range(len(X))])

    def predict_survival_function(self, X, return_array=False):
        """Predict the survival function for the input data.

        Parameters
        ----------
        X : np.array
            Input data of shape `(n_samples, n_features)`.
        return_array : bool, default=False
            If `True`, the output is a numpy array. Otherwise, the output is a list
            of `StepFunction` objects, which can be called to evaluate the survival function at
            specific times.

        Returns
        -------
        Union[list[StepFunction], np.array]
            The predicted survival function.
        """
        times = self.unique_times_ if return_array else self._base_timeline
        cum_hazard = self._predict_cumulative_hazard(X, times)
        survival = np.exp(-cum_hazard)
        if return_array:
            return survival
        times = self.max_time_ * times
        return np.array([StepFunction(times, survival[i]) for i in range(len(X))])
