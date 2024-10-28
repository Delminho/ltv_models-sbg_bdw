from abc import ABC, abstractmethod
from scipy.special import gammaln
from scipy.optimize import minimize
import numpy as np


class LTVModel(ABC):
    def __init__(self, optimization_method, initial_params):
        self.optimization_method = optimization_method
        self.initial_params = initial_params

    @abstractmethod
    def survivor(self, t, params, probabilities=None):
        """Survivor function S"""
        pass

    def predicted_survival(self, periods, params):
        """Generate list of predicted survival probability, i.e. percentage of customers retained of length periods.
        Function 1 in the paper"""
        s = [self.survivor(i, params) for i in range(periods)]
        return s

    def survival_rates(self, data):
        return [1 - data[0]] + [data[i - 1] - data[i] for i in range(1, len(data))]

    @abstractmethod
    def generate_probabilities(self, periods, params):
        """Generate list of churn probabilites of length in periods."""
        pass

    def log_likelihood(self, params, data, survivors=None):
        """Function to maximize to obtain best parameters"""
        if (np.array(params) <= 0).any():
            return -1000
        if survivors is None:
            survivors = self.survival_rates(data)

        probabilities = self.generate_probabilities(len(data), params)
        final_survivor_likelihood = self.survivor(len(data) - 1, params)

        return sum(
            [s * np.log(probabilities[t]) for t, s in enumerate(survivors)]
        ) + data[-1] * np.log(final_survivor_likelihood)

    def log_likelihood_multi_cohort(self, params, data):
        """Function to maximize to obtain ideal alpha and beta parameters using data across multiple (contiguous) cohorts.
        `data` must be a list of cohorts each with an absolute number per observed time unit."""
        if (np.array(params) <= 0).any():
            return -1000

        probabilities = self.generate_probabilities(len(data[0]), params)
        cohorts = len(data)
        total = 0

        for i, cohort in enumerate(data):
            total += sum(
                [
                    (cohort[j] - cohort[j + 1]) * np.log(probabilities[j])
                    for j in range(len(cohort) - 1)
                ]
            )
            total += cohort[-1] * np.log(self.survivor(cohorts - i - 1, params))
        return total

    def optimize(self, data, is_multi_cohort):
        """Search for best parameters"""
        if is_multi_cohort:
            func = lambda x: -self.log_likelihood_multi_cohort(x, data)
        else:
            survivors = self.survival_rates(data)
            func = lambda x: -self.log_likelihood(x, data, survivors)

        res = minimize(
            func,
            self.initial_params or np.random.rand(len(self.param_names)),
            bounds=self.bounds,
            method=self.optimization_method
        )

        if np.isclose(res.x, self.bounds[:, 0]).any() or np.isclose(res.x, self.bounds[:, 1]).any():
            # Parameters are very close to lower or upper bound meaning the optimization has failed
            raise Exception('Optimization Failed: Found parameters are inappropriate.')

        return res

    def preprocess_data(self, data):
        """Preprocessing data for model"""
        if hasattr(data[0], '__iter__'): # is 2D array
            if len(data) == 1:
                data = data[0] # 2D array -> 1D if it contains only 1 array
            else:
                return data # no processing for 2D array 

        # 1D array data needs to be in format [0.8, 0.65, 0.53,...]
        if data[0] > 1:
            data = (np.array(data) / data[0])[1:]
        elif data[0] == 1:
            data = data[1:]

        return data

    def fit(self, data, periods):
        """
        Fits the LTV model to the provided data.

        Example for fitting 1D data: .fit([0.8, 0.65, 0.53, 0.46], 52)
        Example for multiple cohorts: .fit([
                                            [733, 379, 282, 225],
                                            [519, 286, 194],
                                            [557, 292]
                                          ], 52)

        Args:
            data (list or array-like): The input data for fitting the model.
                                       This can be data for one or multiple cohorts.
            periods (int): The number of periods to forecast the retention curve.

        Returns:
            dict: A dictionary containing the optimized model parameters,
                  retention curve, and the loss value.
        """
        data = self.preprocess_data(data)

        is_multi_cohort = True if hasattr(data[0], "__iter__") else False

        res = self.optimize(data, is_multi_cohort)
        if res.status != 0:
            raise Exception("Optimization Failed:", res.message)

        result_dict = {name: value for name, value in zip(self.param_names, res.x)}
        result_dict |= {
            "retention_curve": [1] + self.predicted_survival(periods, res.x),
            "loss": res.fun,
        }

        return result_dict


class SBGModel(LTVModel):
    def __init__(self, optimization_method="nelder-mead", initial_params=None):
        super().__init__(
            optimization_method=optimization_method, initial_params=initial_params
        )
        self.param_names = ["alpha", "beta"]
        self.bounds = np.array([[0.0001, 10000], [0.0001, 10000]])

    def survivor(self, t, params, probabilities=None):
        """Survivor function S"""
        probabilities = probabilities or self.generate_probabilities(t + 1, params)
        s = 1 - probabilities[0]
        for i in range(1, t + 1):
            s = s - probabilities[i]
        return s

    def generate_probabilities(self, periods, params):
        """Generate list of churn probabilites of length in periods."""
        alpha, beta = params
        p = [alpha / (alpha + beta)]
        for t in range(1, periods):
            p.append((beta + t - 1) / (alpha + beta + t) * p[t - 1])
        return p


class BDWModel(LTVModel):
    def __init__(self, optimization_method="nelder-mead", initial_params=None):
        super().__init__(
            optimization_method=optimization_method, initial_params=initial_params
        )
        self.param_names = ["alpha", "beta", "c"]
        self.bounds = np.array([[0.0001, 10000], [0.0001, 10000], [0.0001, 3]])

    def survivor(self, t, params, probabilities=None):
        """Survivor function S"""
        alpha, beta, c = params
        return np.exp(
            gammaln(alpha + beta) + gammaln(beta + (t+1)**c)
            - gammaln(beta) - gammaln(alpha + beta + (t+1)**c)
        )

    def generate_probabilities(self, periods, params):
        """Generate list of churn probabilites of length in periods."""
        survivor_rates = self.predicted_survival(periods, params)
        return self.survival_rates(survivor_rates)
