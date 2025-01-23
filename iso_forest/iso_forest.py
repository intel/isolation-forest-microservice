# iso_forest.py: Base class for the isolation forest classifier. Handles nearly all the work of building and training a model, and includes functions for comparing quant and non models
# Copyright (c) 2025 Intel Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math
import statistics
import numpy as np
from sklearn.utils import (
    check_random_state)
from sklearn.tree._tree import DTYPE as tree_dtype
from sklearn.utils.validation import check_is_fitted, _num_samples
from sklearn.ensemble._bagging import BaggingRegressor
from scipy.sparse import issparse

from base_regression_tree import base_regression_tree

''' SUPPORTED SPLITS '''
RANDOM = "RANDOM"
QUANT = "QUANTIZED"
SUPP_SPLITS = (RANDOM, QUANT)

''' TYPES OF NODES '''
LEAF = "LEAF"
NODE = "NODE"

def c_n(x):
    return 2.0*(np.log(x-1.0)+0.5772156649) - (2.0*(x-1.0)/x)


def auto_opt_bin(data, res_array, mul_factor=1):
    cost, bin_width_compare = [], []
    range = max(data) - min(data)
    upper_stdev_bound = statistics.mean(
        data) + (mul_factor * statistics.stdev(data))
    lower_stdev_bound = statistics.mean(
        data) - (mul_factor * statistics.stdev(data))

    upper_bound_data = data[data <= upper_stdev_bound]
    lower_bound_data = data[data >= lower_stdev_bound]
    data_within_one_stdev = np.intersect1d(upper_bound_data, lower_bound_data)

    for resolution in res_array:
        bin_width, bin_width_count, diff_sub_count, diff_of_subseq_counts = [], [], [], []
        bin_step = (max(data_within_one_stdev) -
                    min(data_within_one_stdev)) * resolution
        range_bin_width = np.arange(min(data), max(data), bin_step)
        num_of_bin = math.floor(range / bin_step)

        counts, centers = np.histogram(data, bins=range_bin_width)
        bin_width.append(centers[1] - centers[0])
        bin_width_compare.append(bin_step)

        diff_of_subseq_counts = [(abs(counts[c] - counts[c-1]))
                                 for c in np.arange(1, len(counts), 1)]
        cost.append((max(diff_of_subseq_counts)) / bin_step)
    assert isinstance(cost, object)
    idx = cost.index(min(cost))
    optimal_bin_width = bin_width_compare[idx]
    assert isinstance(optimal_bin_width, object)
    bin_array_optimal = np.arange(min(data), max(data), optimal_bin_width)
    bin_counts_optimal, bin_centers_optimal = np.histogram(
        data, bin_array_optimal)
    num_bins_optimal = len(bin_centers_optimal)

    return (num_bins_optimal, optimal_bin_width, bin_array_optimal, bin_counts_optimal)


def auto_opt_bin_per_dim(X, start=0.01, end=0.1, step=0.005, quant_bin_per=1.0):
    assert quant_bin_per > 0.0 and quant_bin_per <= 1.0, "ERR | quant bin sel not valid"
    res_array = np.arange(start, end, step)
    bin_opt_num_of_bin_list = []
    bin_arrays_list = []
    bin_counts_avg_list = []
    for dimension in range(X.shape[1]):
        data = X[:, dimension]
        optimal_bin_info = auto_opt_bin(data, res_array)
        optimal_num_of_bin = optimal_bin_info[0]
        optimal_bin_width = optimal_bin_info[1]
        bin_array_optimal = optimal_bin_info[2]
        bin_counts_optimal = optimal_bin_info[3]

        bin_counts_avg = np.zeros(len(bin_counts_optimal)+1)
        for i in range(len(bin_counts_optimal)+1):
            if i == 0:
                bin_counts_avg[i] = bin_counts_optimal[i]/2
            elif i == len(bin_counts_optimal):
                bin_counts_avg[i] = bin_counts_optimal[i-1]/2
            else:
                bin_counts_avg[i] = (
                    bin_counts_optimal[i-1] + bin_counts_optimal[i]) / 2
        bin_opt_num_of_bin_list.append(optimal_num_of_bin)
        bin_counts_avg_list.append(bin_counts_avg)

        inverse_bin_counts_avg = np.negative(bin_counts_avg)
        bin_dimension = -math.ceil(len(inverse_bin_counts_avg) * quant_bin_per)
        select_idx = np.argpartition(inverse_bin_counts_avg, bin_dimension)[
            bin_dimension:]
        bin_array_optimal = bin_array_optimal[np.sort(select_idx)]

        bin_arrays_list.append(bin_array_optimal)
    return [bin_opt_num_of_bin_list, bin_arrays_list, bin_counts_avg_list]


class iso_forest(BaggingRegressor):
    def __init__(self,
                 n_estimators=100,	max_samples="auto", 	max_features=1.0,
                 random_state=None,	type_itree=False,	training_set=None,
                 sub_samp_size=256,	grow_full_tree=False,
                 quant_bin_sel=1.0):
        assert sub_samp_size is not None, "Sampling size is not declared"
        assert type_itree in SUPP_SPLITS, "iTree Type not supported"
        assert training_set is not None, "Training data must be passed @ initialization"
        # ARGS FOR Estimation Tree Regressor LIST
        self.split = type_itree
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = None
        self.sub_sample_size = sub_samp_size
        self.full_grow = grow_full_tree

        # For prediction/Modeling
        self.anomaly_th = None
        self.estimators = []
        self.estimator_features = []

        if self.split is QUANT:
            assert training_set is not None, "Quantized IF needs training set passed in @ init."
            self.dimension, dim_info = training_set.shape[1], dict()
            for dim in range(self.dimension):
                median = np.median(training_set[:, dim])
                std = np.std(training_set[:, dim])
                dim_info[dim] = [median, std]
            self.dimension_info = dim_info
            bin_info = auto_opt_bin_per_dim(
                training_set, quant_bin_per=quant_bin_sel)
            self.bin_num_list = bin_info[0]
            self.bin_arrays = bin_info[1]
            self.bin_counts = bin_info[2]

        else:
            self.dimension = training_set.shape[1]
            self.dimension_info = None
            self.bin_num_list = None
            self.bin_counts = None
            self.bin_arrays = None

    def _set_oob_score(self):
        raise NotImplementedError("Not implemented yet")

    def validate_data(self, X):
        return self._validate_data(X, accept_sparse=["csr"], dtype=tree_dtype)

    def get_max_depth_and_out_shape(self, X):
        if issparse(X):
            X.sort_indices()
        y = check_random_state(self.random_state).uniform(size=X.shape[0])
        max_depth = int(np.ceil(np.log2(max(self.sub_sample_size, 2))))
        max_depth = max_depth if self.full_grow is False else self.sub_sample_size - 1
        self.max_depth = max_depth
        return max_depth, y

    def apply_params_to_est(self, X):
        X = self.validate_data(X)
        md, y = self.get_max_depth_and_out_shape(X)
        estimators = self.get_tree_regr(X,
                                        self.max_features,
                                        self.random_state,
                                        self.n_estimators,
                                        max_depth=md)
        return estimators, y

    def get_tree_regr(self, X, m_s, random_state, n_estimators, max_depth=None):
        etr_list = [base_regression_tree(SUPP_SPLITS,
                                         max_features=m_s,
                                         splitter=self.split,
                                         max_depth=max_depth,
                                         dimension_training=self.dimension,
                                         dimension_info=self.dimension_info,
                                         bin_num_list=self.bin_num_list,
                                         bin_arrays=self.bin_arrays,
                                         bin_counts=self.bin_counts)
                    for tree in range(n_estimators)]
        return etr_list

    def fit(self, X):
        self.num_of_samples = len(X)
        self.estimators, y = self.apply_params_to_est(X)
        self.estimator_features = [np.array(
            [feature for feature in range(X.shape[1])]) for est in range(self.n_estimators)]
        self.estimators = [tree_est.fit(
            X, y, self.sub_sample_size) for tree_est in self.estimators]
        return self

    def get_quant_bin_size(self):
        if self.split is QUANT:
            quant_lengths = [len(dim) for dim in self.bin_arrays]
            avg = np.average(quant_lengths)
            std = np.std(quant_lengths)
            var = std / avg
            return avg, var
        else:
            return 0, 0

    def set_anomaly_threshold(self, anomaly):
        '''
        Note the following and that s is the anomaly threshold:
                if s is close to 1 	-> definite anomaly
                if s is @ 0.5 		-> entire sample does not have any distinct anomaly
                if s is close to 0 	-> safe, normal instance
        '''
        assert (anomaly > 0) and (
            anomaly < 1), f"anomaly threshold {anomaly} not in valid range, e.g. 0 <= s <= 1"
        self.anomaly_th = anomaly
        return self

    def set_leaf_factor(self):
        self.estimators = [tree_est.preprocess_leaf_factor()
                           for tree_est in self.estimators]
        self._std_factor = np.average(
            [tree_est._std for tree_est in self.estimators])
        return self

    def compute_score(self, X):
        tree_depth_scores = [tree.predict(X) for tree in self.estimators]
        tree_depth_ave = np.average(tree_depth_scores)
        score = 2 ** (-1.0 * (tree_depth_ave / c_n(self.sub_sample_size)))
        return score

    def gen_score_matrix(self, X):
        score = [self.compute_score(entry) for entry in X]
        return np.array(score)

    def get_spatial_metrics(self):
        depth_ave, leaf_ave, split_node_ave, total_node_ave = [], [], [], []
        for tree in self.estimators:
            ave_depth, num_leaves, total_nodes, num_split_node = tree.compute_spatial_metrics()
            depth_ave.append(ave_depth)
            leaf_ave.append(num_leaves)
            split_node_ave.append(num_split_node)
            total_node_ave.append(total_nodes)
        ave_depth_tree = np.average(depth_ave)
        ave_leaf_tree = np.average(leaf_ave)
        ave_split_node = np.average(split_node_ave)
        ave_tree_size = np.average(total_node_ave)
        return ave_depth_tree, ave_leaf_tree, ave_split_node, ave_tree_size

    def predict(self, X):
        check_is_fitted(self)
        assert self.anomaly_th is not None, "Anomaly threshold was never set"
        score_matrix = self.gen_score_matrix(X)
        is_nom = np.ones_like(score_matrix, dtype=int)
        is_nom[score_matrix > self.anomaly_th] = -1
        return is_nom, score_matrix
