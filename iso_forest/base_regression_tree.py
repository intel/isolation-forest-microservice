# base_regression_tree.py: Contains the functions for the underlying regression tree. Mostly used by iso_forest.py to build the underlying regression tree for the isolation forest classifier.
# Also contains functions for metric computation for the classifier
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
import numpy as np

import random

from tree_node import tree_node


def c_n(x):
    return 2.0*(np.log(x-1.0)+0.5772156649) - (2.0*(x-1.0)/x)


class base_regression_tree():
    def __init__(self,
                 SUPP_SPLITS, max_features=None, splitter=None, random_state=None,
                 max_depth=None, min_samples_for_split=2,
                 dimension_training=None, dimension_info=None,
                 bin_num_list=None, bin_arrays=None, bin_counts=None):
        self.splitter = splitter if splitter in SUPP_SPLITS else None
        assert self.splitter is not None, f"Split value ({splitter}), not supported \
											Supported split types | {SUPP_SPLITS}"
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_for_split = min_samples_for_split
        self.dim = dimension_training
        self.bin_number_list = bin_num_list
        self.bin_array = bin_arrays
        self.bin_count = bin_counts
        self.tree = None
        self.median_std, self.std_factor = None, None

    def traverse_and_print_the_tree(self, root):
        tab_spacer = root.curr_depth * ' '
        print(f"{tab_spacer}CURR DEPTH | {root.curr_depth}")
        print(f"{tab_spacer}Decide Feat | {root.feature_decision}")
        print(f"{tab_spacer}Decide Val | {root.feature_val}")
        print(f"{tab_spacer}Node ID | {root.id}")
        root = root._children_factor()
        assert len(root.children) < 3, "Children, too many!!!"
        if len(root.children) > 0:
            for root_i in root.children:
                self.traverse_and_print_the_tree(root_i)

    def get_depths_of_tree(self, root, depth_list):
        root_childs = root._children_factor()
        assert len(root_childs) < 3, f"Children, Too Many!! | {root.children}"
        if len(root_childs) > 0:
            for root_i in root_childs:
                self.get_depths_of_tree(root_i, depth_list)
        else:
            depth_list.append(root.curr_depth)
        return depth_list

    def get_total_nodes(self, root, node_count, leaf_count):
        root_childs = root._children_factor()
        assert len(root_childs) < 3, f"Children, Too Many!! | {root.children}"
        if len(root_childs) > 0:
            for root_i in root_childs:
                node_count += 1
                node_count, leaf_count = self.get_total_nodes(
                    root_i, node_count, leaf_count)
        else:
            leaf_count += 1
        return node_count, leaf_count

    def compute_spatial_metrics(self):
        list_of_leaf_depths = self.get_depths_of_tree(self.tree, [])
        node_count, leaf_count = self.get_total_nodes(self.tree, 0, 0)
        total_nodes = leaf_count + node_count
        avg_depth = np.average(list_of_leaf_depths)
        return avg_depth, leaf_count, total_nodes, node_count

    def compute_std_depth(self):
        list_of_leaf_depths = self.get_depths_of_tree(self.tree, [])
        std_depth = np.std(list_of_leaf_depths)
        return std_depth

    def get_depth(self, sample, node, depth):
        if node.id == "LEAF":
            if node.SIZE <= 1:
                return depth
            else:
                return depth + c_n(node.SIZE)
        else:
            sample_val = sample[node.feature_decision]
            if sample_val < node.feature_val:
                return self.get_depth(sample, node.left, depth+1)
            else:
                return self.get_depth(sample, node.right, depth+1)

    def predict(self, sample):
        return self.get_depth(sample, self.tree, 0)

    def grow_the_tree(self, cd, depth, X, bin_array, grow=None):
        feat_idx = random.choice(np.arange(X.shape[1]))
        random.seed(None)
        if grow == "RANDOM":
            if (cd >= depth) or (len(X) <= 1):
                return tree_node(curr_depth=cd, node_size=X.shape[0])
            else:
                first = X.min(axis=0)
                prob = random.uniform(0, 1)
                second = X.max(axis=0) * prob
                third = -1 * X.min(axis=0)
                third = third * prob
                split_val = first + second + third
                split_val = split_val[feat_idx]
                f_dec, f_val = feat_idx, split_val
                left_X, right_X = X[X[:, feat_idx] <
                                    split_val], X[X[:, feat_idx] >= split_val]

                return tree_node(curr_depth=cd+1,
                                 feature_val=f_val,
                                 feature_decision=f_dec,
                                 id="NODE",
                                 left=self.grow_the_tree(
                                     cd+1, depth, left_X, bin_array, grow),
                                 right=self.grow_the_tree(cd+1, depth, right_X, bin_array, grow))

        elif grow == "QUANTIZED":
            if (cd >= depth) or (len(X) <= 1):
                return tree_node(curr_depth=cd, node_size=X.shape[0])
            trunc_bin_arr = bin_array[feat_idx][(bin_array[feat_idx] <= X.max(
                axis=0)[feat_idx]) & (bin_array[feat_idx] >= X.min(axis=0)[feat_idx])]
            if (trunc_bin_arr.size <= 0):
                return tree_node(curr_depth=cd, node_size=X.shape[0])
            else:
                split_val = trunc_bin_arr[random.randint(
                    0, len(trunc_bin_arr)-1)]
                f_dec, f_val = feat_idx, split_val
                left_X, right_X = X[X[:, feat_idx] <
                                    split_val], X[X[:, feat_idx] >= split_val]
                return tree_node(curr_depth=cd+1,
                                 feature_val=f_val,
                                 feature_decision=f_dec,
                                 id="NODE",
                                 left=self.grow_the_tree(
                                     cd+1, depth, left_X, bin_array, grow),
                                 right=self.grow_the_tree(cd+1, depth, right_X, bin_array, grow))
        else:
            assert False

    def preprocess_leaf_factor(self):
        list_of_leaf_depths = self.get_depths_of_tree(self.tree, [])
        self._middle_perc = np.percentile(list_of_leaf_depths, 50.0)
        self._std = np.std(list_of_leaf_depths)
        self._linear_est_std_slope = -1.0 / self._std
        self._linear_est_offset = self._middle_perc / self._std
        return self

    def fit(self, X, y, sample):
        X_idx = random.sample(range(len(X)), sample)
        X_sample = X[X_idx]
        self.tree = self.grow_the_tree(
            0, self.max_depth, X_sample, self.bin_array, grow=self.splitter)
        return self
