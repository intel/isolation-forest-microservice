# tree_node.py: Holds the object definition for a single node on the regression tree
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
class tree_node(object):
    def __init__(self,
                 curr_depth=None, feature_val=None, feature_decision=None,
                 id="LEAF", left=None, right=None, node_size=None):
        self.curr_depth = curr_depth
        self.SIZE = node_size
        self.feature_val = feature_val
        self.feature_decision = feature_decision
        self.right = right
        self.left = left
        self.id = id

    def _children_factor(self):
        children = []
        if self.right is not None:
            children.append(self.right)
        if self.left is not None:
            children.append(self.left)
        return children