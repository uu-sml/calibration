import numpy as np

from . import general


class DataDependentBinning:
    """Binning tree that minimizes number of samples per bin by
    splitting along the axis with highest variance."""

    def __init__(self, min_size=100, threshold="mean"):
        super(DataDependentBinning, self).__init__()

        # Minimum size of bins that are split
        self.min_size = min_size

        # Define threshold for splitting
        if threshold == "mean":
            self.threshold = np.mean
        elif threshold == "median":
            self.threshold = np.median
        elif callable(threshold):
            self.threshold = threshold
        else:
            raise ValueError(
                "'threshold' must be 'mean', 'median', or a function")

    def split(self, parent, allprobs):
        indices = parent.indices
        n = indices.size
        probs = allprobs[indices]

        # Do not split if the number of samples is low
        if n < self.min_size:
            return

        # Split along axis with highest variance
        split_axis = np.argmax(np.var(probs, axis=0))
        split_probs = probs[:, split_axis]
        split_threshold = self.threshold(split_probs)

        # Do not split if all samples end up in one node
        low_probs = split_probs < split_threshold
        number_low_probs = np.sum(low_probs)
        if number_low_probs == 0 or number_low_probs == n:
            return

        # Accept split and create children nodes
        parent.split_axis = split_axis
        parent.split_threshold = split_threshold

        low_node = general.BinningNode(indices[low_probs])
        low_node.split_index = 0
        low_node.parent = parent
        self.split(low_node, allprobs)

        high_node = general.BinningNode(indices[~low_probs])
        high_node.split_index = 1
        high_node.parent = parent
        self.split(high_node, allprobs)

        return parent

    def __repr__(self):
        return "DataDependentBinning(min_size=%r, threshold=%r)" \
            % (self.min_size, self.threshold)
