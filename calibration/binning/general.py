import numpy as np
import anytree


class BinningTree:
    """General structure for binning samples by splitting their indices along
    an axis."""

    def __init__(self, alg):
        self.alg = alg
        self.fitted = False

    def fit(self, probs):
        """Fit binning tree to probabilities."""
        # expand one-dimensional probability vectors
        self.probs = probs if probs.ndim > 1 else np.stack(
            [probs, 1-probs], axis=-1)

        # define root node with all samples and split it with binning algorithm
        self.root = BinningNode(np.arange(self.probs.shape[0]))
        self.alg.split(self.root, self.probs)
        self.fitted = True

        return self

    @property
    def binnumbers(self):
        """Return array with bin numbers of each sample."""

        if not self.fitted:
            raise Exception("BinningTree fit needs to be called first")

        binnumbers = np.empty(self.probs.shape[0], dtype=np.int64)
        for idx, node in enumerate(
                anytree.PreOrderIter(self.root,
                                     filter_=BinningNode.leaf_filter)):
            binnumbers[node.indices] = idx

        return binnumbers

    @property
    def bins(self):
        """Return list with sample indices of each bin."""

        if not self.fitted:
            raise Exception("BinningTree fit needs to be called first")

        bins = []
        for node in anytree.PreOrderIter(self.root,
                                         filter_=BinningNode.leaf_filter):
            bins.append(node.indices)

        return bins

    def bin_data(self, data=None):
        """
        Return list with `data` split up in the bins.

        The data `data` should be an array of size `(N, ...)` where `N`
        is the number of data points to which the binning tree was fit.

        If `data` is `None` (the default), `data` is set to the probabilities
        to which the binning tree was fit.
        """
        if data is None:
            data = self.probs

        return [data[x] for x in self.bins]


class BinningNode(anytree.NodeMixin):
    """Node for saving indices and boundaries."""

    def __init__(self, indices):
        super(BinningNode, self).__init__()
        self.indices = indices

    def _post_attach(self, parent):
        """Clear parent node."""
        try:
            parent.indices = None
        except AttributeError:
            pass

    @staticmethod
    def leaf_filter(x):
        if x.is_leaf:
            try:
                return x.indices.size > 0
            except AttributeError:
                return False

        return False
