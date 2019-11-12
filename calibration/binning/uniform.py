import numpy as np

from ..binning import general


class UniformBinning:
    """Partition probabilities into bins of uniform size."""

    def __init__(self, bins=10):
        self.bins = np.linspace(start=0, stop=1, num=bins+1)[1:-1]

    def split(self, parent, allprobs):
        indices = parent.indices

        # Do not split if no samples remaining
        if indices.size == 0:
            return

        # Increment axis along which samples are split
        split_axis = 0 if parent.is_root else parent.parent.split_axis + 1
        if split_axis + 1 >= allprobs.shape[-1]:
            return
        split_probs = allprobs[indices, split_axis]

        # Obtain indices of every bin
        split_nodes = np.digitize(split_probs, self.bins)

        # Update parent node
        parent.split_axis = split_axis
        parent.split_threshold = self.bins

        # Create child nodes
        for i in range(self.bins.size+1):
            split_indices = indices[split_nodes == i]

            if split_indices.size > 0:
                node = general.BinningNode(split_indices)
                node.split_index = i
                node.parent = parent

                self.split(node, allprobs)

        return self

    def __repr__(self):
        return "UniformBinning(bins=%r)" % (self.bins.size() - 1)
