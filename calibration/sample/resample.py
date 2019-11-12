import numpy as np


def resample_stats(stats, *data, n=1000):
    """
    Resample data set `data` `n` times and evaluate statistic `stats`.
    """
    N = data[0].shape[0]

    for i in range(1, len(data)):
        if data[i].shape[0] != N:
            raise ValueError(
                'Expected batch_size ({}) to match batch_size ({}).'
                .format(data[i].shape[0], N))

    out = []
    for _ in range(n):
        # Resample
        indices = np.random.randint(0, N, size=(N,))

        # Compute statistics
        out.append(stats(*(d[indices] for d in data)))

    return out


class ResampleStats:

    def __init__(self, stats, n=1000):
        self.stats = stats
        self.n = n

    def __call__(self, *data):
        return resample_stats(self.stats, *data, n=self.n)

    def __repr__(self):
        return "ResampleStats(stats=%r, n=%r)" % (self.stats, self.n)
