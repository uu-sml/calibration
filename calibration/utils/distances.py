import numpy as np


# functional implementation

def l1distance(x, y):
    return np.linalg.norm(np.subtract(x, y), ord=1, axis=-1)


def tvdistance(x, y):
    return np.linalg.norm(np.subtract(x, y), ord=1, axis=-1) / 2


def l2distance(x, y):
    return np.linalg.norm(np.subtract(x, y), ord=2, axis=-1)


# state-full implementation

class L1Distance:

    def __call__(self, x, y):
        return l1distance(x, y)

    def __repr__(self):
        return "L1Distance"


class TVDistance:

    def __call__(self, x, y):
        return tvdistance(x, y)

    def __repr__(self):
        return "TVDistance"


class L2Distance:

    def __call__(self, x, y):
        return l2distance(x, y)

    def __repr__(self):
        return "L2Distance"
