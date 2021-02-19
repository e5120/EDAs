import numpy as np
from scipy.stats import entropy


def idx2one_hot(x, max_size):
    """
    The value or index is converted to one-hot expression.

    Parameters
    ----------
    x : numpy.ndarray
        An individual or a population which have values or indexes.
    max_size : int
        Maximum cardinality of the input.

    Returns
    -------
    numpy.ndarray
        One-hot expression of x.
    """
    assert max_size > 0, \
        "max_size({}) must be non-negative integer value".format(max_size)
    return np.identity(max_size)[x]


def conditional_entropy(node, parent, base=np.e, eps=1e-8):
    """
    Estimate the conditional entropy of the node under the set of parent nodes.

    Parameters
    ----------
    node : numpy.ndarray
        Node.
    parent : numpy.ndarray
        A set of parent nodes of the node.
    base : int, default np.e
        Base.
    eps : float, default 1e-8
        A small value that prevents the value from being zero when computing in logarithms.

    Returns
    -------
    float
        Conditional entropy.
    """
    assert node.shape[0] == parent.shape[0] and len(parent.shape) == 2
    n = node.shape[0]
    if parent.shape[1] == 0:
        _, n_counts = np.unique(node, return_counts=True)
        cond_entropy = entropy(n_counts / n, base=base)
    else:
        _, p_counts = np.unique(parent, return_counts=True, axis=0)
        _, n_p_counts = np.unique(np.c_[node, parent], return_counts=True, axis=0)
        p_p = p_counts / n
        p_n_p = n_p_counts / n
        h_p = -np.sum(p_p * np.log(p_p + eps) / np.log(base))
        h_n_p = -np.sum(p_n_p * np.log(p_n_p + eps) / np.log(base))
        cond_entropy = h_n_p - h_p
    return cond_entropy


def estimate_cpt(node, parent, base=2, eps=1e-20):
    """
    Estimate the conditional probability table of a node.

    Parameters
    ----------
    node : numpy.ndarray
        Node.
    parent : numpy.ndarray
        A set of parent nodes of the node.
    base : int, default 2
        Base.
    eps : float, default 1e-20
        A small value that prevents the value of denominator from being zero when computing in division.

    Returns
    -------
    numpy.ndarray
        Conditional probability table.
    """
    assert node.shape[0] == parent.shape[0] and len(parent.shape) == 2
    _, parent_num = parent.shape
    u, u_counts = np.unique(np.c_[node, parent], return_counts=True, axis=0)
    counts = np.zeros(tuple([base] * (parent_num + 1)))
    for idx, count in zip(u, u_counts):
        counts[tuple(idx)] = count
    cpt = counts / np.maximum(eps, np.sum(counts, axis=0))
    return cpt.T


def packbits(bin_array, reverse=True):
    """
    Convert a binary number to a decimal number.

    Parameters
    ----------
    bin_array : numpy.ndarray
        Bit strings.
    reverse : bool, default True
        If the last element of the list is to be treated as the first bit of the bit string, then True, else False.

    Returns
    -------
    numpy.ndarray
        Decimal numbers.
    """
    p = np.power(2, np.arange(bin_array.shape[-1]))
    if reverse:
        p = p[::-1]
    dec_array = np.dot(bin_array, p)
    return dec_array
