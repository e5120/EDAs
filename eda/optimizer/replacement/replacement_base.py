from abc import ABCMeta, abstractmethod


class ReplacementBase(metaclass=ABCMeta):
    """
    Base class of replacement methods.
    """
    def __init__(self, replace_rate, fix_size=True):
        """
        Parameters
        ----------
        replace_rate : float
            Replacement rate, i.e., how many individuals are replaced when the replacement method is applied to a population.
        fix_size : bool, default True
            Whether a population size should be the same before and after the replacement.
        """
        assert 0.0 < replace_rate <= 1.0
        self.replace_rate = replace_rate
        self.fix_size = fix_size

    def __call__(self, parent, p_evals, candidate, c_evals):
        return self.apply(parent, p_evals, candidate, c_evals)

    @abstractmethod
    def apply(self, parent, p_evals, candidate, c_evals):
        """
        Apply replacement to a population.
        The replacement method replaces the individuals in parent with ones in candidate.

        Parameters
        ----------
        parent : numpy.ndarray
            A population which individuals are replaced.
        p_evals : numpy.ndarray
            The evaluation values corresponding to individuals in parent.
        candidate : numpy.ndarray
            A population used for the replacement.
        c_evals : numpy.ndarray
            The evaluation values corresponding to individuals in candidate.

        Returns
        -------
        numpy.ndarray
            Parent population after the individuals are replaced.
        numpy.ndarray
            The evaluation values corresponding to the above population.
        """
        pass

    def __str__(self):
        return 'replacement rate: {}\n' \
               'fix size: {}'.format(self.replace_rate, self.fix_size)
