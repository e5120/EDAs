class MutationBase(object):
    def __init__(self):
        pass

    def apply(self, popoulation, fitness):
        raise NotImplementedError
