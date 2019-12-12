class MutationBase(object):
    def __init__(self):
        pass

    def __call__(self, popoulation, fitness):
        return self.apply(popoulation, fitness)

    def apply(self, popoulation, fitness):
        raise NotImplementedError
