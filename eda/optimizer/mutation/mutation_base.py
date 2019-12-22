class MutationBase(object):
    def __init__(self):
        pass

    def __call__(self, popoulation):
        return self.apply(popoulation)

    def apply(self, popoulation):
        raise NotImplementedError
