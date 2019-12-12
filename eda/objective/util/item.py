class Item(object):
    def __init__(self, n, v, w):
        self.name = n
        self.value = v
        self.weight = w

    def __str__(self):
        return "[{}] weigth = {},\tvalue = {}".format(self.name, self.weight, self.value)
