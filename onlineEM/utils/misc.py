def size_params(x):
    if isinstance(x, (tuple, list)):
        return sum(size_params(y) for y in x)
    else:
        return x.size
