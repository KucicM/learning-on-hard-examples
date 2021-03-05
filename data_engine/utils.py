

def unpack_inner_tuple(func):
    def inner(*args, **kwargs):
        values = func(*args, **kwargs)
        return [v[0] for v in values]
    return inner
