import itertools

def dict2listofdict(d):
    """Transforms a dictionary with lists to a list of dictionaries taking all possible values.
    Example:
        d = {'a':[1,2],'b':[3,4]}
        dict2listofdict(d)
        >> [{'a': 1, 'b': 3},
            {'a': 1, 'b': 4},
            {'a': 2, 'b': 3},
            {'a': 2, 'b': 4}]
    """
    #make all elemts in d iterable
    for key in d:
        if not hasattr(d[key],'__iter__') or isinstance(d[key],str):
            d[key] = [d[key]]

    lod = []
    for entry in itertools.product(*[d[i] for i in d]):
        kwargs = {param: value for param, value in zip(d, entry)}
        lod.append(kwargs.copy())
    return lod