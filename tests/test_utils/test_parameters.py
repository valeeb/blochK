import numpy as np
from blochK.utils.parameters import dict2listofdict

def test_dict2listofdict():
    ds = {'a':[1,2],'b':[3,4],'c':5}
    result = dict2listofdict(ds)
    expected = [{'a': 1, 'b': 3, 'c': 5},
                {'a': 1, 'b': 4, 'c': 5},
                {'a': 2, 'b': 3, 'c': 5},
                {'a': 2, 'b': 4, 'c': 5}]
    assert result == expected   , "dict2listofdict did not produce the expected output"