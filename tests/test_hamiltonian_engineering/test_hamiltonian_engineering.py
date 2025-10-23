import numpy as np
import pickle
from blochK.hamiltonian_engineering.utils import (
    t_values_to_vector,
    vector_to_t_values,
    _derivative_factor,
    constraint_as_matrix_entry,
    total_matrix_from_constraints,
    solve_constraint_matrix,
)


def test_t_values_vector_conversion():
    # check with a sample t_values dictionary in 3d
    t_values = {
        (0, 0, 0): np.array([1, 3, 5, 7]),
        (1, 0, 0): np.array([9 + 10.0j, 11 + 12j, 13 + 14j, 15 + 16j]),
        (0, 1, 0): np.array([17 + 18j, 19 + 20j, 21 + 22j, 23 + 24j]),
        (0, 0, 1): np.array([25 + 26j, 27 + 28j, 29 + 30j, 31 + 32j]),
    }

    t_values_2d = {
        (0, 0): np.array([1, 3, 5, 7.0]),
        (1, 0): np.array([10 + 11j, 12 + 13j, 14 + 15j, 16 + 17j]),
        (0, 1): np.array([18 + 19j, 20 + 21j, 22 + 23j, 24 + 25j]),
    }

    for t in [t_values, t_values_2d]:
        vector, hoppings_list = t_values_to_vector(t)
        reconstructed_t_values = vector_to_t_values(vector, hoppings_list)
        new_vector, new_hoppings_list = t_values_to_vector(reconstructed_t_values)

        for key in t:
            assert np.allclose(
                t[key], reconstructed_t_values[key]
            ), f"Mismatch in t_values for hopping {key}"
        assert np.allclose(
            vector, new_vector
        ), "Mismatch in vector representation after conversion"
        assert (
            hoppings_list == new_hoppings_list
        ), "Mismatch in hoppings list after conversion"


def test_derivative_factor():
    # test _derivative_factor function
    # in 3d
    position, derivative, hopping = (1, 0, 0), (1, 0, 0), (2, 0, 0)
    factor = _derivative_factor(position, derivative, hopping)
    assert np.isclose(
        factor, -1.8185948536513634 + 0.8322936730942848j
    ), "Incorrect derivative factor in 3D case"

    # in 2d
    position, derivative, hopping = (0, 1), (0, 1), (0, 3)
    factor = _derivative_factor(position, derivative, hopping)
    assert np.isclose(
        factor, -0.4233600241796016 + 2.9699774898013365j
    ), "Incorrect derivative factor in 2D case"

def test_constraint_as_matrix_entry():
    # test constraint_as_matrix_entry function
        
    # in 3d
    hoppings_list_3d = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    constraint_1 = ((1, 2, 3), (0, 0, 0), (6, 7, 8, 9))
    constraint_2 = ((1, 2, 3), (0, 1, 0), (6, 7, 8, 9))
    matrix_entry_3d_1, _ = constraint_as_matrix_entry(constraint_1, hoppings_list_3d)
    matrix_entry_3d_2, _ = constraint_as_matrix_entry(constraint_2, hoppings_list_3d)

    # in 2d
    hoppings_list_2d = [(0, 0), (1, 0), (0, 1)]
    constraint_3 = ((1, 2), (0, 0), (4, 5, 6, 7))
    constraint_4 = ((1, 2), (1, 0), (4, 5, 6, 7))
    matrix_entry_2d_3, _ = constraint_as_matrix_entry(constraint_3, hoppings_list_2d)
    matrix_entry_2d_4, _ = constraint_as_matrix_entry(constraint_4, hoppings_list_2d)

    calculated_entries = [
        matrix_entry_3d_1,
        matrix_entry_3d_2,
        matrix_entry_2d_3,
        matrix_entry_2d_4,
    ]

    with open("tests/test_hamiltonian_engineering/test_data/constraint_matrix_entry.pkl", "rb") as f:
        expected_entries = pickle.load(f)
    for calc, exp in zip(calculated_entries, expected_entries):
        assert np.allclose(calc, exp), "Mismatch in constraint matrix entry calculation"

def test_total_matrix_from_constraints():

    # in 3d
    constraint1 = ((1, 2, 3), (0, 0, 0), (1, 2, 3, 4))
    constraint2 = ((4, 5, 6), (1, 0, 0), (5, 6, 7, 8))
    constraints_3d = {constraint1, constraint2}
    hoppings_list_3d = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    matrix_3d, _  = total_matrix_from_constraints(constraints_3d, hoppings_list_3d)

    # in 2d
    constraint3 = ((1, 2), (0, 0), (1, 2, 3, 4))
    constraint4 = ((3, 4), (1, 0), (5, 6, 7, 8))
    constraints_2d = {constraint3, constraint4}
    hoppings_list_2d = [(0, 0), (1, 0), (0, 1)]
    matrix_2d, _ = total_matrix_from_constraints(constraints_2d, hoppings_list_2d)
    print(matrix_2d)

    with open("tests/test_hamiltonian_engineering/test_data/total_constraint_matrix_3d.pkl", "rb") as f:
        (expected_matrix_3d,expected_matrix_2d) = pickle.load(f)

    assert np.allclose(matrix_3d, expected_matrix_3d), "Mismatch in total constraint matrix 3D"
    assert np.allclose(matrix_2d, expected_matrix_2d), "Mismatch in total constraint matrix 2D"

    

