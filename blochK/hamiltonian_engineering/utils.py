import numpy as np
from scipy import linalg as la
from scipy.linalg import lstsq, null_space
from collections.abc import Iterable

pauli_vector = np.array(
    [
        np.array([[1, 0], [0, 1]]),  # identity
        np.array([[0, 1], [1, 0]]),  # sigma x
        np.array([[0, -1j], [1j, 0]]),  # sigma y
        np.array([[1, 0], [0, -1]]),  # sigma z
    ]
)


def d_matrix(d_vector):
    """generates a 2x2 matrix of the form
    x_0 sigma_0 + d_x sigma_x + d_y sigma_y + d_z sigma_z
    """

    return np.sum(d_vector[:, None, None] * pauli_vector, axis=0)


def t_values_to_vector(t_values):
    """Transforms a dictionary of t_values into a vector
    representation of the real and imaginary parts. The (0,0,0) hopping
    only has real parts, so contributes 4 elements, while other hoppings
    contribute 8 elements (real and imaginary parts).

    Args:
        t_values (dict): a dictionary of every lattice vector and the associated (complex)
            d vector for it
    Returns:
        np.ndarray: vector representation
        list: list of hoppings in order, ensures consistent ordering
    """

    hoppings_list = sorted(t_values.keys())
    vector = []
    for hopping in hoppings_list:
        if all(v == 0 for v in hopping):

            vector.extend(t_values[hopping].real)
        else:
            vector.extend(t_values[hopping].real)
            vector.extend(t_values[hopping].imag)
    return np.array(vector), hoppings_list


def vector_to_t_values(vector, hoppings_list):
    """Transforms a vector representation of t_values back into a dictionary.
    Args:
        vector (np.ndarray): vector representation
        hoppings_list (list): list of hoppings in order
    Returns:
        dict: t_values dictionary
    """
    hoppings_list = sorted(hoppings_list)
    t_values = {}
    i = 0
    for hopping in hoppings_list:
        if all(v == 0 for v in hopping):
            real_part = vector[i : i + 4]
            t_values[hopping] = real_part
            i += 4
            continue
        else:
            real_part = vector[i : i + 4]
            imag_part = vector[i + 4 : i + 8]
            t_values[hopping] = real_part + 1j * imag_part
            i += 8
    return t_values


def constraints_from_weyl_node(weyl_pos, chirality = None):
    """Returns the constraints that place a Weyl node of the form
    chirality * sigma . k around the specified weyl position.

    Args:
        weyl_pos (tuple): the position of the Weyl node
        chirality (float): the chirality of the Weyl node

    Returns:
        set: The constraints in the form {(position, derivative, value)}
    """
    if not isinstance(chirality, Iterable):
        chirality = (chirality, chirality, chirality)

    weyl_pos = tuple(weyl_pos)
    zero_order = weyl_pos, (0, 0, 0), (0, 0, 0, 0)
    x_deriv = weyl_pos, (1, 0, 0), (0, chirality[0], 0, 0)
    y_deriv = weyl_pos, (0, 1, 0), (0, 0, chirality[1], 0)
    z_deriv = weyl_pos, (0, 0, 1), (0, 0, 0, chirality[2])
    return {zero_order, x_deriv, y_deriv, z_deriv}


def _derivative_factor(position, derivative, hopping):
    # computes the factor arising from applying the derivative operator
    # to the exponential term e^{-i k . r}
    expo = np.exp(-1j * (np.dot(position, hopping)))
    prefactor = 1
    for dim in range(len(position)):
        prefactor *= (-1j * hopping[dim]) ** (derivative[dim])
    return prefactor * expo


def constraint_as_matrix_entry(constraint, hoppings_list):
    """Given a single constraint, of the form (position, derivative, value),
    which sets an arbitrary derivative of the d-vector at a position to a value.
    Works in n-dimensions.
    Args:
        constraint (tuple): (position, derivative, value), with
            position (n-vector): k-point where the constraint is applied
            derivative (n-vector): which derivative to apply
            value (4-vector): desired value of the d-vector at that point
        hoppings_list (list): list of hoppings in order
    Returns:
        np.ndarray: matrix row corresponding to the constraint
        np.ndarray: value vector corresponding to the constraint
    """
    hoppings_list = sorted(hoppings_list)
    position, derivative, value = constraint

    factors = {}
    for hopping in hoppings_list:
        factor = _derivative_factor(position, derivative, hopping)
        factors[hopping] = factor

    rows = []
    values_out = []
    for j,v in enumerate(value):
        if v is None: 
            continue

        factor_j = {
            hopping: factors[hopping] * np.eye(4)[j] for hopping in hoppings_list
        }
        r, keys = t_values_to_vector(factor_j)
        rows.append(r)
        values_out.append(v)

        # make sure the keys are still in the same order
        assert keys == hoppings_list

    rows = np.array(rows)
    values_out = np.array(values_out)
    return rows, values_out


def total_matrix_from_constraints(constraints, hoppings_list):
    """Given a set of constraints, builds the total constraint matrix
    and value vector. These can then be used to solve for the t_values vector
    that satisfies all constraints.
    Args:
        constraints (set): set of constraints, each of the form
            (position, derivative, value)
        hoppings_list (list): list of hoppings in order
    Returns:
        np.ndarray: total constraint matrix
        np.ndarray: total value vector
    """
    hoppings_list = sorted(hoppings_list)
    rows = []
    values = []
    for constraint in constraints:
        row, value = constraint_as_matrix_entry(constraint, hoppings_list)
        rows.append(row)
        values.extend(value)
    return np.vstack(rows), np.array(values)



def solve_constraint_matrix(matrix, value_vector):
    """Solves the constraint matrix equation Ax = b, where A is the
    constraint matrix, x is the vector of t_values, and b is the value vector.
    Args:
        matrix (np.ndarray): constraint matrix
        value_vector (np.ndarray): value vector
    Returns:
        np.ndarray: solution vector
        np.ndarray: basis for the null space of the constraint matrix
    """

    rank_matrix = np.linalg.matrix_rank(matrix)
    rank_augmented = np.linalg.matrix_rank(np.column_stack([matrix, value_vector]))
    assert rank_matrix == rank_augmented, f"\nSystem is inconsistent (no solution).\nM_rank: {rank_matrix} < Aug_rank: {rank_augmented}"
      
    sol, residuals, _, _ = lstsq(matrix, value_vector)
    assert np.allclose(residuals,0)

    # cleans up the null space
    null_basis = null_space(matrix)
    proj = np.einsum("ij,jk -> ik", null_basis, null_basis.conj().T  )
    proj = np.einsum("ij,j,jk -> ik",proj,np.arange(proj.shape[0])+1 ,proj)
    e,v = la.eigh(proj)
    v_good = v[:, e > 0.1]

    return sol, v_good
