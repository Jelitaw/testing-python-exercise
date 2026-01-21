"""
Tests for functionality checks in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D


def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()

    solver.initialize_domain(w=10.0, h=10.0, dx=1.0, dy=1.0)
    solver.initialize_physical_parameters(d=2.0, T_cold=300.0, T_hot=700.0)

    expected_dt = 0.125

    assert solver.dt == expected_dt
    
    assert solver.D == 2.0
    assert solver.T_cold == 300.0
    assert solver.T_hot == 700.0


def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver = SolveDiffusion2D()

    solver.initialize_domain(w=10.0, h=10.0, dx=1.0, dy=1.0)
    solver.initialize_physical_parameters(d=1.0, T_cold=0.0, T_hot=100.0)

    u = solver.set_initial_condition()

    # shape
    assert u.shape == (solver.nx, solver.ny)

    # center index
    cx_index = int(5 / solver.dx)
    cy_index = int(5 / solver.dy)

    # check center and corner values
    assert u[cx_index, cy_index] == solver.T_hot
    assert u[0,0] == solver.T_cold