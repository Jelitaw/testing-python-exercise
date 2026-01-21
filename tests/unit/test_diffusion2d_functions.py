"""
Tests for functions in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D


def test_initialize_domain():
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()

    # input values
    w = 12.0
    h = 8.0
    dx = 0.5
    dy = 0.25

    # expected results
    expected_nx = 24
    expected_ny = 32

    # call function under test
    solver.initialize_domain(w=w, h=h, dx=dx, dy=dy)

    # assertions
    assert solver.nx == expected_nx
    assert solver.ny == expected_ny

def test_initialize_physical_parameters():
    """
    Check function SolveDiffusion2D.initialize_physical_parameters
    """
    solver = SolveDiffusion2D()

    solver.dx = 1.0
    solver.dy = 2.0

    d = 2.0
    T_cold = 300.0
    T_hot = 700.0

    # dx² = 1
    # dy² = 4
    #
    # dt = (1 * 4) / (2 * 2 * (1 + 4))
    # dt = 4 / 20
    # dt = 0.2
    # expected result
    expected_dt = 0.2

    solver.initialize_physical_parameters(d=d, T_cold=T_cold, T_hot=T_hot)

    # assertions
    assert solver.D == d
    assert solver.T_cold == T_cold
    assert solver.T_hot == T_hot
    assert solver.dt == expected_dt

def test_set_initial_condition():
    """
    Check function SolveDiffusion2D.set_initial_condition
    """
    solver = SolveDiffusion2D()

    solver.nx = 10
    solver.ny = 10
    solver.dx = 1.0
    solver.dy = 1.0
    solver.T_cold = 0.0
    solver.T_hot = 100.0

    # call function under test
    u = solver.set_initial_condition()

    # basic shape check
    assert u.shape == (10, 10)

    # center should be hot
    assert u[5, 5] == solver.T_hot

    # corner should be cold
    assert u[0, 0] == solver.T_cold
