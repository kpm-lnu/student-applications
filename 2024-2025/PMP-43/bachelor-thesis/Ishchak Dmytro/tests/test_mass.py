from dmsir_core.dmsir_dde import simulate
def test_mass():
    arr = simulate(30, 1.)
    tot = arr[:, 1:].sum(1)
    assert abs(tot.max() - tot.min()) < 1e-4
