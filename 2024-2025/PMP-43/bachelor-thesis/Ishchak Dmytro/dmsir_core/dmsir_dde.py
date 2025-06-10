"""
dmsir_dde.py – затримкова D-MSIR-модель (τ=3,7 доби).
Однофайлове ядро: Runge–Kutta 4(5) «dopri5» із SciPy.
"""

from dataclasses import dataclass
from collections  import deque
import numpy as np
from  scipy.integrate import ode


@dataclass
class Params:          # базові параметри (можна змінювати через kwargs)
    beta: float =.28   # передача
    gamma: float =.17  # одужання
    nu: float =.003    # втрата імунітету
    theta: float =.42  # відносна заразність M
    tau: float =3.7    # лаг інфекції (доба)
    N: int=38_000_000


def _rhs(t, y, p: Params, hist: deque):
    """Права частина DDE — інтерполяція D(t-τ) лінійно."""
    D, M, I, R = y

    # лагове Dτ (лінійна інтерполяція)
    while len(hist) > 1 and hist[1][0] >= t - p.tau:
        hist.popleft()
    (t0, y0), (t1, y1) = hist[0], hist[1] if len(hist) > 1 else hist[0]
    α = (t - p.tau - t0) / (t1 - t0 or 1e-9)
    Dτ = y0[0] * (1 - α) + y1[0] * α

    λ = p.beta * np.exp(-p.gamma * p.tau)        # затримковий контакт

    dD = p.nu * R - D / p.tau
    dM = D / p.tau - λ * M
    dI = λ * (M + p.theta * Dτ) - p.gamma * I
    dR = p.gamma * I - p.nu * R
    return np.array([dD, dM, dI, dR])


def simulate(days=180, step=.25, p: Params = Params()):
    """Повертає матрицю T×5  (t, D, M, I, R)."""
    y0 = np.array([.001, 0., .0001, 0.]) * p.N
    hist = deque([(0., y0.copy()), (-p.tau, y0.copy())])

    solver = ode(lambda t, y: _rhs(t, y, p, hist)).set_integrator(
        "dopri5", rtol=1e-7, atol=1e-9
    ).set_initial_value(y0, 0.)

    T, Y = [0.], [y0.copy()]
    while solver.successful() and solver.t < days:
        solver.integrate(solver.t + step)
        hist.appendleft((solver.t, solver.y.copy()))
        T.append(solver.t); Y.append(solver.y.copy())
    return np.column_stack((T, np.vstack(Y)))


if __name__ == "__main__":        # CLI-демо
    arr = simulate()
    I, t = arr[:, 3], arr[:, 0]
    print(f"Пік I = {I.max():.0f} осіб → день {t[I.argmax()]:.1f}")
