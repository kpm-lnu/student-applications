"""
imm_kalman.py
Interacting Multiple Model Kalman Filter (IMM-KF).
Три паралельні моделі: CV (стала швидкість),
                       CA (стале прискорення),
                       CT (координований розворот).

Вектор стану:  x = [u, v, du, dv, ddu, ddv, w, h]  (8-dim)
Вимірювання:   z = [u, v, w, h]                     (4-dim)

Усі координати у пікселях.
"""

import numpy as np
from typing import Optional, Tuple
from src.config import DT, GATE_CHI2


# Default IMM parameters. These live here because the generic project config
# does not define filter-specific tuning constants.
IMM_MU_INIT = [0.7, 0.2, 0.1]
IMM_PI = [
    [0.90, 0.07, 0.03],
    [0.07, 0.90, 0.03],
    [0.03, 0.07, 0.90],
]

NOISE_CV = {"pos": 1.0, "vel": 0.5, "acc": 0.1, "wh": 0.5}
NOISE_CA = {"pos": 1.2, "vel": 0.7, "acc": 0.2, "wh": 0.5}
NOISE_CT = {"pos": 1.0, "vel": 0.7, "acc": 0.2, "wh": 0.5}
MEAS_NOISE = [10.0, 10.0, 25.0, 25.0]


# ═══════════════════════════════════════════════════════════════
#  Допоміжні конструктори матриць
# ═══════════════════════════════════════════════════════════════

def _make_F_cv(dt: float) -> np.ndarray:
    """Матриця переходу: Constant Velocity."""
    F = np.eye(8)
    # Позиція += швидкість * dt
    F[0, 2] = dt
    F[1, 3] = dt
    return F


def _make_F_ca(dt: float) -> np.ndarray:
    """Матриця переходу: Constant Acceleration."""
    F = np.eye(8)
    F[0, 2] = dt;       F[0, 4] = 0.5 * dt ** 2
    F[1, 3] = dt;       F[1, 5] = 0.5 * dt ** 2
    F[2, 4] = dt
    F[3, 5] = dt
    return F


def _make_F_ct(dt: float, omega: float) -> np.ndarray:
    """
    Матриця переходу: Coordinated Turn (лінеаризована).
    omega: кутова швидкість [рад/кадр].
    При |omega| < eps → вироджується у CV.
    """
    F = np.eye(8)
    eps = 1e-6
    if abs(omega) < eps:
        return _make_F_cv(dt)

    sin_odt = np.sin(omega * dt)
    cos_odt = np.cos(omega * dt)

    F[0, 2] =  sin_odt / omega
    F[0, 3] = -(1.0 - cos_odt) / omega
    F[1, 2] =  (1.0 - cos_odt) / omega
    F[1, 3] =  sin_odt / omega
    F[2, 2] =  cos_odt
    F[2, 3] = -sin_odt
    F[3, 2] =  sin_odt
    F[3, 3] =  cos_odt
    return F


def _make_Q(noise: dict, dt: float) -> np.ndarray:
    """Матриця шуму процесу Q (8x8)."""
    Q = np.zeros((8, 8))
    Q[0, 0] = noise["pos"]
    Q[1, 1] = noise["pos"]
    Q[2, 2] = noise["vel"]
    Q[3, 3] = noise["vel"]
    Q[4, 4] = noise["acc"]
    Q[5, 5] = noise["acc"]
    Q[6, 6] = noise["wh"]
    Q[7, 7] = noise["wh"]
    return Q * dt


def _make_H() -> np.ndarray:
    """Матриця спостереження H (4x8): вимірюємо u, v, w, h."""
    H = np.zeros((4, 8))
    H[0, 0] = 1.0   # u
    H[1, 1] = 1.0   # v
    H[2, 6] = 1.0   # w
    H[3, 7] = 1.0   # h
    return H


def _make_R(meas_noise: list) -> np.ndarray:
    """Матриця шуму вимірювання R (4x4)."""
    return np.diag(meas_noise).astype(float)


# ═══════════════════════════════════════════════════════════════
#  Базовий фільтр Калмана (один екземпляр для однієї моделі)
# ═══════════════════════════════════════════════════════════════

class KalmanFilter:
    """Лінійний КФ для одної моделі руху."""

    def __init__(self, F: np.ndarray, Q: np.ndarray,
                 H: np.ndarray, R: np.ndarray):
        self.F = F.copy()
        self.Q = Q.copy()
        self.H = H
        self.R = R
        self.n = F.shape[0]   # розмірність стану

    def predict(self, x: np.ndarray,
                P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Крок прогнозування."""
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray,
               z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Крок оновлення.
        Повертає: (x_upd, P_upd, likelihood)
        """
        # Інновація
        innov = z - self.H @ x_pred
        S     = self.H @ P_pred @ self.H.T + self.R

        # Підсилення Калмана
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        x_upd = x_pred + K @ innov
        I_KH  = np.eye(self.n) - K @ self.H
        P_upd = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T  # Joseph form

        # Правдоподібність (скалярне гауссове правдоподібність)
        m    = innov.shape[0]
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            likelihood = 1e-300
        else:
            maha2     = float(innov.T @ np.linalg.inv(S) @ innov)
            likelihood = np.exp(-0.5 * (maha2 + logdet +
                                        m * np.log(2 * np.pi)))
        return x_upd, P_upd, max(likelihood, 1e-300)

    def mahalanobis(self, x_pred: np.ndarray,
                    P_pred: np.ndarray,
                    z: np.ndarray) -> float:
        """Відстань Махаланобіса (для gate-перевірки)."""
        innov = z - self.H @ x_pred
        S     = self.H @ P_pred @ self.H.T + self.R
        return float(innov.T @ np.linalg.inv(S) @ innov)


# ═══════════════════════════════════════════════════════════════
#  IMM-KF
# ═══════════════════════════════════════════════════════════════

class IMMKalmanFilter:
    """
    Interacting Multiple Model Kalman Filter.
    Підтримує 3 паралельних КФ: CV, CA, CT.

    Публічний інтерфейс:
        filter = IMMKalmanFilter(z_init)
        x, P   = filter.predict()
        filter.update(z)
        mu     = filter.model_probs   # [p_CV, p_CA, p_CT]
    """

    MODEL_NAMES = ["CV", "CA", "CT"]

    def __init__(self, z_init: np.ndarray, dt: float = DT):
        """
        Args:
            z_init: перше вимірювання (u, v, w, h) у пікселях
            dt:     міжкадровий інтервал
        """
        self.dt  = dt
        self.H   = _make_H()
        self.R   = _make_R(MEAS_NOISE)

        # Матриця переходу між моделями (рядок=з, стовпець=в)
        self.Pi  = np.array(IMM_PI, dtype=float)

        # Ймовірності моделей
        self.mu  = np.array(IMM_MU_INIT, dtype=float)
        self.mu /= self.mu.sum()

        # Ініціалізація стану
        x0 = np.zeros(8)
        x0[0], x0[1] = z_init[0], z_init[1]  # u, v
        x0[6], x0[7] = z_init[2], z_init[3]  # w, h

        P0 = np.diag([100., 100., 25., 25., 10., 10., 50., 50.])

        # Три незалежних стани та коваріації
        self.x = [x0.copy(), x0.copy(), x0.copy()]
        self.P = [P0.copy(), P0.copy(), P0.copy()]

        # Будуємо три КФ
        self.filters = [
            KalmanFilter(_make_F_cv(dt), _make_Q(NOISE_CV, dt),
                         self.H, self.R),
            KalmanFilter(_make_F_ca(dt), _make_Q(NOISE_CA, dt),
                         self.H, self.R),
            KalmanFilter(_make_F_ct(dt, 0.0), _make_Q(NOISE_CT, dt),
                         self.H, self.R),
        ]

        # Поточна кутова швидкість для CT-моделі
        self._omega = 0.0

        # Прогнозований стан (зберігаємо для gate-перевірки ззовні)
        self.x_pred: Optional[np.ndarray] = None
        self.P_pred: Optional[np.ndarray] = None

    # ── Публічний API ─────────────────────────────────────────

    @property
    def model_probs(self) -> np.ndarray:
        """Поточний розподіл ймовірностей [p_CV, p_CA, p_CT]."""
        return self.mu.copy()

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Крок 1+2 IMM: змішування + паралельна фільтрація.
        Повертає комбіновану апріорну оцінку (x_pred, P_pred).
        """
        r = len(self.filters)

        # ── Крок 1: ймовірності змішування μ^{i|j} ───────────
        c_bar = self.Pi.T @ self.mu          # нормуючий вектор
        mix   = np.zeros((r, r))
        for i in range(r):
            for j in range(r):
                mix[i, j] = (self.Pi[i, j] * self.mu[i]) / (c_bar[j] + 1e-300)

        # ── Крок 2: змішані початкові умови ──────────────────
        x0_mix = []
        P0_mix = []
        for j in range(r):
            xj = sum(mix[i, j] * self.x[i] for i in range(r))
            Pj = np.zeros((8, 8))
            for i in range(r):
                d  = (self.x[i] - xj).reshape(-1, 1)
                Pj += mix[i, j] * (self.P[i] + d @ d.T)
            x0_mix.append(xj)
            P0_mix.append(Pj)

        # ── Оновлення CT-матриці F з поточною omega ──────────
        self._update_omega()
        self.filters[2].F = _make_F_ct(self.dt, self._omega)

        # ── Крок 2 продовження: паралельне прогнозування ─────
        self._x_pred_each = []
        self._P_pred_each = []
        for j, flt in enumerate(self.filters):
            xp, Pp = flt.predict(x0_mix[j], P0_mix[j])
            self._x_pred_each.append(xp)
            self._P_pred_each.append(Pp)

        # ── Комбінована апріорна оцінка ───────────────────────
        x_pred = sum(self.mu[j] * self._x_pred_each[j]
                     for j in range(r))
        P_pred = np.zeros((8, 8))
        for j in range(r):
            d       = (self._x_pred_each[j] - x_pred).reshape(-1, 1)
            P_pred += self.mu[j] * (self._P_pred_each[j] + d @ d.T)

        self.x_pred = x_pred
        self.P_pred = P_pred
        return x_pred.copy(), P_pred.copy()

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Крок 3+4+5 IMM: паралельне оновлення,
        оновлення ймовірностей моделей, злиття оцінок.

        Args:
            z: вимірювання [u, v, w, h]
        Returns:
            (x_upd, P_upd)
        """
        r = len(self.filters)
        lambdas = []
        x_upd_each = []
        P_upd_each = []

        # ── Крок 3: паралельне оновлення + правдоподібності ──
        for j, flt in enumerate(self.filters):
            xu, Pu, lam = flt.update(
                self._x_pred_each[j], self._P_pred_each[j], z
            )
            x_upd_each.append(xu)
            P_upd_each.append(Pu)
            lambdas.append(lam)

        # ── Крок 4: оновлення ймовірностей моделей ───────────
        c_bar = self.Pi.T @ self.mu
        mu_new = np.array([lambdas[j] * c_bar[j] for j in range(r)])
        mu_sum = mu_new.sum()
        if mu_sum < 1e-300:
            mu_new = np.array(IMM_MU_INIT, dtype=float)
        else:
            mu_new /= mu_sum
        self.mu = mu_new

        # ── Крок 5: злиття фінальних оцінок ──────────────────
        x_upd = sum(self.mu[j] * x_upd_each[j] for j in range(r))
        P_upd = np.zeros((8, 8))
        for j in range(r):
            d      = (x_upd_each[j] - x_upd).reshape(-1, 1)
            P_upd += self.mu[j] * (P_upd_each[j] + d @ d.T)

        # Зберігаємо апостеріорні оцінки
        self.x = x_upd_each
        self.P = P_upd_each

        return x_upd.copy(), P_upd.copy()

    def gate_check(self, z: np.ndarray) -> bool:
        """
        Перевіряє, чи вимірювання z знаходиться у воротах
        (Mahalanobis <= chi2(4, 0.95) = 9.488).
        """
        if self.x_pred is None or self.P_pred is None:
            return True   # ворота ще не ініціалізовані
        # Використовуємо комбінований P_pred
        S    = self.H @ self.P_pred @ self.H.T + self.R
        innov = z - self.H @ self.x_pred
        maha2 = float(innov.T @ np.linalg.inv(S) @ innov)
        return maha2 <= GATE_CHI2

    def get_bbox(self) -> np.ndarray:
        """
        Повертає поточний bounding box [u, v, w, h] у пікселях
        із комбінованої оцінки стану.
        """
        if self.x_pred is not None:
            x = self.x_pred
        else:
            x = self.x[0]
        return np.array([x[0], x[1], x[6], x[7]])

    # ── Приватні методи ───────────────────────────────────────

    def _update_omega(self):
        """
        Оцінює кутову швидкість omega з поточного стану CV/CA.
        Формула: omega ≈ (vx * ay - vy * ax) / (vx^2 + vy^2)
        """
        # Середньозважений стан для оцінки
        x_avg = sum(self.mu[j] * self.x[j] for j in range(3))
        vx, vy = x_avg[2], x_avg[3]
        ax, ay = x_avg[4], x_avg[5]
        denom  = vx ** 2 + vy ** 2
        if denom < 1e-6:
            self._omega = 0.0
        else:
            self._omega = float((vx * ay - vy * ax) / denom)
        # Обмежуємо кутову швидкість розумним діапазоном
        self._omega = np.clip(self._omega, -np.pi / 2, np.pi / 2)
