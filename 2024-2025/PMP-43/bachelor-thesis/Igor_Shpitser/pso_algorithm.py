# -*- coding: utf-8 -*-
"""
pso_algorithm.py — роєва оптимізація для осесиметричної квадратури.

• PSO — шукає максимум fitness за замовчуванням.
• fitness_function — відношення помилок: ref_norm / err.
• get_K_ref — еталонна K-матриця (Gauss nG×nG).
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss

from axisymmetric_quadrature_pso import PSOQuadrature
from element import AxisymmetricElement
from shapeFunction import LinearQuadrilateralShapeFunction

# ──────────────────────────────────────────────────────────
# 15×15 дуже точна квадратура (для get_K_ref, якщо потрібно)
_pts15, _wts15 = leggauss(15)
_xi15, _eta15  = np.meshgrid(_pts15, _pts15, indexing="ij")
_w15           = np.outer(_wts15, _wts15)
_QUAD_REF      = PSOQuadrature(_xi15.ravel(), _eta15.ravel(), _w15.ravel())

# ──────────────────────────────────────────────────────────
# Частинка рою
class Particle:
    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self.position      = position.copy()
        self.velocity      = velocity.copy()
        self.best_position = position.copy()
        self.best_value    = -np.inf  # для максимізації

# ───────── мінімальний mesh для обчислення K ─────────
class _Node:
    def __init__(self, r: float, z: float):
        self.r = r
        self.z = z

class _MiniMesh:
    """Містить лише nodes та node_dof."""
    def __init__(self, coords: np.ndarray, node_dof: int = 2):
        self.nodes    = {i: _Node(r, z) for i, (r, z) in enumerate(coords)}
        self.node_dof = node_dof

# ──────────────────────────────────────────────────────────
# Функція пристосованості
def fitness_function(position: np.ndarray,
                     coords:   np.ndarray,
                     material,
                     K_ref:    np.ndarray,
                     shape_func) -> float:
    """
    fitness = ||K2 - K_ref||_F / ||K_test - K_ref||_F
    де K2 — стандартна 2×2 Gauss–Legendre,
         K_test — ваша тестова PSO-квадратура.
    """
    n       = len(position) // 3
    xi      = position[0:n]
    eta     = position[n:2*n]
    weights = position[2*n:3*n]

    # 1) K_test
    quad_test = PSOQuadrature(xi=xi, eta=eta, weights=weights)
    node_ids  = list(range(len(coords)))
    mini_mesh = _MiniMesh(coords)
    K_test    = AxisymmetricElement(0, node_ids, material, shape_func, quad_test) \
                    .compute_element_stiffness(mini_mesh)

    # 2) K2 (2×2 Gauss–Legendre)
    pts2, wts2 = leggauss(2)
    xi2, eta2  = np.meshgrid(pts2, pts2, indexing="ij")
    w2         = np.outer(wts2, wts2)
    quad2      = PSOQuadrature(xi2.ravel(), eta2.ravel(), w2.ravel())
    K2         = AxisymmetricElement(0, node_ids, material, shape_func, quad2) \
                    .compute_element_stiffness(mini_mesh)

    # 3) обчислення помилок
    err = np.max(np.abs(K_test - K_ref))
    ref_norm = np.max(np.abs(K2 - K_ref))
    print('err', err, 'ref_norm', ref_norm)
    return ref_norm / err

# ──────────────────────────────────────────────────────────
# Еталонна K-матриця (звичайна Gauss nG×nG)
def get_K_ref(coords: np.ndarray, material, *, nG: int = 15,
              shape_func=None) -> np.ndarray:
    if shape_func is None:
        shape_func = LinearQuadrilateralShapeFunction()

    pts, wts = leggauss(nG)
    xi, eta  = np.meshgrid(pts, pts, indexing="ij")
    w        = np.outer(wts, wts)
    quad_ref = PSOQuadrature(xi.ravel(), eta.ravel(), w.ravel())

    node_ids  = list(range(len(coords)))
    mini_mesh = _MiniMesh(coords)
    return AxisymmetricElement(
        0, node_ids, material, shape_func, quad_ref
    ).compute_element_stiffness(mini_mesh)

# ──────────────────────────────────────────────────────────
# Клас PSO
class PSO:
    def __init__(self,
                 *,
                 fitness_func,
                 n_particles: int,
                 dim: int,
                 lo: np.ndarray,
                 hi: np.ndarray,
                 w: float = 0.7,
                 c1: float = 1.4,
                 c2: float = 1.4,
                 max_iter: int = 200,
                 patience: int = 20,
                 tol: float = 1e-6,
                 target_fitness: float | None = None,
                 maximize: bool = True,
                 rng=None):
        """
        :param maximize: якщо True — шукаємо максимум fitness
        """
        self.fitness        = fitness_func
        self.maximize       = maximize
        self.np             = int(n_particles)
        self.dim            = int(dim)
        self.lo             = np.asarray(lo, dtype=float)
        self.hi             = np.asarray(hi, dtype=float)
        self.w, self.c1, self.c2 = w, c1, c2
        self.max_iter       = max_iter
        self.patience       = patience
        self.improve_thr    = tol
        self.target_fitness = target_fitness
        self.rng            = np.random.default_rng(rng)

        self._init_swarm()

    def _init_swarm(self):
        span = self.hi - self.lo
        self.swarm = []
        for _ in range(self.np):
            pos = self.rng.uniform(self.lo, self.hi)
            vel = self.rng.uniform(-span, span)
            p = Particle(pos, vel)
            p.best_value = -np.inf if self.maximize else np.inf
            self.swarm.append(p)
        self.gbest_position = None
        self.gbest_value    = -np.inf if self.maximize else np.inf

    def optimize(self):
        no_improve = 0
        for _ in range(self.max_iter):
            improved = False
            for p in self.swarm:
                p.position = np.clip(p.position, self.lo, self.hi)
                val = self.fitness(p.position)

                # локальне оновлення
                better_local = val > p.best_value if self.maximize else val < p.best_value
                if better_local:
                    p.best_value    = val
                    p.best_position = p.position.copy()

                # глобальне оновлення
                better_global = val > self.gbest_value if self.maximize else val < self.gbest_value
                if better_global:
                    self.gbest_value    = val
                    self.gbest_position = p.position.copy()
                    improved = True
                    # рання зупинка по порогу
                    if (self.target_fitness is not None and
                        ((self.maximize and val >= self.target_fitness) or
                         (not self.maximize and val <= self.target_fitness))):
                        return self.gbest_position, self.gbest_value

            no_improve = 0 if improved else no_improve + 1
            if no_improve >= self.patience:
                break

            # оновлення швидкостей і позицій
            for p in self.swarm:
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)
                cognitive = self.c1 * r1 * (p.best_position - p.position)
                social    = self.c2 * r2 * (self.gbest_position - p.position)
                p.velocity  = self.w * p.velocity + cognitive + social
                p.position += p.velocity

        return self.gbest_position, self.gbest_value



# # -*- coding: utf-8 -*-
# """
# pso_algorithm.py — роєва оптимізація для осесиметричної квадратури.
#
# • PSO — **максимізуємо** fitness за замовчуванням.
# • fitness_function — відносна Frobenius-помилка (за потреби мінімізувати, можна передати maximize=False).
# • get_K_ref — еталонна K-матриця Gauss-Legendre nG×nG.
# """
#
# from __future__ import annotations
# import numpy as np
# from numpy.polynomial.legendre import leggauss
#
# from axisymmetric_quadrature_pso import PSOQuadrature
# from element import AxisymmetricElement
# from shapeFunction import LinearQuadrilateralShapeFunction
#
#
# class Particle:
#     def __init__(self, position: np.ndarray, velocity: np.ndarray):
#         self.position      = position.copy()
#         self.velocity      = velocity.copy()
#         self.best_position = position.copy()
#         self.best_value    = -np.inf  # для максимізації
#
#
# class _Node:
#     def __init__(self, r: float, z: float):
#         self.r = r
#         self.z = z
#
# class _MiniMesh:
#     """Мінімальний mesh: тільки .nodes та .node_dof."""
#     def __init__(self, coords: np.ndarray, node_dof: int = 2):
#         self.nodes    = {i: _Node(r, z) for i, (r, z) in enumerate(coords)}
#         self.node_dof = node_dof
#
#
# def fitness_function(position: np.ndarray,
#                      coords:   np.ndarray,
#                      material,
#                      K_ref:    np.ndarray,
#                      shape_func) -> float:
#     """
#     Відносна помилка Frobenius:
#       ||K_test - K_ref||_F / ||K_ref||_F
#     (якщо хочете максимізувати, інвертуйте значення, наприклад return 1/(err+ε)).
#     """
#     n       = len(position) // 3
#     xi      = position[0:n]
#     eta     = position[n:2*n]
#     weights = position[2*n:3*n]
#
#     quad_test = PSOQuadrature(xi=xi, eta=eta, weights=weights)
#     node_ids  = list(range(len(coords)))
#     mini_mesh = _MiniMesh(coords)
#
#     K_test = AxisymmetricElement(
#         0, node_ids, material, shape_func, quad_test
#     ).compute_element_stiffness(mini_mesh)
#
#     err = np.linalg.norm(K_test - K_ref, ord='fro')
#     ref_norm = np.linalg.norm(K_ref, ord='fro')
#     rel_err = err / ref_norm if ref_norm != 0 else err
#
#     # Щоб _максимізувати_ — повертаємо обернену величину:
#     return 1.0 / (rel_err + 1e-12)
#
#
# def get_K_ref(coords: np.ndarray, material, *,
#               nG: int = 2,
#               shape_func=None) -> np.ndarray:
#     """
#     Еталонна K-матриця через стандартну Gauss nG×nG.
#     """
#     if shape_func is None:
#         shape_func = LinearQuadrilateralShapeFunction()
#
#     pts, wts = leggauss(nG)
#     xi, eta  = np.meshgrid(pts, pts, indexing="ij")
#     w        = np.outer(wts, wts)
#     quad_ref = PSOQuadrature(xi.ravel(), eta.ravel(), w.ravel())
#
#     node_ids  = list(range(len(coords)))
#     mini_mesh = _MiniMesh(coords)
#
#     return AxisymmetricElement(
#         0, node_ids, material, shape_func, quad_ref
#     ).compute_element_stiffness(mini_mesh)
#
#
# class PSO:
#     def __init__(self,
#                  *,
#                  fitness_func,
#                  n_particles: int,
#                  dim: int,
#                  lo: np.ndarray,
#                  hi: np.ndarray,
#                  w: float = 0.7,
#                  c1: float = 1.4,
#                  c2: float = 1.4,
#                  max_iter: int = 200,
#                  patience: int = 20,
#                  tol: float = 1e-6,
#                  target_fitness: float | None = None,
#                  maximize: bool = True,
#                  rng=None):
#         """
#         :param maximize: якщо True — **максимізуємо** fitness
#         """
#         self.fitness        = fitness_func
#         self.maximize       = maximize
#         self.np             = int(n_particles)
#         self.dim            = int(dim)
#         self.lo             = np.asarray(lo, dtype=float)
#         self.hi             = np.asarray(hi, dtype=float)
#         self.w, self.c1, self.c2 = w, c1, c2
#         self.max_iter       = max_iter
#         self.patience       = patience
#         self.improve_thr    = tol
#         self.target_fitness = target_fitness
#         self.rng            = np.random.default_rng(rng)
#
#         self._init_swarm()
#
#     def _init_swarm(self):
#         span = self.hi - self.lo
#         self.swarm = []
#         for _ in range(self.np):
#             pos = self.rng.uniform(self.lo, self.hi)
#             vel = self.rng.uniform(-span, span)
#             p = Particle(pos, vel)
#             # безпечне ініціалізоване best_value
#             p.best_value = -np.inf if self.maximize else np.inf
#             self.swarm.append(p)
#         self.gbest_position = None
#         self.gbest_value    = -np.inf if self.maximize else np.inf
#
#     def optimize(self):
#         """
#         Цикл PSO з підтримкою maximize/minimize та ранньою зупинкою.
#         """
#         no_improve = 0
#         for _ in range(self.max_iter):
#             improved = False
#             for p in self.swarm:
#                 p.position = np.clip(p.position, self.lo, self.hi)
#                 val = self.fitness(p.position)
#
#                 # обираємо, як порівнювати
#                 better = (val > p.best_value) if self.maximize else (val < p.best_value)
#                 if better:
#                     p.best_value    = val
#                     p.best_position = p.position.copy()
#
#                 global_better = (val > self.gbest_value) if self.maximize else (val < self.gbest_value)
#                 if global_better:
#                     self.gbest_value    = val
#                     self.gbest_position = p.position.copy()
#                     improved = True
#                     if (self.target_fitness is not None
#                             and ((self.maximize and val >= self.target_fitness)
#                                  or (not self.maximize and val <= self.target_fitness))):
#                         return self.gbest_position, self.gbest_value
#
#             no_improve = 0 if improved else no_improve + 1
#             if no_improve >= self.patience:
#                 break
#
#             # оновлюємо швидкість і позицію
#             for p in self.swarm:
#                 r1 = self.rng.random(self.dim)
#                 r2 = self.rng.random(self.dim)
#                 cognitive = self.c1 * r1 * (p.best_position - p.position)
#                 social    = self.c2 * r2 * (self.gbest_position - p.position)
#                 p.velocity = self.w * p.velocity + cognitive + social
#                 p.position += p.velocity
#
#         return self.gbest_position, self.gbest_value
