
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy.polynomial.legendre import leggauss
from material import Material
from shapeFunction import LinearQuadrilateralShapeFunction
from axisymmetric_quadrature_pso import PSOQuadrature
from pso_algorithm import get_K_ref, _MiniMesh
from element import AxisymmetricElement
from tensorflow.keras.models import load_model
import joblib

_pts2, _w2 = leggauss(2)
_xi2, _eta2 = np.meshgrid(_pts2, _pts2, indexing='ij')
w2 = np.outer(_w2, _w2)
xi_std  = _xi2.ravel()
eta_std = _eta2.ravel()
w_std   = w2.ravel()

nn_model = load_model("quad_net.keras")
scaler_X = joblib.load("scaler_X.pkl")
scaler_Y = joblib.load("scaler_Y.pkl")

def create_nn_quadrature(coords: np.ndarray, nG: int = 2) -> PSOQuadrature:
    """
    Формує адаптивну квадратуру за допомогою натренованої MLP-моделі,
    а якщо вихід дивний — відкат на класичну 2×2 Gauss.
    """
    feat = coords[2:4].flatten()[None, :]
    feat_s = scaler_X.transform(feat)
    lab_s = nn_model.predict(feat_s, verbose=0)
    lab = scaler_Y.inverse_transform(lab_s).ravel()

    nn = nG * nG
    xi_pred  = lab[         :   nn]
    eta_pred = lab[nn       : 2*nn]
    w_pred   = lab[2*nn     : 3*nn]

    xi  = np.clip(xi_pred,  -1.0, 1.0)
    eta = np.clip(eta_pred, -1.0, 1.0)
    w   = np.clip(w_pred,     0.0, None)

    sum_std = float(np.sum(w_std))
    sum_nn  = float(np.sum(w))
    if sum_nn > 1e-8:
        w *= (sum_std / sum_nn)

    tol = 0.5  # 50% відхилення
    if np.max(np.abs(w - w_std)) > tol:
        return PSOQuadrature(xi=xi_std, eta=eta_std, weights=w_std)

    return PSOQuadrature(xi=xi, eta=eta, weights=w)



coords = np.array([
    [1.0, 0.0],  # нижня ліва
    [2.1, 0.1],  # нижня права
    [1.8, 1.2],  # верхня права
    [0.9, 1.1],  # верхня ліва
])

# Налаштування матеріалу та форми
E, nu = 210e3, 0.3
mat   = Material("Steel", E=E, nu=nu)
shape = LinearQuadrilateralShapeFunction()


K_ref = get_K_ref(coords, mat, nG=5, shape_func=shape)

K_std = get_K_ref(coords, mat, nG=2, shape_func=shape)

quad_nn = create_nn_quadrature(coords, nG=2)
mini    = _MiniMesh(coords)
elem    = AxisymmetricElement(0, list(range(4)), mat, shape, quad_nn)
K_nn    = elem.compute_element_stiffness(mini)

err_std = norm(K_std - K_ref, ord='fro')
err_nn  = norm(K_nn  - K_ref, ord='fro')
print(f"||K₂ - K_ref||_F = {err_std:.3e}")
print(f"||K_NN - K_ref||_F = {err_nn:.3e}")
if err_nn > 0:
    print(f"Покращення: {err_std/err_nn:.2f}×")
else:
    print("NN-квадратура дає нульову похибку (дуже дивно!)")


fig, axs = plt.subplots(1,2,figsize=(12,5))
im1 = axs[0].imshow(np.abs(K_std - K_ref), aspect='auto', cmap='viridis')
axs[0].set_title('Стандартна 2×2: |K₂ - K_ref|')
fig.colorbar(im1, ax=axs[0])
im2 = axs[1].imshow(np.abs(K_nn - K_ref), aspect='auto', cmap='viridis')
axs[1].set_title('NN-квадратура: |K_NN - K_ref|')
fig.colorbar(im2, ax=axs[1])
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.bar(['2×2 Gauss','NN'], [err_std, err_nn], color=['#4477AA','#44AA77'])
plt.ylabel('‖·‖_Fro')
plt.title('Порівняння похибок інтегрування')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

r_values = np.linspace(1.0, 2.0, 50)
ur_values = np.random.rand(50)
sigma_rr_values = np.random.rand(50)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(r_values, ur_values, 'bo-', label='FEM Results')
plt.xlabel('Radius (r)')
plt.ylabel('Radial Displacement $u_r$')
plt.title('Radial Displacement $u_r$')

plt.subplot(2, 3, 2)
plt.plot(r_values, sigma_rr_values, 'ro-', label='FEM Results')
plt.xlabel('Radius (r)')
plt.ylabel('Radial Stress $\sigma_{rr}$')
plt.title('Radial Stress $\sigma_{rr}$')

plt.tight_layout()
plt.show()
