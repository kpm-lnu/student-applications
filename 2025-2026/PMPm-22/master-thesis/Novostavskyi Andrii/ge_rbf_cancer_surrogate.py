import numpy as np
from scipy.optimize import minimize

params = {
	"alpha1": 0.7, "mu1": 0.2, "phi1": 1.0,
	"alpha2": 0.9, "mu2": 0.5, "gamma1": 0.4,
	"s": 0.5, "rho": 0.5, "omega": 0.3,
	"gamma2": 0.29, "mu3": 0.3,
	"e1": 0.4, "e2": 0.3, "e3": 0.3,
	"Pi": 0.5, "theta": 1.0, "eta_g": 0.1,
}
g0 = 0.3
A_obj, B_obj, C_obj = 5.0, 1.5, 1.5
x0_state = np.array([1.0, 1e-5, 1.15, 0.5])
T_end, n_steps = 26.0, 500
t_grid = np.linspace(0.0, T_end, n_steps + 1)
dt = t_grid[1] - t_grid[0]

D = 12
N_SEG = 6
_seg_edges = np.linspace(0, n_steps + 1, N_SEG + 1, dtype=int)


def w_to_controls(w):
	u1, u2 = np.empty(n_steps + 1), np.empty(n_steps + 1)
	for k in range(N_SEG):
		sl = slice(_seg_edges[k], _seg_edges[k + 1])
		u1[sl] = np.clip(w[k], 0.0, 1.0)
		u2[sl] = np.clip(w[k + N_SEG], 0.0, 1.0)
	return u1, u2


def _f(x, u1v, u2v):
	N, T, M, E = x
	p = params
	dN = (N * (p["alpha1"] - p["mu1"] * N - p["phi1"] * T)
		  - p["eta_g"] * g0 * (1 - u1v) * N
		  - (1 - u2v) * p["e1"] * N * E)
	dT = (T * (p["alpha2"] - p["mu2"] * T)
		  - p["gamma1"] * M * T
		  + g0 * (1 - u1v) * T
		  + (1 - u2v) * p["e2"] * N * E)
	dM = (p["s"] + p["rho"] * M * T / (p["omega"] + T)
		  - p["gamma2"] * M * T
		  - g0 * (1 - u1v) * M
		  - p["mu3"] * M
		  - (1 - u2v) * p["e3"] * M * E)
	dE = (1 - u2v) * p["Pi"] - p["theta"] * E
	return np.array([dN, dT, dM, dE])


def _lam_dot(lam, x, u1v, u2v):
	N, T, M, E = x
	l1, l2, l3, l4 = lam
	p = params
	dl1 = (l1 * (2 * p["mu1"] * N - p["alpha1"] + p["phi1"] * T
				 + p["eta_g"] * g0 * (1 - u1v)
				 + (1 - u2v) * p["e1"] * E)
		   - l2 * (1 - u2v) * p["e2"] * E)
	dl2 = (l1 * p["phi1"] * N
		   - A_obj * T
		   - l2 * (p["alpha2"] - 2 * p["mu2"] * T
				   - p["gamma1"] * M + g0 * (1 - u1v))
		   - l3 * (p["rho"] * M * p["omega"] / (p["omega"] + T) ** 2
				   - p["gamma2"] * M))
	dl3 = (l2 * p["gamma1"] * T
		   - l3 * (p["rho"] * T / (p["omega"] + T)
				   - p["gamma2"] * T
				   - g0 * (1 - u1v)
				   - p["mu3"]
				   - (1 - u2v) * p["e3"] * E))
	dl4 = (l1 * (1 - u2v) * p["e1"] * N
		   - l2 * (1 - u2v) * p["e2"] * N
		   + l3 * (1 - u2v) * p["e3"] * M
		   + l4 * p["theta"])
	return np.array([dl1, dl2, dl3, dl4])


def forward_sweep(u1, u2):
	X = np.zeros((4, n_steps + 1))
	X[:, 0] = x0_state
	for i in range(n_steps):
		xi = X[:, i]
		a, b = u1[i], u1[i + 1]
		am = 0.5 * (a + b)
		c, d = u2[i], u2[i + 1]
		cm = 0.5 * (c + d)
		k1 = _f(xi,                  a,  c)
		k2 = _f(xi + 0.5 * dt * k1, am, cm)
		k3 = _f(xi + 0.5 * dt * k2, am, cm)
		k4 = _f(xi + dt * k3,        b,  d)
		X[:, i + 1] = np.maximum(
			xi + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
	return X


def backward_sweep(X, u1, u2):
	L = np.zeros((4, n_steps + 1))
	for i in range(n_steps, 0, -1):
		li = L[:, i]
		xa, xb = X[:, i], X[:, i - 1]
		xm = 0.5 * (xa + xb)
		a, b = u1[i], u1[i - 1]
		am = 0.5 * (a + b)
		c, d = u2[i], u2[i - 1]
		cm = 0.5 * (c + d)
		k1 = _lam_dot(li,                  xa, a,  c)
		k2 = _lam_dot(li - 0.5 * dt * k1, xm, am, cm)
		k3 = _lam_dot(li - 0.5 * dt * k2, xm, am, cm)
		k4 = _lam_dot(li - dt * k3,        xb, b,  d)
		L[:, i - 1] = li - (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
	return L


def _compute_J(X, u1, u2):
	return float(np.trapezoid(
		0.5 * A_obj * X[1] ** 2
		+ 0.5 * B_obj * u1 ** 2
		+ 0.5 * C_obj * u2 ** 2,
		t_grid))


class OracleCounter:

	def __init__(self):
		self.n_calls = 0

	def J_and_grad(self, w):
		self.n_calls += 1
		u1, u2 = w_to_controls(w)
		X = forward_sweep(u1, u2)
		L = backward_sweep(X, u1, u2)
		J = _compute_J(X, u1, u2)
		N_s, T_s, M_s, E_s = X
		l1, l2, l3, l4 = L
		p = params
		dJdu1 = (B_obj * u1
				 + l1 * p["eta_g"] * g0 * N_s
				 - l2 * g0 * T_s
				 + l3 * g0 * M_s)
		dJdu2 = (C_obj * u2
				 + l1 * p["e1"] * N_s * E_s
				 - l2 * p["e2"] * N_s * E_s
				 + l3 * p["e3"] * M_s * E_s
				 - l4 * p["Pi"])
		grad = np.zeros(D)
		for k in range(N_SEG):
			sl = slice(_seg_edges[k], _seg_edges[k + 1])
			grad[k]         = float(np.trapezoid(dJdu1[sl], t_grid[sl]))
			grad[k + N_SEG] = float(np.trapezoid(dJdu2[sl], t_grid[sl]))
		return J, grad


if __name__ == "__main__":

	print("=" * 62)
	print("STEP 1  FBSM direct reference optimum")
	print("=" * 62)
	u1_f = np.full(n_steps + 1, 0.5)
	u2_f = np.full(n_steps + 1, 0.5)
	for it in range(200):
		X_f = forward_sweep(u1_f, u2_f)
		L_f = backward_sweep(X_f, u1_f, u2_f)
		N_s, T_s, M_s, E_s = X_f
		l1, l2, l3, l4 = L_f
		p = params
		u1_new = np.clip(
			(l2 * g0 * T_s - l1 * p["eta_g"] * g0 * N_s
			 - l3 * g0 * M_s) / B_obj, 0.0, 1.0)
		u2_new = np.clip(
			(-l1 * p["e1"] * N_s * E_s + l2 * p["e2"] * N_s * E_s
			 - l3 * p["e3"] * M_s * E_s + l4 * p["Pi"]) / C_obj,
			0.0, 1.0)
		u1_nx = 0.5 * u1_f + 0.5 * u1_new
		u2_nx = 0.5 * u2_f + 0.5 * u2_new
		err = max(np.linalg.norm(u1_nx - u1_f, np.inf),
				  np.linalg.norm(u2_nx - u2_f, np.inf))
		u1_f, u2_f = u1_nx, u2_nx
		if err < 1e-5:
			print(f"  FBSM converged in {it + 1} iters  (err={err:.2e})")
			break
	J_direct = _compute_J(forward_sweep(u1_f, u2_f), u1_f, u2_f)
	print(f"  J_direct = {J_direct:.6f}")
	w_fbsm = np.zeros(D)
	for k in range(N_SEG):
		sl = slice(_seg_edges[k], _seg_edges[k + 1])
		w_fbsm[k]         = float(np.mean(u1_f[sl]))
		w_fbsm[k + N_SEG] = float(np.mean(u2_f[sl]))
	w_fbsm = np.clip(w_fbsm, 0.0, 1.0)

	print("\n" + "=" * 62)
	print("STEP 2  Sampling training data (LHS, N=50)")
	print("=" * 62)
	N_TRAIN = 50
	oracle  = OracleCounter()
	rng     = np.random.default_rng(42)
	perm    = np.stack([rng.permutation(N_TRAIN) for _ in range(D)], axis=1)
	W       = (perm + rng.random((N_TRAIN, D))) / N_TRAIN
	W[0]    = w_fbsm
	for i in range(1, 5):
		W[i] = np.clip(w_fbsm + rng.normal(0.0, 0.05, D), 0.01, 0.99)
	J_vals = np.zeros(N_TRAIN)
	G_vals = np.zeros((N_TRAIN, D))
	for i in range(N_TRAIN):
		J_vals[i], G_vals[i] = oracle.J_and_grad(W[i])
		if (i + 1) % 10 == 0:
			print(f"  sampled {i + 1}/{N_TRAIN}")
	print(f"  Oracle calls : {oracle.n_calls}")
	print(f"  J in [{J_vals.min():.4f}, {J_vals.max():.4f}]")

	J_mean = J_vals.mean()
	J_std  = J_vals.std() + 1e-14
	J_norm = (J_vals - J_mean) / J_std
	G_norm = G_vals / J_std

	print("\n" + "=" * 62)
	print("STEP 3  Hermite GE-RBF kernel (Gaussian, median-gamma)")
	print("=" * 62)
	delta = W[:, None, :] - W[None, :, :]
	r2    = (delta ** 2).sum(axis=2)
	gamma = 1.0 / (2.0 * float(np.median(r2[r2 > 0])))
	print(f"  gamma = {gamma:.6f}")
	sz  = N_TRAIN * (D + 1)
	print(f"  Matrix size: {sz} x {sz}")
	phi = np.exp(-gamma * r2)
	K   = np.zeros((sz, sz))
	K[:N_TRAIN, :N_TRAIN] = phi
	K[:N_TRAIN, N_TRAIN:] = (
		2.0 * gamma * delta * phi[:, :, None]
	).reshape(N_TRAIN, N_TRAIN * D)
	K[N_TRAIN:, :N_TRAIN] = (
		-2.0 * gamma * delta * phi[:, :, None]
	).transpose(0, 2, 1).reshape(N_TRAIN * D, N_TRAIN)
	outer   = np.einsum("ijk,ijl->ijkl", delta, delta)
	K_gg_4d = phi[:, :, None, None] * (
		2.0 * gamma * np.eye(D)[None, None]
		- 4.0 * gamma ** 2 * outer)
	K[N_TRAIN:, N_TRAIN:] = (
		K_gg_4d.transpose(0, 2, 1, 3).reshape(N_TRAIN * D, N_TRAIN * D))
	nugget = max(1e-8 * np.trace(K) / sz, 1e-10)
	K_reg  = K + nugget * np.eye(sz)
	print(f"  nugget = {nugget:.3e}")
	b_rhs = np.empty(sz)
	b_rhs[:N_TRAIN] = J_norm
	for i in range(N_TRAIN):
		b_rhs[N_TRAIN + i * D: N_TRAIN + (i + 1) * D] = G_norm[i]
	alpha_sol, _, rank, sv = np.linalg.lstsq(K_reg, b_rhs, rcond=None)
	cond = sv[0] / (sv[-1] + 1e-300)
	res  = float(np.linalg.norm(K_reg @ alpha_sol - b_rhs))
	print(f"  lstsq: rank={rank},  cond~{cond:.3e},  |res|={res:.3e}")
	alpha_v = alpha_sol[:N_TRAIN]
	beta_m  = alpha_sol[N_TRAIN:].reshape(N_TRAIN, D)

	def _surrogate(w_star):
		diff    = w_star[None, :] - W
		r2_s    = (diff ** 2).sum(axis=1)
		phi_s   = np.exp(-gamma * r2_s)
		b_dot_d = (beta_m * diff).sum(axis=1)
		s_val   = float((alpha_v * phi_s).sum()
						+ 2.0 * gamma * (phi_s * b_dot_d).sum())
		common  = alpha_v + 2.0 * gamma * b_dot_d
		ds_dw   = 2.0 * gamma * (
			phi_s @ beta_m
			- np.einsum("j,j,jl->l", phi_s, common, diff))
		return J_mean + J_std * s_val, J_std * ds_dw

	print("\n" + "=" * 62)
	print("STEP 4  L-BFGS-B surrogate minimisation (20 starts)")
	print("=" * 62)
	bounds = [(0.0, 1.0)] * D
	rng2   = np.random.default_rng(7)
	starts = ([w_fbsm.copy()]
			  + [rng2.uniform(0.0, 1.0, D) for _ in range(19)])
	best_J, best_w = np.inf, None
	for ws in starts:
		res = minimize(
			_surrogate, ws, method="L-BFGS-B", jac=True,
			bounds=bounds,
			options={"maxiter": 600, "ftol": 1e-15, "gtol": 1e-9})
		if res.fun < best_J:
			best_J, best_w = res.fun, res.x.copy()
	print(f"  Surrogate minimum  J_surr = {best_J:.6f}")

	print("\n" + "=" * 62)
	print("STEP 5  Validation")
	print("=" * 62)
	J_true, _ = oracle.J_and_grad(best_w)
	rel_err   = abs(J_true - J_direct) / (abs(J_direct) + 1e-14)
	print(f"  J_direct  (FBSM)          = {J_direct:.6f}")
	print(f"  J_surr_min (surrogate)    = {best_J:.6f}")
	print(f"  J_true @ w_surr (oracle)  = {J_true:.6f}")
	print(f"\n  Relative error = {rel_err:.4%}")
	if rel_err < 0.05:
		print("  RESULT: PASSED  (< 5% relative error)")
	else:
		print("  RESULT: WARNING -- relative error >= 5%")
	print(f"\n  Total oracle calls : {oracle.n_calls}")
	print(f"  u1 segments : {np.round(best_w[:N_SEG], 4)}")
	print(f"  u2 segments : {np.round(best_w[N_SEG:], 4)}")
