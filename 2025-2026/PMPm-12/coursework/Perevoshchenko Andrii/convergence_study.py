import os
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.assembly import assemble_all
from src.basis import shape_gradients_physical
from src.mesh import TriangularMesh
from src.solver import Solver
from src.utils import h1_seminorm_error, l2_error, linf_error

OUTDIR = "results"
os.makedirs(OUTDIR, exist_ok=True)
N_WORKERS = min(os.cpu_count() or 4, 8)


def compute_rates(h_list, err_list):
    rates = [np.nan]
    for i in range(1, len(h_list)):
        if err_list[i] > 1e-16 and err_list[i - 1] > 1e-16:
            rates.append(
                np.log(err_list[i - 1] / err_list[i])
                / np.log(h_list[i - 1] / h_list[i])
            )
        else:
            rates.append(np.nan)
    return rates


def extract_at_coarse_nodes(u_fine, nx_fine, nx_coarse):
    ratio = nx_fine // nx_coarse
    nc1 = nx_coarse + 1
    nf1 = nx_fine + 1
    u_out = np.empty(nc1 * nc1)
    for i in range(nc1):
        for j in range(nc1):
            u_out[i * nc1 + j] = u_fine[(i * ratio) * nf1 + (j * ratio)]
    return u_out


def l2_error_nodal(mesh, u1, u2):
    err_sq = 0.0
    for e in range(mesh.n_elements):
        ids = mesh.elements[e]
        area = mesh.element_area(e)
        err_sq += (np.mean(u1[ids]) - np.mean(u2[ids])) ** 2 * area
    return np.sqrt(err_sq)


def h1_error_nodal(mesh, u1, u2):
    err_sq = 0.0
    for e in range(mesh.n_elements):
        ids = mesh.elements[e]
        coords = mesh.element_nodes(e)
        area = mesh.element_area(e)
        grad_phys, _ = shape_gradients_physical(coords)
        diff = grad_phys.T @ (u1[ids] - u2[ids])
        err_sq += np.dot(diff, diff) * area
    return np.sqrt(err_sq)


def _print_table(title, rows, cols):
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")
    hdr = "  ".join(f"{c:>10s}" for c in cols)
    print(hdr)
    print("─" * len(hdr))
    for row in rows:
        parts = []
        for v in row:
            if isinstance(v, int):
                parts.append(f"{v:>10d}")
            elif isinstance(v, float):
                if np.isnan(v):
                    parts.append(f"{'—':>10s}")
                elif abs(v) < 100 and abs(v) > 1e-2:
                    parts.append(f"{v:>10.4f}")
                else:
                    parts.append(f"{v:>10.2e}")
            else:
                parts.append(f"{str(v):>10s}")
        print("  ".join(parts))


def _w_stationary(nx):
    D = 1.0
    mesh = TriangularMesh(1.0, 1.0, nx, nx)
    M, K, C, _ = assemble_all(mesh, D, [0.0, 0.0])

    def f_func(x, y, _t):
        return 2.0 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    solver = Solver(mesh, M, K, C, dt=1.0, theta=1.0)
    u_h = solver.solve_stationary(
        f_func=f_func, g_func=lambda x, y, t: 0.0
    )

    def u_ex(x, y, _t):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def grad_ex(x, y, _t):
        return np.array([
            np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
            np.pi * np.sin(np.pi * x) * np.cos(np.pi * y),
        ])

    return dict(
        nx=nx, h=mesh.hmax(),
        l2=l2_error(mesh, u_h, u_ex),
        linf=linf_error(mesh, u_h, u_ex),
        h1=h1_seminorm_error(mesh, u_h, grad_ex),
    )


def _w_diffusion(nx):
    D = 0.01
    T = 0.5
    dt = T / nx

    mesh = TriangularMesh(1.0, 1.0, nx, nx)
    M, K, C, _ = assemble_all(mesh, D, [0.0, 0.0])
    u0 = np.array([
        np.sin(np.pi * x) * np.sin(np.pi * y)
        for x, y in mesh.nodes
    ])

    solver = Solver(mesh, M, K, C, dt, theta=0.5)
    sols, ts = solver.solve(u0, T, g_func=lambda x, y, t: 0.0,
                            store_every=10 ** 7)

    def u_ex(x, y, t):
        return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(
            -2 * np.pi ** 2 * D * t
        )

    def grad_ex(x, y, t):
        e = np.exp(-2 * np.pi ** 2 * D * t)
        return np.array([
            np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * e,
            np.pi * np.sin(np.pi * x) * np.cos(np.pi * y) * e,
        ])

    return dict(
        nx=nx, h=mesh.hmax(), dt=dt,
        l2=l2_error(mesh, sols[-1], u_ex, ts[-1]),
        linf=linf_error(mesh, sols[-1], u_ex, ts[-1]),
        h1=h1_seminorm_error(mesh, sols[-1], grad_ex, ts[-1]),
    )


def _w_advdiff(args):
    nx, D, a1, a2, T, theta = args
    a = [a1, a2]
    dt = T / (2 * nx)

    mesh = TriangularMesh(1.0, 1.0, nx, nx)
    M, K, C, _ = assemble_all(mesh, D, a)
    u0 = np.array([
        np.exp(-((x - 0.25) ** 2 + (y - 0.25) ** 2) / (2 * 0.08 ** 2))
        for x, y in mesh.nodes
    ])

    solver = Solver(mesh, M, K, C, dt, theta=theta)
    sols, ts = solver.solve(u0, T, g_func=lambda x, y, t: 0.0,
                            store_every=10 ** 7)

    return dict(nx=nx, h=mesh.hmax(), dt=dt, u=sols[-1], t=ts[-1])


def _w_temporal(args):
    dt, theta, nx, D, T = args

    mesh = TriangularMesh(1.0, 1.0, nx, nx)
    M, K, C, _ = assemble_all(mesh, D, [0.0, 0.0])
    u0 = np.array([
        np.sin(np.pi * x) * np.sin(np.pi * y)
        for x, y in mesh.nodes
    ])

    solver = Solver(mesh, M, K, C, dt, theta=theta)
    sols, ts = solver.solve(u0, T, g_func=lambda x, y, t: 0.0,
                            store_every=10 ** 7)

    return dict(dt=dt, theta=theta, u=sols[-1], t=ts[-1])


def _convergence_plot(h, errs, labels, ref_slopes, title, fname,
                      xlabel="h (розмір елемента)", ylabel="Похибка"):
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = "os^vDp"
    h = np.asarray(h, dtype=float)

    for i, (e, lab) in enumerate(zip(errs, labels)):
        e = np.asarray(e, dtype=float)
        ax.loglog(h, e, f"{markers[i % len(markers)]}-", label=lab,
                  linewidth=1.5, markersize=6)

    for p, style, slabel in ref_slopes:
        y0 = errs[0][0] * 1.5
        y_ref = y0 * (h / h[0]) ** p
        ax.loglog(h, y_ref, style, alpha=0.4, label=slabel)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUTDIR}/{fname}")


def study_1():
    mesh_sizes = [8, 16, 32, 64, 128]
    print("\n▶ Study 1: Stationary Poisson  −Δu = f,  u|∂Ω = 0")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        res = sorted(pool.map(_w_stationary, mesh_sizes),
                     key=lambda r: r["nx"])
    print(f"  Completed in {time.time() - t0:.1f}s ({N_WORKERS} workers)")

    h = [r["h"] for r in res]
    l2 = [r["l2"] for r in res]
    li = [r["linf"] for r in res]
    h1 = [r["h1"] for r in res]
    rl2, rli, rh1 = compute_rates(h, l2), compute_rates(h, li), compute_rates(h, h1)

    rows = []
    for i, r in enumerate(res):
        rows.append([r["nx"], r["h"], r["l2"], rl2[i],
                      r["linf"], rli[i], r["h1"], rh1[i]])
    _print_table("Study 1: −Δu = f  (exact solution)", rows,
                 ["nx", "h", "L2", "p(L2)", "L∞", "p(L∞)", "H1", "p(H1)"])

    _convergence_plot(
        h, [l2, li, h1],
        [f"L₂  (p ≈ {np.nanmean(rl2[1:]):.2f})",
         f"L∞  (p ≈ {np.nanmean(rli[1:]):.2f})",
         f"H¹  (p ≈ {np.nanmean(rh1[1:]):.2f})"],
        [(2, "k--", "O(h²)"), (1, "k:", "O(h)")],
        "Збіжність: стаціонарна задача Пуассона",
        "convergence_1_stationary.png",
    )
    return res


def study_2():
    mesh_sizes = [8, 16, 32, 64, 128]
    print("\n▶ Study 2: Non-stationary diffusion, Crank-Nicolson (Δt = T/nx)")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        res = sorted(pool.map(_w_diffusion, mesh_sizes),
                     key=lambda r: r["nx"])
    print(f"  Completed in {time.time() - t0:.1f}s")

    h = [r["h"] for r in res]
    l2 = [r["l2"] for r in res]
    li = [r["linf"] for r in res]
    h1 = [r["h1"] for r in res]
    rl2, rli, rh1 = compute_rates(h, l2), compute_rates(h, li), compute_rates(h, h1)

    rows = []
    for i, r in enumerate(res):
        rows.append([r["nx"], r["h"], r["dt"], r["l2"], rl2[i],
                      r["h1"], rh1[i]])
    _print_table("Study 2: diffusion  ∂u/∂t − D Δu = 0  (CN, exact sol.)", rows,
                 ["nx", "h", "Δt", "L2", "p(L2)", "H1", "p(H1)"])

    _convergence_plot(
        h, [l2, li, h1],
        [f"L₂  (p ≈ {np.nanmean(rl2[1:]):.2f})",
         f"L∞  (p ≈ {np.nanmean(rli[1:]):.2f})",
         f"H¹  (p ≈ {np.nanmean(rh1[1:]):.2f})"],
        [(2, "k--", "O(h²)"), (1, "k:", "O(h)")],
        "Збіжність: нестаціонарна дифузія (Кранка–Ніколсона)",
        "convergence_2_diffusion.png",
    )
    return res


def study_3():
    D, a1, a2, T, theta = 0.01, 1.0, 0.5, 0.4, 0.5
    mesh_sizes = [8, 16, 32, 64]
    nx_ref = 128
    all_nx = mesh_sizes + [nx_ref]

    print(f"\n▶ Study 3: Advection-diffusion, reference on {nx_ref}×{nx_ref}")

    args = [(nx, D, a1, a2, T, theta) for nx in all_nx]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        raw = list(pool.map(_w_advdiff, args))
    print(f"  Completed in {time.time() - t0:.1f}s")

    by_nx = {r["nx"]: r for r in raw}
    u_ref = by_nx[nx_ref]["u"]

    results = []
    for nx in mesh_sizes:
        u_coarse = by_nx[nx]["u"]
        u_ref_on_coarse = extract_at_coarse_nodes(u_ref, nx_ref, nx)
        mesh_c = TriangularMesh(1.0, 1.0, nx, nx)
        e_l2 = l2_error_nodal(mesh_c, u_coarse, u_ref_on_coarse)
        e_linf = np.max(np.abs(u_coarse - u_ref_on_coarse))
        e_h1 = h1_error_nodal(mesh_c, u_coarse, u_ref_on_coarse)
        results.append(dict(
            nx=nx, h=by_nx[nx]["h"], l2=e_l2, linf=e_linf, h1=e_h1
        ))

    h = [r["h"] for r in results]
    l2 = [r["l2"] for r in results]
    li = [r["linf"] for r in results]
    h1 = [r["h1"] for r in results]
    rl2, rli, rh1 = compute_rates(h, l2), compute_rates(h, li), compute_rates(h, h1)

    rows = []
    for i, r in enumerate(results):
        rows.append([r["nx"], r["h"], r["l2"], rl2[i],
                      r["linf"], rli[i], r["h1"], rh1[i]])
    _print_table(
        f"Study 3: advection-diffusion  (ref = {nx_ref}×{nx_ref})", rows,
        ["nx", "h", "L2", "p(L2)", "L∞", "p(L∞)", "H1", "p(H1)"]
    )

    _convergence_plot(
        h, [l2, li, h1],
        [f"L₂  (p ≈ {np.nanmean(rl2[1:]):.2f})",
         f"L∞  (p ≈ {np.nanmean(rli[1:]):.2f})",
         f"H¹  (p ≈ {np.nanmean(rh1[1:]):.2f})"],
        [(2, "k--", "O(h²)"), (1, "k:", "O(h)")],
        "Збіжність: адвекція-дифузія (відносно реф. розв'язку)",
        "convergence_3_advdiff.png",
    )
    return results


def study_4():
    D = 0.01
    T = 0.5
    nx = 32
    dt_list = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    dt_ref = T / 4000

    print(f"\n▶ Study 4: Temporal convergence (nx={nx}, ref Δt={dt_ref:.1e})")

    args_euler = [(dt, 1.0, nx, D, T) for dt in dt_list]
    args_cn    = [(dt, 0.5, nx, D, T) for dt in dt_list]
    args_ref   = [(dt_ref, 1.0, nx, D, T), (dt_ref, 0.5, nx, D, T)]
    all_args   = args_euler + args_cn + args_ref

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        all_res = list(pool.map(_w_temporal, all_args))
    print(f"  Completed in {time.time() - t0:.1f}s")

    ne = len(dt_list)
    res_euler = all_res[:ne]
    res_cn    = all_res[ne:2 * ne]
    ref_euler = all_res[2 * ne]
    ref_cn    = all_res[2 * ne + 1]

    res_euler.sort(key=lambda r: -r["dt"])
    res_cn.sort(key=lambda r: -r["dt"])

    mesh = TriangularMesh(1.0, 1.0, nx, nx)

    dt_e, l2_e = [], []
    for r in res_euler:
        dt_e.append(r["dt"])
        l2_e.append(l2_error_nodal(mesh, r["u"], ref_euler["u"]))

    dt_c, l2_c = [], []
    for r in res_cn:
        dt_c.append(r["dt"])
        l2_c.append(l2_error_nodal(mesh, r["u"], ref_cn["u"]))

    r_e = compute_rates(dt_e, l2_e)
    r_c = compute_rates(dt_c, l2_c)

    rows = []
    for i in range(len(dt_list)):
        rows.append([dt_e[i], l2_e[i], r_e[i], l2_c[i], r_c[i]])
    _print_table(
        f"Study 4: temporal convergence (nx={nx}, ref Δt={dt_ref:.1e})",
        rows,
        ["Δt", "L2(Euler)", "p(Euler)", "L2(CN)", "p(CN)"],
    )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(dt_e, l2_e, "o-",
              label=f"Ейлер  (p ≈ {np.nanmean(r_e[1:]):.2f})", linewidth=1.5)
    ax.loglog(dt_c, l2_c, "s-",
              label=f"CN  (p ≈ {np.nanmean(r_c[1:]):.2f})", linewidth=1.5)
    dt_arr = np.array(dt_e, dtype=float)
    ax.loglog(dt_arr, l2_e[0] * (dt_arr / dt_arr[0]) ** 1,
              "k--", alpha=0.4, label="O(Δt)")
    ax.loglog(dt_arr, l2_c[0] * (dt_arr / dt_arr[0]) ** 2,
              "k:", alpha=0.4, label="O(Δt²)")
    ax.set_xlabel("Δt")
    ax.set_ylabel("L₂ похибка (відн. реф.)")
    ax.set_title("Часова збіжність: неявний Ейлер vs Кранка–Ніколсона")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "convergence_4_temporal.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUTDIR}/convergence_4_temporal.png")

    res_e_out = [dict(dt=dt_e[i], l2=l2_e[i]) for i in range(len(dt_e))]
    res_c_out = [dict(dt=dt_c[i], l2=l2_c[i]) for i in range(len(dt_c))]
    return res_e_out, res_c_out


def summary_figure(r1, r2, r3, r4e, r4c):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    def _plot_spatial(ax, res, title, has_h1=True):
        h = [r["h"] for r in res]
        l2 = [r["l2"] for r in res]
        h1 = [r["h1"] for r in res] if has_h1 else None
        rl2 = compute_rates(h, l2)
        ax.loglog(h, l2, "o-", label=f"L₂ (p≈{np.nanmean(rl2[1:]):.2f})")
        if has_h1:
            rh1 = compute_rates(h, h1)
            ax.loglog(h, h1, "^-", label=f"H¹ (p≈{np.nanmean(rh1[1:]):.2f})")
        h_ref = np.array(h)
        ax.loglog(h_ref, l2[0] * (h_ref / h_ref[0]) ** 2,
                  "k--", alpha=0.3, label="O(h²)")
        ax.loglog(h_ref, (h1 or l2)[0] * (h_ref / h_ref[0]) ** 1,
                  "k:", alpha=0.3, label="O(h)")
        ax.set_xlabel("h")
        ax.set_ylabel("Похибка")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.2)

    _plot_spatial(axes[0, 0], r1, "(a) Стаціонарний Пуассон")
    _plot_spatial(axes[0, 1], r2, "(b) Дифузія (нестаціонарна, CN)")
    _plot_spatial(axes[1, 0], r3, "(c) Адвекція-дифузія (реф.)")

    ax = axes[1, 1]
    dt_e = [r["dt"] for r in r4e]
    l2_e = [r["l2"] for r in r4e]
    dt_c = [r["dt"] for r in r4c]
    l2_c = [r["l2"] for r in r4c]
    re = compute_rates(dt_e, l2_e)
    rc = compute_rates(dt_c, l2_c)
    ax.loglog(dt_e, l2_e, "o-", label=f"Ейлер (p≈{np.nanmean(re[1:]):.2f})")
    ax.loglog(dt_c, l2_c, "s-", label=f"CN (p≈{np.nanmean(rc[1:]):.2f})")
    dta = np.array(dt_e, dtype=float)
    ax.loglog(dta, l2_e[0] * (dta / dta[0]) ** 1, "k--", alpha=0.3, label="O(Δt)")
    ax.loglog(dta, l2_c[0] * (dta / dta[0]) ** 2, "k:", alpha=0.3, label="O(Δt²)")
    ax.set_xlabel("Δt")
    ax.set_ylabel("L₂ похибка")
    ax.set_title("(d) Часова збіжність", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)

    fig.suptitle("Аналіз збіжності МСЕ-солвера", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "convergence_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → {OUTDIR}/convergence_summary.png")


if __name__ == "__main__":
    print(f"Workers: {N_WORKERS}  (cpu_count={os.cpu_count()})")
    t_total = time.time()

    r1 = study_1()
    r2 = study_2()
    r3 = study_3()
    r4e, r4c = study_4()

    summary_figure(r1, r2, r3, r4e, r4c)

    print(f"\n{'═' * 72}")
    print(f"  Total time: {time.time() - t_total:.1f}s")
    print(f"{'═' * 72}")
