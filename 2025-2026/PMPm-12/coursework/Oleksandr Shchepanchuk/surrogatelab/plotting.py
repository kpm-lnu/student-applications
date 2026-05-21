"""Figure generation for the reaction-diffusion / adaptive-sampling study.

The module is split into Part-A helpers (physics of the model: ODE
solution, phase portrait, PDE surfaces, boundary traces, spatial phase)
and Part-B helpers (the sampling comparison: design locations, error
convergence, samples-to-tolerance, metric bars, surrogate surfaces).

Every function takes already-computed data plus an output path, draws one
figure and saves it.  Nothing here solves a PDE or fits a surrogate, so
the plots stay decoupled from the experiment driver.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D proj)

# ----------------------------------------------------------------------
# Shared style
# ----------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})

#: consistent colour / marker per sampler -- space-filling cool, adaptive warm
SAMPLER_STYLE: dict[str, dict] = {
    "Random":   dict(color="#8c8c8c", marker="o", ls="--"),
    "LHS":      dict(color="#1f77b4", marker="s", ls="--"),
    "Halton":   dict(color="#17becf", marker="^", ls="--"),
    "P-greedy": dict(color="#ff7f0e", marker="D", ls="-"),
    "f-greedy": dict(color="#9467bd", marker="v", ls="-"),
    "β-greedy(β=0.5)": dict(color="#bcbd22", marker="h", ls="-"),
    "MEPE":     dict(color="#d62728", marker="*", ls="-"),
    "EIGF":     dict(color="#e377c2", marker="X", ls="-"),
}


def _style(name: str) -> dict:
    return SAMPLER_STYLE.get(name, dict(color="black", marker="o", ls="-"))


def _save(fig, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ======================================================================
# PART A -- physics of the model
# ======================================================================


def plot_ode_solution(t, Y, path: str) -> str:
    """Time series ``y1(t), y2(t)`` of the ODE model."""
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(t, Y[0], color="#1f77b4", lw=2, label=r"$y_1(t)$ (ресурс)")
    ax.plot(t, Y[1], color="#d62728", lw=2, label=r"$y_2(t)$ (споживач)")
    ax.set_xlabel("$t$")
    ax.set_ylabel("популяція")
    ax.set_title("ЗДР-модель: чисельний розв'язок для базових параметрів")
    ax.legend()
    return _save(fig, path)


def plot_phase_portrait(problem, t, Y, path: str) -> str:
    """Phase portrait: vector field, trajectory and equilibria."""
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    y1_max = max(12.0, 1.25 * np.max(Y[0]))
    y2_max = max(12.0, 1.25 * np.max(Y[1]))
    g1, g2 = np.meshgrid(np.linspace(0, y1_max, 22),
                         np.linspace(0, y2_max, 22))
    u, v = problem.reaction(g1, g2, problem.p_base)
    speed = np.hypot(u, v) + 1e-12
    ax.quiver(g1, g2, u / speed, v / speed, speed,
              cmap="Blues", alpha=0.7, width=0.0032, pivot="mid")
    ax.plot(Y[0], Y[1], color="#d62728", lw=2, label="траєкторія")
    for (e1, e2) in problem.equilibria():
        ax.plot(e1, e2, "ko", ms=9, mfc="gold", mec="black", zorder=5)
        ax.annotate(f"({e1:.0f}, {e2:.0f})", (e1, e2),
                    textcoords="offset points", xytext=(8, 6))
    ax.plot(Y[0, 0], Y[1, 0], "k^", ms=9, mfc="white", label="старт $(2,4)$")
    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")
    ax.set_title("Фазовий портрет реакційної системи")
    ax.legend(loc="upper right")
    return _save(fig, path)


def plot_pde_surface(x, t, Y, which: int, path: str) -> str:
    """3-D surface ``y_i(x, t)`` of the PDE solution (which = 0 or 1)."""
    name = ["y_1", "y_2"][which]
    X, T = np.meshgrid(x, t, indexing="ij")
    sx = max(1, len(x) // 90)
    st = max(1, len(t) // 90)
    fig = plt.figure(figsize=(7.4, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X[::sx, ::st], T[::sx, ::st], Y[which, ::sx, ::st],
                    cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_zlabel(f"${name}$")
    ax.set_title(f"Поверхня розв'язку ДРЧП ${name}(x,t)$")
    ax.view_init(elev=26, azim=-128)
    return _save(fig, path)


def plot_pde_traces(x, t, Y, problem, path: str) -> str:
    """Boundary (``x = xe``) and mid-point (``x = 0``) time traces."""
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    mid1 = np.array([np.interp(0.0, x, Y[0, :, k]) for k in range(len(t))])
    mid2 = np.array([np.interp(0.0, x, Y[1, :, k]) for k in range(len(t))])
    ax.plot(t, Y[0, -1, :], color="#1f77b4", ls="--", label=r"$y_1(x_e,t)$")
    ax.plot(t, Y[1, -1, :], color="#d62728", ls="--", label=r"$y_2(x_e,t)$")
    ax.plot(t, mid1, color="#1f77b4", lw=2, label=r"$y_1(0,t)$")
    ax.plot(t, mid2, color="#d62728", lw=2, label=r"$y_2(0,t)$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("значення")
    ax.set_title("Часові ряди на межі $x=x_e$ та у середині $x=0$")
    ax.legend(ncol=2, fontsize=9)
    return _save(fig, path)


def plot_spatial_phase(omega: float, path: str) -> str:
    """Spatial phase portrait of the stationary equation (lecture case B).

    For t-independent solutions on the branch ``y1 = 0`` the stationary
    equation ``D2 y2'' + p4 y2 = 0`` is written as ``y2' = w,
    w' = -omega^2 y2`` with ``omega = sqrt(p4/D2)``.  The origin is a
    centre: trajectories in the ``(y2, w)`` plane are closed orbits, i.e.
    bounded oscillatory stationary patterns in space.
    """
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    lim = 6.0
    gy, gw = np.meshgrid(np.linspace(-lim, lim, 21),
                         np.linspace(-lim, lim, 21))
    u, v = gw, -(omega ** 2) * gy
    speed = np.hypot(u, v)
    ax.streamplot(gy, gw, u, v, color=speed, cmap="winter", density=1.0,
                  linewidth=0.8, arrowsize=0.9)
    for r in (1.5, 3.0, 4.5):                  # a few closed orbits
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(r * np.cos(th), r * omega * np.sin(th),
                color="#d62728", lw=1.6)
    ax.plot(0, 0, "o", color="black", ms=8)
    ax.annotate("центр (0, 0)", (0.3, 0.4))
    ax.set_xlabel("$y_2$")
    ax.set_ylabel("$w = y_2'$")
    ax.set_title("Просторовий фазовий портрет стаціонарного рівняння\n"
                 r"($y_2' = w,\ w' = -(p_4/D_2)\,y_2$)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    return _save(fig, path)


# ======================================================================
# PART B -- sampling comparison
# ======================================================================


def _qoi_background(ax, bg_X, bg_Z, per_dim, problem, log_scale=True):
    """Filled QoI contour used behind sample-location scatter plots."""
    g = bg_X[:, 0].reshape(per_dim, per_dim)
    h = bg_X[:, 1].reshape(per_dim, per_dim)
    Z = bg_Z.reshape(per_dim, per_dim)
    field = np.log10(Z) if log_scale else Z
    cf = ax.contourf(g, h, field, levels=18, cmap="cividis", alpha=0.92)
    return cf


def plot_design_panel(designs, samplers, bg_X, bg_Z, per_dim, problem,
                      qoi, n, path: str, seed: int = 0) -> str:
    """Grid of sample-location panels -- one per sampler -- over the QoI field.

    Shows *where* each strategy places its ``n`` training points; the
    background is ``log10`` of the QoI so feature-rich regions are visible.
    """
    ncol = 3
    nrow = int(np.ceil(len(samplers) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.9 * nrow),
                             squeeze=False, constrained_layout=True)
    lo, hi = problem.bounds[:, 0], problem.bounds[:, 1]
    cf = None
    for idx, (ax, sname) in enumerate(zip(axes.ravel(), samplers)):
        row, col = divmod(idx, ncol)
        cf = _qoi_background(ax, bg_X, bg_Z, per_dim, problem)
        X = designs.get((qoi, sname, seed, n))
        if X is not None:
            mu = problem.from_unit(X)
            st = _style(sname)
            ax.scatter(mu[:, 0], mu[:, 1], s=42, c=st["color"],
                       edgecolors="white", linewidths=0.7, zorder=3)
        ax.set_title(sname, fontsize=11)
        if row == nrow - 1:
            ax.set_xlabel(f"${problem.param_names[0]}$")
        if col == 0:
            ax.set_ylabel(f"${problem.param_names[1]}$")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.grid(False)
    for ax in axes.ravel()[len(samplers):]:
        ax.axis("off")
    fig.suptitle(f"Розміщення $n={n}$ точок кожним семплером  "
                 f"(фон: $\\log_{{10}}{qoi}$)", fontsize=12)
    if cf is not None:
        fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.85,
                     label=f"$\\log_{{10}}\\,{qoi}$")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_convergence(summary: pd.DataFrame, qoi: str, metric: str,
                     path: str, ylabel: str | None = None,
                     logy: bool = True) -> str:
    """Median error vs budget ``n`` with an inter-quartile band per sampler."""
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    sub = summary[summary["qoi"] == qoi]
    for sname in SAMPLER_STYLE:
        d = sub[sub["sampler"] == sname].sort_values("n")
        if d.empty:
            continue
        st = _style(sname)
        ax.plot(d["n"], d["median"], color=st["color"], marker=st["marker"],
                ls=st["ls"], lw=2, ms=6, label=sname)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("\u0440\u043e\u0437\u043c\u0456\u0440 \u0442\u0440\u0435\u043d\u0443\u0432\u0430\u043b\u044c\u043d\u043e\u0433\u043e \u043d\u0430\u0431\u043e\u0440\u0443 $n_s$")
    ax.set_ylabel(ylabel or metric.upper())
    ax.set_title(f"\u0417\u0431\u0456\u0436\u043d\u0456\u0441\u0442\u044c \u043f\u043e\u043c\u0438\u043b\u043a\u0438 \u0441\u0443\u0440\u043e\u0433\u0430\u0442\u0430 \u0434\u043b\u044f ${qoi}$")
    ax.legend(ncol=2, fontsize=9)
    return _save(fig, path)


def plot_samples_to_tol(stt: pd.DataFrame, qoi: str, tol: float, n_max: int,
                        path: str) -> str:
    """Bar chart of the median budget needed to reach the tolerance ``tau``."""
    sub = stt[stt["qoi"] == qoi].copy()
    order = [s for s in SAMPLER_STYLE if s in set(sub["sampler"])]
    sub = sub.set_index("sampler").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    xs = np.arange(len(sub))
    for i, row in sub.iterrows():
        st = _style(row["sampler"])
        med = row["n_star_median"]
        if np.isnan(med):
            ax.bar(i, n_max, color=st["color"], alpha=0.35, hatch="//")
            ax.text(i, n_max * 0.5, "цензуровано", rotation=90, ha="center",
                    va="center", fontsize=9)
        else:
            ax.bar(i, med, color=st["color"])
            ax.text(i, med + 0.6, f"{med:.0f}", ha="center", fontsize=9)
            if row["n_censored"]:
                ax.text(i, med + 3.0, f"({row['n_censored']} цнз.)",
                        ha="center", fontsize=7, color="grey")
    ax.set_xticks(xs)
    ax.set_xticklabels(sub["sampler"], rotation=15)
    ax.set_ylabel(r"кількість точок до досягнення толерантності $n^*(\tau)$")
    ax.set_title(f"Вартість досягнення NRMSE $\\leq \\tau={tol}$ для ${qoi}$ "
                 f"(менше — краще)")
    ax.set_axisbelow(True)
    return _save(fig, path)


def plot_metric_bars(results: pd.DataFrame, qoi: str, n: int, path: str) -> str:
    """Grouped bar chart of four quality metrics at a fixed budget ``n``."""
    sub = results[(results["qoi"] == qoi) & (results["n"] == n)]
    metrics = [("nrmse", "NRMSE"), ("mre", "MRE"),
               ("max_re", "MAX_RE"), ("r2", "$R^2$")]
    order = [s for s in SAMPLER_STYLE if s in set(sub["sampler"])]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    for ax, (key, label) in zip(axes.ravel(), metrics):
        med = [np.nanmedian(sub[sub["sampler"] == s][key]) for s in order]
        colors = [_style(s)["color"] for s in order]
        ax.bar(np.arange(len(order)), med, color=colors)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=20, fontsize=8)
        ax.set_title(label)
        ax.set_axisbelow(True)
    fig.suptitle(f"Метрики якості для ${qoi}$ при $n_s={n}$", fontsize=12)
    return _save(fig, path)


def plot_field_comparison(bg_X, true_Z, pred_Z, per_dim, problem, qoi,
                          design_X, path: str, sampler: str = "") -> str:
    """Three-panel comparison of the surrogate against the true QoI field.

    Left: the true functional over the parameter plane.  Middle: the RBF
    surrogate's prediction, with the training points overlaid.  Right: the
    absolute error field ``|surrogate - true|``.  Together they show *where*
    the surrogate reproduces the real solution field and where it departs.
    """
    mu = problem.from_unit(bg_X)
    g = mu[:, 0].reshape(per_dim, per_dim)
    h = mu[:, 1].reshape(per_dim, per_dim)
    T = true_Z.reshape(per_dim, per_dim)
    P = pred_Z.reshape(per_dim, per_dim)
    E = np.abs(P - T)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), constrained_layout=True)
    lo, hi = problem.bounds[:, 0], problem.bounds[:, 1]
    vmin, vmax = float(min(T.min(), P.min())), float(max(T.max(), P.max()))

    c0 = axes[0].contourf(g, h, T, levels=18, cmap="viridis",
                          vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Точне поле ${qoi}({problem.param_names[0]},"
                      f"{problem.param_names[1]})$")
    fig.colorbar(c0, ax=axes[0], shrink=0.9)

    c1 = axes[1].contourf(g, h, P, levels=18, cmap="viridis",
                          vmin=vmin, vmax=vmax)
    if design_X is not None:
        d = problem.from_unit(design_X)
        axes[1].scatter(d[:, 0], d[:, 1], s=26, c="white",
                        edgecolors="black", linewidths=0.6, zorder=3)
    ttl = "RBF-сурогат" + (f" ({sampler})" if sampler else "")
    axes[1].set_title(ttl + " + тренувальні точки")
    fig.colorbar(c1, ax=axes[1], shrink=0.9)

    c2 = axes[2].contourf(g, h, E, levels=18, cmap="magma")
    axes[2].set_title(f"Абсолютна похибка $|\\hat{{{qoi}}}-{qoi}|$")
    fig.colorbar(c2, ax=axes[2], shrink=0.9)

    for ax in axes:
        ax.set_xlabel(f"${problem.param_names[0]}$")
        ax.set_ylabel(f"${problem.param_names[1]}$")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
    fig.suptitle(f"Сурогат vs точне поле для ${qoi}$", fontsize=12)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_qoi_surface(bg_X, bg_Z, per_dim, problem, qoi, path: str,
                     pred_Z=None) -> str:
    """True QoI surface, optionally beside the surrogate prediction."""
    g = problem.from_unit(bg_X)[:, 0].reshape(per_dim, per_dim)
    h = problem.from_unit(bg_X)[:, 1].reshape(per_dim, per_dim)
    Z = bg_Z.reshape(per_dim, per_dim)
    ncol = 1 if pred_Z is None else 2
    fig = plt.figure(figsize=(6.6 * ncol, 5.0))
    ax = fig.add_subplot(1, ncol, 1, projection="3d")
    ax.plot_surface(g, h, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel(f"${problem.param_names[0]}$")
    ax.set_ylabel(f"${problem.param_names[1]}$")
    ax.set_zlabel(f"${qoi}$")
    ax.set_title(f"Точний функціонал ${qoi}({problem.param_names[0]},"
                 f"{problem.param_names[1]})$")
    if pred_Z is not None:
        ax2 = fig.add_subplot(1, ncol, 2, projection="3d")
        ax2.plot_surface(g, h, pred_Z.reshape(per_dim, per_dim),
                         cmap="plasma", linewidth=0, antialiased=True)
        ax2.set_xlabel(f"${problem.param_names[0]}$")
        ax2.set_ylabel(f"${problem.param_names[1]}$")
        ax2.set_zlabel(f"${qoi}$")
        ax2.set_title("Передбачення RBF-сурогата")
    return _save(fig, path)


def plot_pred_vs_exact(y_true, y_pred_by_sampler: dict, qoi: str,
                       path: str) -> str:
    """Predicted-vs-exact scatter for several samplers on the test set."""
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    lo = min(np.min(y_true), *[np.min(v) for v in y_pred_by_sampler.values()])
    hi = max(np.max(y_true), *[np.max(v) for v in y_pred_by_sampler.values()])
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ідеал")
    for sname, yp in y_pred_by_sampler.items():
        st = _style(sname)
        ax.scatter(y_true, yp, s=22, color=st["color"], alpha=0.7,
                   marker=st["marker"], label=sname)
    ax.set_xlabel(f"точне ${qoi}$")
    ax.set_ylabel(f"сурогатне ${qoi}$")
    ax.set_title(f"Передбачене vs точне ${qoi}$ на тестовому наборі")
    ax.legend(fontsize=9)
    ax.set_aspect("equal", "box")
    return _save(fig, path)


def plot_kernel_comparison(summary: pd.DataFrame, qoi: str, path: str) -> str:
    """Convergence of the cubic vs Gaussian surrogate kernel (LHS designs)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    palette = {"gaussian": "#d62728", "cubic": "#1f77b4"}
    for kern in ("gaussian", "cubic"):
        d = summary[(summary["qoi"] == qoi) &
                    (summary["kernel"] == kern)].sort_values("n")
        if d.empty:
            continue
        ax.plot(d["n"], d["median"], "o-", color=palette[kern], lw=2,
                label=f"{kern} RBF")
    ax.set_yscale("log")
    ax.set_xlabel("розмір тренувального набору $n_s$")
    ax.set_ylabel("NRMSE")
    ax.set_title(f"Порівняння ядер сурогата для ${qoi}$ (LHS-дизайн)")
    ax.legend()
    return _save(fig, path)


def plot_clustering_diagnostic(traj: dict, path: str, qoi: str) -> str:
    """Two-panel diagnostic of the f-greedy failure mode.

    ``traj`` maps a sampler name to a list of ``(n, min_gap, cond)`` tuples.
    Left panel: smallest pairwise distance in the design vs ``n`` (a
    clustering indicator).  Right panel: kernel-matrix condition number
    vs ``n``.  Together they expose the over-clustering spiral.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    for sname, rows in traj.items():
        st = _style(sname)
        ns = [r[0] for r in rows]
        axL.plot(ns, [r[1] for r in rows], color=st["color"],
                 marker=st["marker"], lw=2, label=sname)
        axR.plot(ns, [r[2] for r in rows], color=st["color"],
                 marker=st["marker"], lw=2, label=sname)
    axL.set_yscale("log")
    axL.set_xlabel("розмір тренувального набору $n_s$")
    axL.set_ylabel("мін. парна відстань у дизайні")
    axL.set_title("Розрідженість дизайну (менше = кластеризація)")
    axL.legend(fontsize=9)
    axR.set_yscale("log")
    axR.set_xlabel("розмір тренувального набору $n_s$")
    axR.set_ylabel(r"$\mathrm{cond}(\Phi)$")
    axR.set_title("Обумовленість ядрової матриці")
    axR.legend(fontsize=9)
    fig.suptitle(f"Чому f-greedy нестабільний ({qoi}): кластеризація "
                 f"спричиняє погану обумовленість", fontsize=12)
    return _save(fig, path)


def plot_overall_summary(summary_json: dict, path: str) -> str:
    """Coursework summary figure: samples-to-tolerance per (problem, sampler).

    One panel per problem; the bar height is the median ``n*`` for the primary
    QoI. A shared legend below identifies the samplers; missing/censored
    samplers are shown as hatched bars at the panel's ``n_max``.
    """
    problems = list(summary_json["problems"])
    samplers = list(summary_json["samplers"])
    samples_per_problem = summary_json["samples_to_tolerance"]
    qois_per_problem = summary_json["qois"]
    tolerances = summary_json.get("tolerances", {})

    ncol = min(3, len(problems))
    nrow = int(np.ceil(len(problems) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.8 * ncol, 3.6 * nrow),
        squeeze=False, constrained_layout=True,
    )

    for index, problem in enumerate(problems):
        ax = axes[index // ncol][index % ncol]
        qoi = qois_per_problem[problem][0]
        per_sampler = samples_per_problem[problem].get(qoi, {})
        cap = max(
            (value for value in per_sampler.values() if value is not None),
            default=1.0,
        )
        for bar_index, sampler in enumerate(samplers):
            style = _style(sampler)
            value = per_sampler.get(sampler)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                ax.bar(bar_index, cap, color=style["color"], alpha=0.35, hatch="//")
                ax.text(bar_index, cap * 0.5, "цензуровано", rotation=90,
                        ha="center", va="center", fontsize=8)
            else:
                ax.bar(bar_index, value, color=style["color"])
                ax.text(bar_index, value, f"{value:.0f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(len(samplers)))
        ax.set_xticklabels(samplers, rotation=30, ha="right", fontsize=8)
        tol = tolerances.get(problem)
        title = f"{problem} (${qoi}$)"
        if tol is not None:
            title += f", $\\tau = {tol}$"
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("$n^*(\\tau)$")
        ax.set_axisbelow(True)

    for ax in axes.ravel()[len(problems):]:
        ax.axis("off")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=_style(sampler)["color"], label=sampler)
        for sampler in samplers
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=min(len(samplers), 4), bbox_to_anchor=(0.5, -0.04), fontsize=9,
    )
    fig.suptitle("Точки до досягнення толерантності за задачами "
                 "(менше — краще)", fontsize=12)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_adaptive_growth(snapshots: list[tuple[int, np.ndarray]],
                         bg_X, bg_Z, per_dim, problem, qoi,
                         sampler_name, path: str) -> str:
    """How an adaptive sampler grows its design: several budget snapshots.

    White points are already in the design, the coloured point is the most
    recent addition for that snapshot.
    """
    k = len(snapshots)
    fig, axes = plt.subplots(1, k, figsize=(3.5 * k, 3.7), squeeze=False,
                             constrained_layout=True)
    lo, hi = problem.bounds[:, 0], problem.bounds[:, 1]
    for j, (ax, (n, X)) in enumerate(zip(axes[0], snapshots)):
        _qoi_background(ax, bg_X, bg_Z, per_dim, problem)
        mu = problem.from_unit(X)
        ax.scatter(mu[:-1, 0], mu[:-1, 1], s=28, c="white",
                   edgecolors="black", linewidths=0.6, zorder=3)
        ax.scatter(mu[-1, 0], mu[-1, 1], s=90, c=_style(sampler_name)["color"],
                   edgecolors="white", linewidths=1.0, marker="*", zorder=4)
        ax.set_title(f"$n = {n}$")
        ax.set_xlabel(f"${problem.param_names[0]}$")
        if j == 0:
            ax.set_ylabel(f"${problem.param_names[1]}$")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.grid(False)
    fig.suptitle(f"{sampler_name}: \u0440\u0456\u0441\u0442 \u0434\u0438\u0437\u0430\u0439\u043d\u0443 \u043d\u0430 $\\log_{{10}}{qoi}$ "
                 f"(\u2605 = \u043d\u0430\u0439\u043d\u043e\u0432\u0456\u0448\u0430 \u0442\u043e\u0447\u043a\u0430)", fontsize=12)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
