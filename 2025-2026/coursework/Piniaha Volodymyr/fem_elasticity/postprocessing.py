from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from .solver import element_values_to_nodes


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def displacement_components(d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ux = d[0::2]
    uy = d[1::2]
    mag = np.sqrt(ux ** 2 + uy ** 2)
    return ux, uy, mag


def save_text_report(path: str | Path, title: str, params: dict, result) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"Задача: {title}",
        "-" * 70,
        f"Кількість вузлів: {result.n_nodes}",
        f"Кількість елементів: {result.n_elements}",
        f"Кількість ступенів вільності: {result.n_dofs}",
        "",
        "Вхідні параметри:",
    ]

    for key, value in params.items():
        lines.append(f"{key}: {value}")

    lines.extend([
        "",
        "Основні результати:",
        f"Максимальне переміщення: {result.max_displacement:.6e} м",
        f"Максимальне |sigma_x|: {result.max_sigma_x:.6e} Па",
        f"Максимальне |sigma_y|: {result.max_sigma_y:.6e} Па",
        f"Максимальне |tau_xy|: {result.max_tau_xy:.6e} Па",
        f"Максимальне напруження за Мізесом: {result.max_von_mises:.6e} Па",
        f"Час розрахунку: {result.elapsed_time:.6f} с",
    ])

    p.write_text("\n".join(lines), encoding="utf-8")


def plot_mesh(mesh, path: str | Path, title: str = "Скінченно-елементна сітка") -> None:
    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.triplot(triang, linewidth=0.45)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_deformed_mesh(mesh, d: np.ndarray, path: str | Path, scale: float | None = None, title: str = "Деформована сітка") -> float:
    ux, uy, mag = displacement_components(d)
    max_dim = max(np.ptp(mesh.nodes[:, 0]), np.ptp(mesh.nodes[:, 1]))
    max_disp = max(float(np.max(mag)), 1e-30)

    if scale is None:
        scale = 0.08 * max_dim / max_disp

    deformed = mesh.nodes.copy()
    deformed[:, 0] += scale * ux
    deformed[:, 1] += scale * uy

    triang0 = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    triang1 = mtri.Triangulation(deformed[:, 0], deformed[:, 1], mesh.elements)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.triplot(triang0, color="0.75", linewidth=0.35, label="Початкова сітка")
    ax.triplot(triang1, linewidth=0.55, label="Деформована сітка")
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_title(f"{title} (масштаб переміщень: {scale:.2e})")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return scale


def plot_nodal_field(mesh, values: np.ndarray, path: str | Path, title: str, colorbar_label: str) -> None:
    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.tricontourf(triang, values, levels=18)
    ax.triplot(triang, color="k", linewidth=0.15, alpha=0.25)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_title(title)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_von_mises(mesh, result, path: str | Path, title: str = "Поле напружень за Мізесом") -> None:
    nodal_vm = element_values_to_nodes(len(mesh.nodes), mesh.elements, result.von_mises)
    plot_nodal_field(mesh, nodal_vm, path, title, "Напруження за Мізесом, Па")


def write_summary_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
