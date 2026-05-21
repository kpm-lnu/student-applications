from pathlib import Path
from fem_elasticity.postprocessing import ensure_dir, write_summary_csv
from examples.cantilever_beam import run as run_beam


def run(output_dir: str | Path = "results/mesh_convergence") -> dict:
    out = ensure_dir(output_dir)

    cases = [
        ("Груба сітка", 18, 5),
        ("Середня сітка", 34, 8),
        ("Густа сітка", 54, 12),
    ]

    rows = []
    print("\n" + "=" * 72)
    print("АНАЛІЗ ВПЛИВУ ГУСТИНИ СІТКИ НА РЕЗУЛЬТАТИ")
    print("=" * 72)

    for name, nx, ny in cases:
        case_dir = out / name.replace(" ", "_").lower()
        res = run_beam(case_dir, nx=nx, ny=ny)
        row = {
            "Тип сітки": name,
            "nx": nx,
            "ny": ny,
            "Вузли": res["Вузли"],
            "Елементи": res["Елементи"],
            "Максимальне переміщення, м": res["Максимальне переміщення, м"],
            "Максимальне напруження за Мізесом, Па": res["Максимальне напруження за Мізесом, Па"],
            "Час, с": res["Час, с"],
        }
        rows.append(row)

    write_summary_csv(out / "zalezhnist_vid_sitky.csv", rows)

    print("\nПідсумкова таблиця збіжності:")
    for row in rows:
        print(
            f"{row['Тип сітки']}: елементів={row['Елементи']}, "
            f"u_max={row['Максимальне переміщення, м']}, "
            f"sigma_vM_max={row['Максимальне напруження за Мізесом, Па']}"
        )

    return {
        "Назва задачі": "Вплив густини сітки",
        "Вузли": rows[-1]["Вузли"],
        "Елементи": rows[-1]["Елементи"],
        "Максимальне переміщення, м": rows[-1]["Максимальне переміщення, м"],
        "Максимальне напруження за Мізесом, Па": rows[-1]["Максимальне напруження за Мізесом, Па"],
        "Час, с": rows[-1]["Час, с"],
        "Додатково": "Таблиця збіжності збережена",
    }


if __name__ == "__main__":
    run()
