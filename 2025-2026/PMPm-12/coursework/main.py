import argparse
from pathlib import Path

from fem_elasticity.postprocessing import write_summary_csv
from examples.plate_with_hole import run as run_plate
from examples.cantilever_beam import run as run_beam
from examples.pressure_ring import run as run_ring
from examples.mesh_convergence import run as run_convergence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Розв'язування двовимірних задач теорії пружності методом скінченних елементів."
    )
    parser.add_argument(
        "--example",
        choices=["plate", "beam", "ring", "convergence", "all"],
        default="all",
        help="Приклад для запуску: plate, beam, ring, convergence або all."
    )
    args = parser.parse_args()

    rows = []

    if args.example in ("plate", "all"):
        rows.append(run_plate())

    if args.example in ("beam", "all"):
        rows.append(run_beam())

    if args.example in ("ring", "all"):
        rows.append(run_ring())

    if args.example in ("convergence", "all"):
        rows.append(run_convergence())

    if rows:
        out = Path("results")
        out.mkdir(exist_ok=True)
        write_summary_csv(out / "summary.csv", rows)

        print("\n" + "=" * 72)
        print("ЗАГАЛЬНЕ ПОРІВНЯННЯ РЕЗУЛЬТАТІВ")
        print("=" * 72)
        for row in rows:
            print(
                f"{row['Назва задачі']}: "
                f"вузлів={row['Вузли']}, "
                f"елементів={row['Елементи']}, "
                f"u_max={row['Максимальне переміщення, м']}, "
                f"sigma_vM_max={row['Максимальне напруження за Мізесом, Па']}, "
                f"час={row['Час, с']} с"
            )
        print("\nЗведену таблицю збережено у файлі results/summary.csv")


if __name__ == "__main__":
    main()
