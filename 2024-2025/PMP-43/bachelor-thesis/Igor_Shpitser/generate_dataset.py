import argparse
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from material import Material
from shapeFunction import LinearQuadrilateralShapeFunction
from pso_algorithm import PSO, fitness_function, get_K_ref
from time import perf_counter

def is_convex_quad(pts):
    def cross_z(a, b, c):
        bax, bay = b[0]-a[0], b[1]-a[1]
        cbx, cby = c[0]-b[0], c[1]-b[1]
        return bax*cby - bay*cbx
    signs = []
    for i in range(4):
        cz = cross_z(pts[i], pts[(i+1)%4], pts[(i+2)%4])
        if abs(cz)>1e-8: signs.append(cz)
    return all(s>0 for s in signs) or all(s<0 for s in signs)

def optimize_on_quad(quad, mat, shape_func, args):
    """Окремий PSO-run на одному чотирикутнику."""
    coords = np.array(quad)

    K_ref = get_K_ref(
        coords,
        mat,
        nG=15,
        shape_func=shape_func
    )

    def fit(pos):
        print(f"✓ Done: best fitness = {fitness_function(pos, coords, mat, K_ref, shape_func):.6g}")
        return fitness_function(pos, coords, mat, K_ref, shape_func)

    DIM = 3 * args.ng * args.ng
    lo = np.concatenate([
        np.full(args.ng**2, -1.0),
        np.full(args.ng**2, -1.0),
        np.full(args.ng**2,  0.0),
    ])
    hi = np.concatenate([
        np.full(args.ng**2,  1.0),
        np.full(args.ng**2,  1.0),
        np.full(args.ng**2,  6.0),
    ])

    pso = PSO(
        fitness_func   = fit,
        n_particles    = args.npart,
        dim            = DIM,
        lo             = lo,
        hi             = hi,
        tol            = 1e-6,
        target_fitness = args.fitthr,
        maximize       = True
    )
    result = perf_counter()

    print(f"▶ Running PSO on one quadrilateral…")
    best_pos, best_fit = pso.optimize()
    print(f"✓ Done: best fitness = {best_fit:.6g}")
    result1 = perf_counter()
    print(f"time {result1-result:.6f}")

    return {
        "coords":   quad,
        "features": list(coords.flatten()),
        "labels":   best_pos.tolist(),
        "fitness":  best_fit
    }

def main():
    parser = argparse.ArgumentParser(description="Генерація датасету для PSO + NN")
    parser.add_argument("--ng",     type=int,   default=2,    help="точок Гауса на вісь")
    parser.add_argument("--npart",  type=int,   default=120,  help="число частинок у PSO")
    parser.add_argument("--fitthr", type=float, default=4.0,  help="поріг fitness для зупинки")
    parser.add_argument("--ndat",   type=int,   default=1000,  help="розмір датасету")
    parser.add_argument("--out",    type=str,   default="dataset.json", help="ім'я вихідного файлу")
    args = parser.parse_args()

    mat        = Material("Generic", E=2*0.7*(1+0.3), nu=0.3)
    shape_func = LinearQuadrilateralShapeFunction()

    rng   = np.random.default_rng(123)
    quads = []
    while len(quads) < args.ndat:
        bl = [ 1.0, 0.0]
        br = [ 2.0, 0.0]
        ul = [rng.uniform(0.5,8.0), rng.uniform(0.5,8.0)]
        ur = [rng.uniform(0.5,8.0), rng.uniform(0.5,8.0)]
        quad = [bl, br, ur, ul]
        if is_convex_quad(quad):
            quads.append(quad)

    results = []
    with ProcessPoolExecutor() as exe:
        futures = [exe.submit(optimize_on_quad, quad, mat, shape_func, args)
                   for quad in quads]
        for fut in as_completed(futures):
            results.append(fut.result())

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Збережено {len(results)} записів у {args.out}")

if __name__ == "__main__":
    main()

