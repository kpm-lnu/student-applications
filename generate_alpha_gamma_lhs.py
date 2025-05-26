import numpy as np
import pandas as pd
from scipy.stats import qmc

# ---------- налаштування ----------
n_points = 40            # скільки точок згенерувати
param_bounds = [             # межі параметрів: [min, max] для кожного
    [0.3, 1.5],              # alpha  ∈ [0.3; 1.5]
    [0.1, 0.35],             # gamma ∈ [0.1; 0.35]
]
outfile = f"alpha_gamma_lhs_{n_points}.csv"
# ----------------------------------

# 1) створюємо семпл LHS у [0;1]^d
sampler = qmc.LatinHypercube(d=len(param_bounds))
sample_unit = sampler.random(n=n_points)

# 2) масштабуємо до потрібних меж
lower_bounds = [b[0] for b in param_bounds]
upper_bounds = [b[1] for b in param_bounds]
sample_scaled = qmc.scale(sample_unit, lower_bounds, upper_bounds)

# 3) оформлюємо у DataFrame (додаємо sample_index для зручності)
df = pd.DataFrame(sample_scaled, columns=["alpha", "gamma"])
df.insert(0, "sample_index", np.arange(1, n_points + 1))

# 4) зберігаємо
df.to_csv(outfile, index=False)
print(f"✅ CSV із {n_points} точками збережено як '{outfile}'")
