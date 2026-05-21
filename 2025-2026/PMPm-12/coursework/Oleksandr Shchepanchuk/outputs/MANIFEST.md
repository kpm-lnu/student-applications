# Карта артефактів для §5 курсової

Згенеровано автоматично з `build_manifest.py`.

**Всього файлів:** 54 PDF + 6 PNG + 28 CSV

---

## PRIMARY — у тексті курсової

Ці файли вставляються в LaTeX через `\includegraphics{...}`.

### Графіки PDF

| Файл | Підрозділ | Призначення |
|---|---|---|
| `branin_surface.pdf` | §5.1 | §5.1 ілюстрація задачі Бренін |
| `heat_kappa_schema.pdf` | §5.1 | §5.1 схема κ(x) для Heat-AMICon |
| `convergence_branin_f.pdf` | §5.4 | §5.4 збіжність — Branin/f |
| `convergence_heat_amicon_q.pdf` | §5.4 | §5.4 збіжність — Heat-AMICon/q |
| `convergence_advdiff_amicon_q.pdf` | §5.4 | §5.4 збіжність — AdvDiff/q |
| `convergence_rd_2d_J.pdf` | §5.4 | §5.4 збіжність — RD-2D/J |
| `convergence_rd_4d_J.pdf` | §5.4 | §5.4 збіжність — RD-4D/J |
| `samples_to_tol_branin_f.pdf` | §5.4 | §5.4 бюджет — Branin/f |
| `samples_to_tol_heat_amicon_q.pdf` | §5.4 | §5.4 бюджет — Heat-AMICon/q |
| `samples_to_tol_advdiff_amicon_q.pdf` | §5.4 | §5.4 бюджет — AdvDiff/q |
| `samples_to_tol_rd_2d_J.pdf` | §5.4 | §5.4 бюджет — RD-2D/J |
| `samples_to_tol_rd_4d_J.pdf` | §5.4 | §5.4 бюджет — RD-4D/J |
| `overall_summary.pdf` | §5.5 | §5.5 зведене порівняння |
| `beta_greedy_sweep.pdf` | §4.3 | §4.3 sweep по β |
| `rbf_vs_kriging_rd_2d_convergence.pdf` | §5.6 | §5.6 RBF vs Kriging — основне |
| `rbf_vs_kriging_rd_2d_metrics.pdf` | §5.6 | §5.6 RBF vs Kriging — метрики |
| `convergence_heat_amicon_q_paper.pdf` | §5.4 | §5.4 paper-compat — Heat |
| `convergence_advdiff_amicon_q_paper.pdf` | §5.4 | §5.4 paper-compat — AdvDiff |
| `samples_to_tol_heat_amicon_q_paper.pdf` | §5.4 | §5.4 paper-compat — Heat бюджет |
| `samples_to_tol_advdiff_amicon_q_paper.pdf` | §5.4 | §5.4 paper-compat — AdvDiff бюджет |

### Графіки PNG (для §5.1)

| Файл | Підрозділ | Призначення |
|---|---|---|
| `ode_solution.png` | §5.1 | §5.1 ODE-розв'язок для RD |
| `pde_surface_y1.png` | §5.1 | §5.1 PDE-поверхня y1 для RD |
| `spatial_phase.png` | §5.1 | §5.1 просторова фаза для RD |

### Design panels (приклади для §5.2)

| Файл | Підрозділ | Призначення |
|---|---|---|
| `design_panel_heat_amicon_q_beta_greedy_0.5.pdf` | §5.2 | §5.2 β-greedy на Heat (адаптивний) |
| `design_panel_heat_amicon_q_LHS.pdf` | §5.2 | §5.2 LHS на Heat (неадаптивний, для контрасту) |

### Таблиці CSV

| Файл | Підрозділ | Призначення |
|---|---|---|
| `winner_per_problem.csv` | §5.5 | §5.5 переможці по кожній задачі |
| `speedup_vs_lhs.csv` | §5.5 | §5.5 прискорення vs LHS |
| `beta_greedy_sweep.csv` | §4.3 | §4.3 sweep по β дані |
| `branin_surrogate_summary_nrmse.csv` | §5.4 | §5.4 NRMSE — Branin |
| `heat_amicon_surrogate_summary_nrmse.csv` | §5.4 | §5.4 NRMSE — Heat-AMICon |
| `advdiff_amicon_surrogate_summary_nrmse.csv` | §5.4 | §5.4 NRMSE — AdvDiff |
| `rd_2d_surrogate_summary_nrmse.csv` | §5.4 | §5.4 NRMSE — RD-2D |
| `rd_4d_surrogate_summary_nrmse.csv` | §5.4 | §5.4 NRMSE — RD-4D |
| `heat_amicon_paper_samples_to_tol.csv` | §5.4 | §5.4 paper-compat hits — Heat |
| `advdiff_amicon_paper_samples_to_tol.csv` | §5.4 | §5.4 paper-compat hits — AdvDiff |

## SECONDARY — резерв (НЕ у тексті, для питань комісії)

### Графіки PDF

| Файл | Призначення |
|---|---|
| `convergence_rd_2d_J2.pdf` | RD-2D/J2 — допоміжна QoI |
| `convergence_rd_4d_J2.pdf` | RD-4D/J2 — допоміжна QoI |
| `samples_to_tol_rd_2d_J2.pdf` | RD-2D/J2 — бюджет |
| `samples_to_tol_rd_4d_J2.pdf` | RD-4D/J2 — бюджет |
| `rbf_vs_kriging_branin_convergence.pdf` | RBF vs Kriging — Branin |
| `rbf_vs_kriging_branin_metrics.pdf` | RBF vs Kriging — Branin метрики |
| `rbf_vs_kriging_heat_amicon_convergence.pdf` | RBF vs Kriging — Heat |
| `rbf_vs_kriging_heat_amicon_metrics.pdf` | RBF vs Kriging — Heat метрики |
| `rbf_vs_kriging_convergence.pdf` | RBF vs Kriging (legacy) |
| `rbf_vs_kriging_metrics.pdf` | RBF vs Kriging (legacy) |

**+22 додаткових design panels** (всі комбінації задача×семплер):

- `design_panel_advdiff_amicon_q_EIGF.pdf`
- `design_panel_advdiff_amicon_q_Halton.pdf`
- `design_panel_advdiff_amicon_q_LHS.pdf`
- `design_panel_advdiff_amicon_q_MEPE.pdf`
- `design_panel_advdiff_amicon_q_P-greedy.pdf`
- `design_panel_advdiff_amicon_q_beta_greedy_0.5.pdf`
- `design_panel_branin_f_EIGF.pdf`
- `design_panel_branin_f_Halton.pdf`
- `design_panel_branin_f_LHS.pdf`
- `design_panel_branin_f_MEPE.pdf`
- `design_panel_branin_f_P-greedy.pdf`
- `design_panel_branin_f_beta_greedy_0.5.pdf`
- `design_panel_heat_amicon_q_EIGF.pdf`
- `design_panel_heat_amicon_q_Halton.pdf`
- `design_panel_heat_amicon_q_MEPE.pdf`
- `design_panel_heat_amicon_q_P-greedy.pdf`
- `design_panel_rd_2d_J_EIGF.pdf`
- `design_panel_rd_2d_J_Halton.pdf`
- `design_panel_rd_2d_J_LHS.pdf`
- `design_panel_rd_2d_J_MEPE.pdf`
- `design_panel_rd_2d_J_P-greedy.pdf`
- `design_panel_rd_2d_J_beta_greedy_0.5.pdf`

## DIAGNOSTIC — для дослідницького аналізу

Сирі дані для повторюваності, не для тексту.

- `pde_traces.png` — RD — сліди на межах
- `pde_surface_y2.png` — RD — поверхня y2
- `ode_phase_portrait.png` — RD — фазовий портрет ODE
- `equilibria.csv` — RD — рівноваги
- `functionals_base.csv` — RD — функціонали для базових параметрів
- `rbf_vs_kriging.csv` — RBF vs Kriging legacy CSV
- `rbf_vs_kriging_all.csv` — RBF vs Kriging об'єднана таблиця
- `branin_surrogate_results.csv` — raw results — Branin
- `heat_amicon_surrogate_results.csv` — raw results — Heat
- `advdiff_amicon_surrogate_results.csv` — raw results — AdvDiff
- `rd_2d_surrogate_results.csv` — raw results — RD-2D
- `rd_4d_surrogate_results.csv` — raw results — RD-4D
- `branin_samples_to_tolerance.csv` — raw tol table — Branin
- `heat_amicon_samples_to_tolerance.csv` — raw tol table — Heat
- `advdiff_amicon_samples_to_tolerance.csv` — raw tol table — AdvDiff
- `rd_2d_samples_to_tolerance.csv` — raw tol table — RD-2D
- `rd_4d_samples_to_tolerance.csv` — raw tol table — RD-4D
- `heat_amicon_paper_results.csv` — raw paper-compat — Heat
- `advdiff_amicon_paper_results.csv` — raw paper-compat — AdvDiff
- `heat_amicon_paper_summary_nrmse.csv` — paper-compat summary — Heat
- `advdiff_amicon_paper_summary_nrmse.csv` — paper-compat summary — AdvDiff

---

## Підсумок NRMSE після rerun (без J1)

| Задача / QoI | Найкращий | NRMSE | LHS NRMSE | Покращення | Толер. |
|---|---|---|---|---|---|
| Branin / f | P-greedy | 0.007 ✓ | 0.033 | 4.4× | τ=0.05 |
| Heat-AMICon / q | β-greedy(β=0.5) | 0.214 ✗ | 0.603 | 2.8× | τ=0.1 |
| AdvDiff-AMICon / q | P-greedy | 0.241 ✗ | 0.632 | 2.6× | τ=0.1 |
| RD-2D / J | P-greedy | 0.053 ✓ | 0.055 | 1.0× | τ=0.2 |
| RD-2D / J2 | β-greedy(β=0.5) | 0.097 ✓ | 0.102 | 1.0× | τ=0.2 |
| RD-4D / J | β-greedy(β=0.5) | 0.063 ✓ | 0.079 | 1.2× | τ=0.2 |
| RD-4D / J2 | P-greedy | 0.077 ✓ | 0.101 | 1.3× | τ=0.2 |

## Samples-to-tolerance hits

Скільки семплерів досягли τ для кожної задачі:

| Задача / QoI | Hit | Miss |
|---|---|---|
| Branin / f | 5/8 | 3/8 |
| Heat-AMICon / q | 0/8 | 8/8 |
| AdvDiff-AMICon / q | 0/8 | 8/8 |
| RD-2D / J | 8/8 | 0/8 |
| RD-2D / J2 | 8/8 | 0/8 |
| RD-4D / J | 8/8 | 0/8 |
| RD-4D / J2 | 8/8 | 0/8 |
