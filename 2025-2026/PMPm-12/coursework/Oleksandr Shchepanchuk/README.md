# surrogatelab — адаптивний семплінг для RBF-сурогатів

Магістерська курсова: *«Адаптивний семплінг для побудови сурогатних
моделей на основі радіально-базисних функцій»* (Львівський національний
університет імені Івана Франка, 2026).

Пакет реалізує:

- **8 семплерів:** 3 space-filling (Random, LHS, Halton) + 5 жадібних
  адаптивних (P-greedy, f-greedy, β-greedy, MEPE, EIGF).
- **2 сурогатні моделі:** `RBFSurrogate` (6 ядер: gaussian, MQ, IMQ,
  cubic, thin-plate, linear) із Rippa–Fasshauer LOOCV для вибору
  shape-параметра, та `KrigingSurrogate` (sklearn ARD-Matérn 5/2 з
  multi-start MLE).
- **5 тестових задач** із гладкими і негладкими QoI.
- **Equal-budget benchmark driver** — однакова кількість FOM-обчислень
  для всіх семплерів.

## Структура

```
surrogatelab/
├── problems.py          Problem-інтерфейс + 5 тестових задач
├── surrogates.py        RBFSurrogate + KrigingSurrogate
├── sampling.py          8 семплерів (registry-based)
├── fem.py               P1-скінченні елементи на 1D-сітці
├── metrics.py           NRMSE, MAE, R², MAX_RE, samples-to-tolerance
├── experiment.py        equal-budget driver
└── plotting.py          функції генерації фігур

run_experiments.py       основний production-прогін
rerun_amicon_only.py     re-run Heat/AdvDiff (n_max=120, інших не чіпає)
paper_eval_amicon.py     paper-сумісна оцінка (2000 random test pts, 10 seeds)
diagnostics_*.py         окремі діагностичні скрипти
replot_uk.py             регенерація PDF з вже існуючих CSV (без FOM-solves)
replot_paper_amicon.py   те саме для 4 paper-compat AMICon-PDF
regen_part_a_pngs.py     регенерація Part-A PNG (ЗДР/ДРЧП-фізика RD)
regen_panels_amicon.py   регенерація design_panels для Heat/AdvDiff
make_figures_zip.py      пакування PRIMARY-фігур у figures.zip для Prism
build_manifest.py        генерація outputs/MANIFEST.md (карта артефактів)
cleanup_j1.py            одноразова фільтрація J1 з артефактів
paper_vs_grid_comparison.py  side-by-side: 81-grid vs 2000-random
reproduce_all.sh         повна реплікація з нуля

outputs/
├── figures/             PDF (54) + PNG (6) — основні візуалізації
├── tables/              CSV (28) — сирі результати + summary-таблиці
├── summary.json         повний JSON-summary експерименту
└── MANIFEST.md          карта артефактів за підрозділами курсової

tests/                   pytest юніт-тести (19 шт.)
figures.zip              25 PRIMARY-фігур для одноразового завантаження у Prism
```

## Тестові задачі

| Назва | Розмірність | QoI | Особливість |
|---|---|---|---|
| Branin | 2D | f | аналітична, три глобальні мінімуми |
| Heat-AMICon | 2D | q | 1D steady heat, кусково-стала κ (розрив на x=0.5) |
| AdvDiff-AMICon | 2D | q | 1D steady advection-diffusion (upwind FD при високому Pe) |
| RD-2D | 2D | J, J2 | 2-видова reaction-diffusion (Variant-9), 2 активних параметри |
| RD-4D | 4D | J, J2 | те ж, 4 активних параметри |

## Встановлення

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Залежності: `numpy ≥ 1.24`, `scipy ≥ 1.10`, `pandas ≥ 2.0`,
`matplotlib ≥ 3.7`, `scikit-learn ≥ 1.3`, `pytest ≥ 7.0`.

## Повна реплікація результатів (~75-90 хв)

```bash
bash reproduce_all.sh
```

Послідовно виконує: `pytest` → `run_experiments.py` →
`diagnostics_beta_greedy.py` → `diagnostics_rbf_vs_kriging.py` →
`diagnostics_design_panels.py` → `diagnostics_problem_illustrations.py`
+ `diagnostics_winner_table.py`.

## Запуск окремих частин

### Основний benchmark (5 задач × 8 семплерів × 1 seed)

```bash
.venv/bin/python run_experiments.py            # production: ~40 хв, n=120 для AMICon, n=150 для RD-4D
.venv/bin/python run_experiments.py --quick    # smoke: ~2 хв
```

Виходи: `outputs/tables/{slug}_surrogate_*.csv`,
`outputs/figures/convergence_*.pdf`, `outputs/figures/samples_to_tol_*.pdf`,
`outputs/figures/overall_summary.pdf`, `outputs/summary.json`.

### Окремо AMICon-задачі при n_max=120

```bash
.venv/bin/python rerun_amicon_only.py
```

Інші задачі не зачіпає; інкрементально оновлює `summary.json`.

### Paper-сумісна оцінка (як у AMICon-2026)

```bash
.venv/bin/python paper_eval_amicon.py
```

Використовує **2000 uniform-random** тест-точок (seed=0), **10 seeds**,
n-grid `[5,10,15,20,30,40,60,80,100,120]`. Виходи з суфіксом `_paper_`.

### Діагностики

```bash
.venv/bin/python diagnostics_beta_greedy.py        # β ∈ [0, 1] sweep
.venv/bin/python diagnostics_rbf_vs_kriging.py     # RBF vs Kriging на 3 задачах
.venv/bin/python diagnostics_convergence.py        # сходження FEM-сітки
.venv/bin/python diagnostics_design_panels.py      # розміщення точок (4 задачі × 6 семплерів)
.venv/bin/python diagnostics_problem_illustrations.py  # Branin-surface, κ-схема
.venv/bin/python diagnostics_winner_table.py       # winner_per_problem.csv, speedup_vs_lhs.csv
```

### Юніт-тести

```bash
.venv/bin/python -m pytest tests/ -v   # 19 тестів, ~5 с
```

## Робочі скрипти (післяобробка)

| Скрипт | Призначення |
|---|---|
| `replot_uk.py` | Регенерує convergence/samples_to_tol PDF із CSV (без FOM-solves) |
| `replot_paper_amicon.py` | Те саме для 4 paper-compat AMICon-PDF |
| `regen_part_a_pngs.py` | Регенерує 6 Part-A PNG (ЗДР/ДРЧП-фізика RD) |
| `regen_panels_amicon.py` | Регенерує design_panels для Heat/AdvDiff |
| `cleanup_j1.py` | Видаляє J1 з артефактів (одноразово, вже виконано) |
| `build_manifest.py` | Будує `outputs/MANIFEST.md` — карту артефактів за §§ курсової |
| `make_figures_zip.py` | Пакує 25 PRIMARY-фігур у `figures.zip` для Prism |

## Артефакти

Після повного прогону у `outputs/`:

- **`MANIFEST.md`** — карта 54 PDF + 6 PNG + 28 CSV, класифікованих як
  PRIMARY (у тексті курсової), SECONDARY (резерв) і DIAGNOSTIC (для
  повторюваності).
- **`summary.json`** — повний `n_seeds`, `pde_nx`, `samples_to_tolerance`,
  `speedup_vs_LHS`, `median_nrmse_at_max_budget` для всіх задач і
  семплерів.
- **`figures.zip`** (у корені) — 25 PRIMARY-фігур для одноразового
  завантаження у LaTeX-редактор.

## Конфігурація `run_experiments.py`

Ключові константи у блоці `CONFIG`:

```python
PRODUCTION_BUDGETS = {
    "Branin":          [5, 10, ..., 60],       # 12 точок
    "Heat-AMICon":     [5, 10, ..., 120],      # 24 точки
    "AdvDiff-AMICon":  [5, 10, ..., 120],      # 24 точки
    "RD-2D":           [5, 10, ..., 60],       # 12 точок
    "RD-4D":           [10, 20, ..., 150],     # 15 точок
}
CONFIG = {
    "pde_nx": 81,           # сітка для RD (через J1-J2 sanity check)
    "seeds": [0],           # 1 seed (детерміністичне порівняння)
    "surrogate_kernel": "gaussian",
    "nugget": 1e-10,
    "log_transform": True,  # per-problem override через LOG_TRANSFORM_PER_PROBLEM
    "min_distance": 0.01,
    "tolerances": {"Branin": 0.05, "Heat-AMICon": 0.10, ..., "RD-2D": 0.20},
}
```

`--quick` режим: `pde_nx=41`, `seeds=[0,1]`, скорочена n-grid.

## Кешування FOM-обчислень

`outputs/fom_cache.pkl` зберігає вже обчислені пари `(μ → QoI)` між
запусками. Це **різко** прискорює повторні прогони: перший
`run_experiments.py` ~40 хв, повторний (без зміни задач) ~2 хв.

Для чистого прогону: видалити кеш-файл.

## Як зацитувати з LaTeX

Усі PDF / PNG мають стабільні імена (їх НЕ міняти між прогонами —
`\includegraphics{...}` у тезі прив'язано до них). Карта файл→підрозділ
— у `outputs/MANIFEST.md`. Приклад:

```latex
\begin{figure}[h]
  \includegraphics[width=0.9\linewidth]{figures/convergence_branin_f.pdf}
  \caption{Збіжність помилки сурогата на функції Бреніна.}
  \label{fig:branin-conv}
\end{figure}
```

## Примітка про одне зерно

`seeds=[0]` обрано свідомо: при 1 зерні криві збіжності — це чисті
значення помилки, не медіани, і смуги IQR (q25/q75) на графіках
відсутні (вони збігалися б із кривою). Усі підписи на графіках це
відображають: «Збіжність помилки сурогата», без «медіана за seeds».

Для оцінки розкиду seeds → див. `paper_eval_amicon.py` (10 seeds на
Heat/AdvDiff з 2000 random test points — paper-сумісна методологія).

## Структура CSV у `outputs/tables/`

| Файл | Опис |
|---|---|
| `{slug}_surrogate_results.csv` | Сирі рядки `(qoi, sampler, seed, n, nrmse, mae, r2, ...)` |
| `{slug}_surrogate_summary_nrmse.csv` | Агреговане `(qoi, sampler, n, median, q25, q75)` |
| `{slug}_samples_to_tolerance.csv` | На-семплер `n*(τ)` із кількістю цензурованих |
| `winner_per_problem.csv` | Найкращий семплер на кожну (задача, QoI) |
| `speedup_vs_lhs.csv` | Прискорення кожного adaptive-семплера проти LHS |
| `rbf_vs_kriging_all.csv` | 3 задачі × 5 budgets × 10 seeds × 2 surrogates |
| `beta_greedy_sweep.csv` | β-greedy NRMSE при β ∈ {0, 0.25, 0.5, 0.75, 1} |

## Як додати нову задачу

1. Створити підклас `Problem` у `surrogatelab/problems.py` (визначити
   `bounds`, `qoi_names`, `evaluate(mu)`).
2. Експортувати через `surrogatelab/__init__.py`.
3. Додати запис у `_production_problems()` у `run_experiments.py`.
4. Додати рядки у `PRODUCTION_BUDGETS` і `tolerances` у `CONFIG`.
5. (опційно) додати юніт-тест у `tests/test_problems.py`.

## Як додати новий семплер

1. Створити підклас `Sampler` (для space-filling) або `GreedySampler`
   (для адаптивного) у `surrogatelab/sampling.py`. Реалізувати
   `build(...)`. Декорувати `@register_sampler`.
2. Додати ім'я у `PRODUCTION_SAMPLERS` у `run_experiments.py`.
3. Додати стиль (колір, маркер) у `SAMPLER_STYLE` у `plotting.py`.

## Ліцензія

Освітній проєкт. Код вільний для академічного використання, посилання
на курсову обов'язкове.
