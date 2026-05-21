#!/usr/bin/env bash
# reproduce_all.sh — повне відтворення всіх результатів курсової.
# Запуск:  bash reproduce_all.sh
set -e

VENV=".venv/bin/python"
cd "$(dirname "$0")"

echo "=== [1/6] Pytest ==="
$VENV -m pytest tests/ -v

echo ""
echo "=== [2/6] Production run (5 problems × 8 samplers × 1 seed) ==="
echo "    Очікуваний час: ~40 хв на 4-ядерному CPU"
$VENV run_experiments.py

echo ""
echo "=== [3/6] β-greedy sweep diagnostic ==="
$VENV diagnostics_beta_greedy.py

echo ""
echo "=== [4/6] RBF vs Kriging comparison (3 problems) ==="
echo "    Очікуваний час: ~25-35 хв"
$VENV diagnostics_rbf_vs_kriging.py

echo ""
echo "=== [5/6] Design panels (2D problems) ==="
$VENV diagnostics_design_panels.py

echo ""
echo "=== [6/6] Problem illustrations + winner / speedup tables ==="
$VENV diagnostics_problem_illustrations.py
$VENV diagnostics_winner_table.py

echo ""
echo "=== ВСЕ ГОТОВО ==="
echo "Артефакти у outputs/figures/ і outputs/tables/"
echo "Загальний час: ~75-90 хв"
