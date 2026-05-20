from __future__ import annotations

from dataclasses import dataclass, field

from src.analysis.convergence import ConvergenceAnalyzer, ConvergenceRow
from src.analysis.error_calculator import ErrorCalculator
from src.analysis.trace_solution import TraceSolution
from src.plotter.plotter_service import PlotterService
from src.problem.fractional_problem import FractionalProblem, FractionalProblemConfig
from src.solver.fractional_fem_solver import FractionalFemSolver
from src.solver.solution_result import SolutionResult


@dataclass(slots=True)
class ConvergenceApplication:
    config: FractionalProblemConfig = field(default_factory=FractionalProblemConfig)

    def run(self) -> None:
        problem = FractionalProblem(self.config)
        solver = FractionalFemSolver(problem)
        error_calculator = ErrorCalculator(problem)
        convergence_analyzer = ConvergenceAnalyzer()
        rows, last_solution = self._run_convergence_study(
            solver=solver,
            error_calculator=error_calculator,
            convergence_analyzer=convergence_analyzer,
        )
        self._print_convergence_table(rows)
        self._print_maximum_trace_value(last_solution)
        PlotterService.plot_trace_solution(last_solution, problem)
        PlotterService.plot_solution(last_solution, title="U(x,y)")

    def _run_convergence_study(
        self,
        solver: FractionalFemSolver,
        error_calculator: ErrorCalculator,
        convergence_analyzer: ConvergenceAnalyzer,
    ) -> tuple[list[ConvergenceRow], SolutionResult]:
        rows: list[ConvergenceRow] = []
        last_solution: SolutionResult | None = None

        for n in self.config.mesh_sizes:
            solution = solver.solve(n_x=n, n_y=n)
            rows.append(
                ConvergenceRow(
                    N=n,
                    h=self.config.L / n,
                    l2_error=error_calculator.compute_l2_error(solution.vertices, solution.values),
                    w21_error=error_calculator.compute_w21_error(solution.vertices, solution.values),
                )
            )
            last_solution = solution

        convergence_analyzer.add_orders(rows)

        if last_solution is None:
            raise RuntimeError("No solution was computed")

        return rows, last_solution

    @staticmethod
    def _print_convergence_table(rows: list[ConvergenceRow]) -> None:
        print("\nConvergence table")
        print("N      h              L2 error       L2 order      H1 error       H1 order")
        print("-" * 75)

        for row in rows:
            l2_order = "-" if row.l2_order is None else f"{row.l2_order:.4f}"
            w21_order = "-" if row.w21_order is None else f"{row.w21_order:.4f}"
            print(
                f"{row.N:<6d} "
                f"{row.h:<14.6f} "
                f"{row.l2_error:<14.6e} "
                f"{l2_order:<13} "
                f"{row.w21_error:<14.6e} "
                f"{w21_order:<13} "
            )

    @staticmethod
    def _print_maximum_trace_value(solution: SolutionResult) -> None:
        _, u_values = TraceSolution.extract(solution.vertices, solution.values)
        print("\nMaximum value on y = 0:", u_values.max())
