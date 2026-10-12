import typing
import dataclasses

from src.cma_es.math_cmaes import *


def k_best_selection(pop: list[np.ndarray], fitnesses: list[float], k: int) -> list[np.ndarray]:
    """
    Args:
        pop: population
        fitnesses: fitnesses of the population
        k: number of offspring to select

    Returns:
        selected population
    """
    assert len(pop) == len(fitnesses), "Population and fitnesses must have the same length"
    assert k > 0, "Number of offspring to select must be positive"

    sorted_indices: list[np.intp] = np.argsort(fitnesses)
    return [pop[i] for i in sorted_indices[:k]]


@dataclasses.dataclass
class RunHistory:
    run_id: int
    best_solution: np.ndarray
    best_fitness: float
    mean_fitness: float
    
    mean: np.ndarray
    covm: np.ndarray


class CMA_ES:
    def __init__(self, selectionf: typing.Callable, step_size: float, gamma: float, selection_ratio: float = 0.5, verbose: bool = False) -> None:
        self.selectionf: typing.Callable = selectionf
        self.step_size = step_size
        self.gamma = gamma
        self.selection_ratio = selection_ratio
        self.m_g: np.ndarray | None = None
        self.covm_g: np.ndarray | None = None
        self.verbose: bool = verbose

        self.history: list[RunHistory] = []


    def _log_run_to_history(self, run_id: int, best_solution: np.ndarray, best_fitness: float, mean: np.ndarray, covm: np.ndarray) -> None:
        """
        Args:
            run_id: run ID
            best_solution: best solution
            best_fitness: best fitness
            mean: mean of the population
            covm: adapted covariance matrix
        """
        self.history.append(RunHistory(run_id, best_solution, best_fitness, mean, covm))


    def _log_run_to_console(self, run_id: int, best_solution: np.ndarray, best_fitness: float, mean: np.ndarray, covm: np.ndarray) -> None:
        """
        Args:
            run_id: run ID
            best_solution: best solution
        """
        print(f"Run {run_id}: Best Fitness: {best_fitness}")


    def optimize(self, init_pop: np.ndarray, fitf: typing.Callable, runs: int = 100) -> tuple[np.ndarray, float]:
        """
        Args:
            init_pop: initial population
            fitf: fitness function
            runs: number of runs

        Returns:
            best solution and its fitness
        """
        assert len(init_pop) > 0, "Initial population must be non-empty"
        assert runs > 0, "Number of runs must be positive"

        k: int = len(init_pop) # Population size
        uniform_weights_all_pop: list[float] = [1.0 / k] * k        
        selection_size: int = int(k * self.selection_ratio) # How many offspring to select for the next generation
        uniform_weights_selected_pop: list[float] = [1.0 / selection_size] * selection_size

        self.m_g = calc_wmean(init_pop, uniform_weights_all_pop)
        self.covm_g = estimate_covariance_matrix(init_pop, uniform_weights_all_pop)

        best_solution: np.ndarray = init_pop[0]
        best_fitness: float = fitf(best_solution)

        for i in range(runs):
            new_pop: list[np.ndarray] = create_new_population(self.m_g, self.step_size, self.covm_g, k)
            new_fitnesses: list[float] = [fitf(point) for point in new_pop]
            mean_fitness: float = np.mean(new_fitnesses)

            selected_pop: list[np.ndarray] = self.selectionf(new_pop, new_fitnesses, selection_size)

            best_index: np.intp = np.argmin(new_fitnesses)
            if new_fitnesses[best_index] < best_fitness:
                best_solution = new_pop[best_index]
                best_fitness = new_fitnesses[best_index]

            new_covm = estimate_covariance_matrix(selected_pop, uniform_weights_selected_pop)
            self.covm_g = rank_mu_update_covariance_matrix(new_covm, self.covm_g, self.gamma)
            self.m_g = calc_wmean(selected_pop, uniform_weights_selected_pop)
            
            self._log_run_to_history(i + 1, best_solution, best_fitness, mean_fitness, self.m_g, self.covm_g)
            if self.verbose:
                self._log_run_to_console(i + 1, best_solution, best_fitness, self.m_g, self.covm_g)

        return best_solution, best_fitness