from src.cma_es.cmaes import *
from src.cma_es.plot_utils import *

def rastrigin(x: np.ndarray) -> float:
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))



def main() -> None:
    init_pop: np.ndarray = np.random.uniform(-3, 3, size=(5, 2))
    cma_es = CMA_ES(k_best_selection, 0.1, 0.5, verbose=True)
    best_sol, best_fit = cma_es.optimize(init_pop, rastrigin, runs=200)
    print(f"Best solution: {best_sol}")
    print(f"Best fitness: {best_fit}")

    plot_fitness_history(cma_es.history, "fitness_history.png")

if __name__ == "__main__":
    main()