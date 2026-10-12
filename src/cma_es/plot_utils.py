import seaborn as sns
import matplotlib.pyplot as plt

from src.cma_es.cmaes import RunHistory


def plot_fitness_history(history: list[RunHistory], filename: str | None = None) -> None:
    """
    Args:
        history: history of runs
    """
    sns.lineplot(x=[run.run_id for run in history], y=[run.best_fitness for run in history])
    if filename is not None:
        plt.savefig(filename)
    plt.show()