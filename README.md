### Simplified CMA-ES, written for learning
---
This implementation is based on [The CMA Evolution Strategy: A Tutorial](https://arxiv.org/abs/1604.00772) by Nikolaus Hansen.

#### Installation

```bash
git clone https://github.com/p-pawe-l/cma-es.git
cd cma-es
uv sync
```

#### Run the example

```bash
uv run main.py
```

#### Usage

```python
import numpy as np

from src.cma_es.cmaes import CMA_ES, k_best_selection
from src.cma_es.plot_utils import plot_fitness_history


def fitness(x: np.ndarray) -> float:
    return (x[0] - 2) ** 2 + (x[1] + 1) ** 2  # minimum at (2, -1)


init_pop = np.random.uniform(-3, 3, size=(10, 2))  # 10 points in 2D

cma_es = CMA_ES(k_best_selection, step_size=0.5, gamma=0.5, verbose=True)
best_sol, best_fit = cma_es.optimize(init_pop, fitness, runs=200)

print(f"Best solution: {best_sol}")
print(f"Best fitness: {best_fit}")

plot_fitness_history(cma_es.history, "fitness_history.png")
```

#### Parameters

- `selectionf` - selection function, e.g. `k_best_selection`
- `step_size` - scale of the sampling step around the mean
- `gamma` - covariance matrix learning rate, between 0 and 1
- `selection_ratio` - fraction of the population used to update the mean and covariance (default `0.5`)
- `verbose` - print the best fitness after each generation

The population size is taken from `init_pop`. Lower fitness is better.
