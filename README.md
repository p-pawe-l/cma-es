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

#### Limitations

This is a simplified version of CMA-ES. Compared to the full algorithm described in Hansen's tutorial, it is missing:

- **Step-size adaptation** - `step_size` stays fixed for the whole run. Full CMA-ES adapts it every generation using cumulative step-size adaptation (CSA) with the evolution path $p_\sigma$, so it can take large steps far from the optimum and small steps near it.
- **Evolution paths and rank-one update** - the covariance matrix is updated only with the rank-$\mu$ update. Full CMA-ES also tracks the evolution path $p_c$ and uses it for a rank-one update, which helps learn directions the search keeps moving in.
- **Covariance of the steps** - the covariance is estimated around the mean of the selected points. Full CMA-ES uses the steps from the old mean scaled by the step size, $y_i = (x_i - m_{old}) / \sigma$. Without this, the covariance shrinks every generation and the search can stop before it reaches the optimum.
- **Weighted recombination** - all selected points have equal weights. Full CMA-ES gives better points larger weights (log-decreasing) and can also use negative weights for the worst points.
- **Default parameters** - population size, learning rates and damping are set by hand. Full CMA-ES derives them from the problem dimension $n$, e.g. population size $\lambda = 4 + \lfloor 3 \ln n \rfloor$.
- **Initialization** - the initial mean and covariance are estimated from `init_pop`. Full CMA-ES starts from a single point $x_0$, an initial step size $\sigma_0$ and $C = I$.
- **Stopping criteria** - the algorithm always runs a fixed number of generations. Full CMA-ES stops on fitness tolerance, stagnation or a badly conditioned covariance matrix.
- **Restarts** - no restart strategies such as IPOP or BIPOP, which help on multimodal functions like Rastrigin.
- **Constraints** - no handling of bounds or constraints.
