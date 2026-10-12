import numpy as np 
import math 


def probe_normal_dist_offspring(m_g: np.ndarray, step_size: float, covm_g: np.ndarray) -> np.ndarray:
    """
    Args:
        m_g: mean of the normal distribution
        step_size: step size of the normal distribution
        covm_g: covariance matrix of the normal distribution

    Returns:
        offspring sampled from the normal distribution
    """

    offspring: np.ndarray = step_size * np.random.multivariate_normal(m_g, covm_g)
    return offspring


def create_new_population(m_g: np.ndarray, std_dev_g: float, covm_g: np.ndarray, k: int) -> list[np.ndarray]:
    """
    Args:
        m_g: mean of the normal distribution
        std_dev_g: standard deviation of the normal distribution
        covm_g: covariance matrix of the normal distribution
        k: number of offspring to sample

    Returns:
        new population of offspring as a numpy array
    """

    offsprings: list = []
    for _ in range(k):
        offsprings.append(probe_normal_dist_offspring(m_g, std_dev_g, covm_g))
    return offsprings


def calc_wmean(points: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """
    Args:
        points: list of offsprings
        weights: list of weights of each offspring

    Returns:
        weighted mean of the offsprings
    """
    
    assert len(points) == len(weights), "Points and weights must have the same length"
    assert math.isclose(sum(weights), 1.0, rel_tol=1e-6), "Weights must sum to 1"
    weighted_points: list[np.ndarray] = [point * weight for point, weight in zip(points, weights)]
    return np.sum(weighted_points, axis=0)

