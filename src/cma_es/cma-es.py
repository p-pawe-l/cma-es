import numpy as np 


def probe_normal_dist_offspring(m_g: np.ndarray, std_dev_g: float, covm_g: np.ndarray) -> np.ndarray:
    """
    Args:
        m_g: mean of the normal distribution
        std_dev_g: standard deviation of the normal distribution
        covm_g: covariance matrix of the normal distribution

    Returns:
        offspring sampled from the normal distribution
    """

    offspring = std_dev_g * np.random.multivariate_normal(np.zeros(len(m_g)), covm_g)
    offspring = m_g + offspring
    return offspring


def create_new_population(m_g: np.ndarray, std_dev_g: float, covm_g: np.ndarray, k: int) -> np.ndarray:
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
    return np.array(offsprings)


def calc_wmean(points, weights):
    """
    Calculate the weighted mean of a set of points.

    Args:
        points: numpy array of shape (n_points, n_dimensions)
        weights: numpy array of shape (n_points,)

    Returns:
        numpy array of shape (n_dimensions,)
    """
    
    assert len(points) == len(weights), "Points and weights must have the same length"
    assert np.sum(weights) == 1, "Weights must sum to 1"
    weighted_points = points * weights
    return np.sum(weighted_points, axis=0)