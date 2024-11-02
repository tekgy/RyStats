"""Parallel analysis implementations using Velicer's Minimum Average Partial (MAP) criterion
for determining number of components/factors to retain.

The MAP criterion examines the average squared partial correlations after 
successively removing components. The number of components that minimizes
this criterion represents the optimal dimensionality where systematic variance 
has been maximally separated from error variance [1].

Empirically Demonstrated Advantages of MAP over traditional parallel analysis:
- Does not rely on comparing eigenvalues to random data [1]
- More robust to violations of distributional assumptions [2]
- Tends to give more parsimonious solutions [2]
- Performs well with highly correlated items [1]

Correlation Methods:
Empirically supported characteristics:
- Pearson: Requires continuous data and linear relationships [3]
- Polychoric: Appropriate for ordinal data assumed to reflect underlying 
  continuous normal distributions [4]
- Spearman/Kendall: Robust to violations of normality and outliers [3]

Common Rules of Thumb for Method Selection:
Note: These are conventional guidelines, not strict requirements
- Pearson: Often used with n > 300, normally distributed continuous data
- Spearman: Popular choice for n > 100 when normal assumptions violated
- Kendall: Sometimes preferred for small samples (n < 30)
- Polychoric: Commonly used for Likert scales, usually with n > 200

Resampling Methods:
Theoretical Properties:
- Permutation: Preserves individual variable distributions
- Bootstrap: Preserves correlation structure
- Distribution-preserving: Attempts to maintain both (newly implemented method)

Practical Recommendations (based on common practice):
1. For Likert scales: Consider polychoric correlations
2. For continuous normal data: Pearson correlations typically sufficient
3. For small samples: Consider Kendall's tau
4. For mixed data types: Compare multiple methods
5. Always examine factor interpretability alongside statistical criteria

References:
[1] Velicer, W. F. (1976). Determining the number of components from the 
    matrix of partial correlations. Psychometrika, 41(3), 321-327.
[2] O'Connor, B. P. (2000). SPSS and SAS programs for determining the 
    number of components using parallel analysis and Velicer's MAP test. 
    Behavior Research Methods, Instruments, & Computers, 32(3), 396-402.
[3] Garrido, L. E., Abad, F. J., & Ponsoda, V. (2013). A new look at Horn's 
    parallel analysis with ordinal variables. Psychological Methods, 18(4), 454-474.
[4] Olsson, U. (1979). Maximum likelihood estimation of the polychoric 
    correlation coefficient. Psychometrika, 44(4), 443-460.
"""

from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
from concurrent import futures
from itertools import repeat
import numpy as np
from scipy.stats import spearmanr, kendalltau, skew, kurtosis
from RyStats.common.polychoric import polychoric_correlation_serial
from RyStats.dimensionality.map import minimum_average_partial
from numpy.random import SeedSequence


__all__ = ["parallel_analysis_map", "parallel_analysis_map_serial"]


def _get_correlation_function(method):
    """Returns the correlation function."""
    if method[0] == 'pearsons':
        func = np.corrcoef
    elif method[0] == 'spearman':
        return lambda x: spearmanr(x.T)[0]
    elif method[0] == 'kendall':
        def kendall_matrix(x):
            n = x.shape[0]
            corr = np.eye(n)
            for i in range(n):
                for j in range(i):
                    corr[i,j] = corr[j,i] = kendalltau(x[i], x[j])[0]
            return corr
        return kendall_matrix
    elif method[0] == 'polychoric':
        func = lambda x: polychoric_correlation_serial(x, start_val=method[1], stop_val=method[2])
    else:
        raise ValueError('Unknown correlation method {}'.format(method[0]))
    return func


def _get_resampling_function(method):
    """Returns the resampling function."""
    if method == 'permutation':
        return lambda rng, data: rng.permutation(data, axis=1)
    elif method == 'bootstrap':
        def simple_bootstrap(rng, data):
            n_samples = data.shape[1]
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            return data[:, indices]
        return simple_bootstrap
    elif method == 'distribution':
        def distribution_preserving(rng, data):
            n_samples = data.shape[1]
            resampled = np.zeros_like(data)
            for i in range(data.shape[0]):
                orig_mean = np.mean(data[i])
                orig_std = np.std(data[i])
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                resampled[i] = data[i, indices]
                resampled[i] = (resampled[i] - np.mean(resampled[i])) / np.std(resampled[i])
                resampled[i] = resampled[i] * orig_std + orig_mean
            return resampled
        return distribution_preserving
    else:
        raise ValueError('Unknown resampling method {}'.format(method))


def parallel_analysis_map_serial(raw_data, n_iterations, correlation=('pearsons',),
                               resampling='permutation', seed=None):
    """Estimate dimensionality using MAP criterion from resampled data.

    Args:
        raw_data:  [n_items x n_observations] Raw collected data
        n_iterations:  Number of iterations to run
        correlation: Method to construct correlation matrix either:
                    ('pearsons',) for continuous data
                    ('spearman',) for rank correlations
                    ('kendall',) for rank correlations
                    ('polychoric', min_val, max_val) for ordinal data
        resampling: Method to resample data:
                    'permutation' - permute each variable
                    'bootstrap' - random sampling with replacement
                    'distribution' - preserves distribution properties
        seed:  (integer) Random number generator seed value

    Returns:
        suggested_factors: mode of suggested number of factors
        factor_counts: number of factors suggested in each iteration
    """
    random_seeds = SeedSequence(seed).spawn(n_iterations)
    n_items = raw_data.shape[0]
    raw_data = raw_data.reshape(1, -1)

    correlation_method = _get_correlation_function(correlation)
    resampling_method = _get_resampling_function(resampling)

    factor_suggestions = np.zeros(n_iterations, dtype=int)

    for ndx, rseed in enumerate(random_seeds):
        rng_local = np.random.default_rng(rseed)
        new_data = resampling_method(rng_local, raw_data).reshape(n_items, -1)
        local_correlation = correlation_method(new_data)
        n_factors, _ = minimum_average_partial(local_correlation)
        factor_suggestions[ndx] = n_factors
    
    suggested_factors = int(np.mode(factor_suggestions)[0])
    return suggested_factors, factor_suggestions


def parallel_analysis_map(raw_data, n_iterations, correlation=('pearsons',),
                         resampling='permutation', seed=None, num_processors=2):
    """Estimate dimensionality using MAP criterion from resampled data.

    Args:
        raw_data:  [n_items x n_observations] Raw collected data
        n_iterations:  Number of iterations to run
        correlation: Method to construct correlation matrix either:
                    ('pearsons',) for continuous data
                    ('spearman',) for rank correlations
                    ('kendall',) for rank correlations
                    ('polychoric', min_val, max_val) for ordinal data
        resampling: Method to resample data:
                    'permutation' - permute each variable
                    'bootstrap' - random sampling with replacement
                    'distribution' - preserves distribution properties
        seed:  (integer) Random number generator seed value
        num_processors: number of processors on a multi-core cpu to use

    Returns:
        suggested_factors: mode of suggested number of factors
        factor_counts: number of factors suggested in each iteration
    """
    if num_processors == 1:
        return parallel_analysis_map_serial(raw_data, n_iterations, correlation,
                                         resampling, seed)
    
    random_seeds = SeedSequence(seed).spawn(n_iterations)
    chunk_seeds = np.array_split(random_seeds, num_processors)

    n_items = raw_data.shape[0]
    raw_data = raw_data.reshape(1, -1)
    
    with SharedMemoryManager() as smm:
        shm = smm.SharedMemory(size=raw_data.nbytes)
        shared_buff = np.ndarray(raw_data.shape, dtype=raw_data.dtype, buffer=shm.buf)
        shared_buff[:] = raw_data[:]

        with futures.ThreadPoolExecutor(max_workers=num_processors) as pool:
            results = pool.map(_map_engine, repeat(shm.name), repeat(correlation),
                             repeat(resampling), repeat(n_items), 
                             repeat(raw_data.dtype), repeat(raw_data.shape), 
                             chunk_seeds)

    factor_suggestions = np.concatenate(list(results))
    suggested_factors = int(np.mode(factor_suggestions)[0])
    
    return suggested_factors, factor_suggestions


def _map_engine(name, correlation, resampling, n_items, dtype, shape, subset):
    """MAP engine for distributed computing."""
    correlation_method = _get_correlation_function(correlation)
    resampling_method = _get_resampling_function(resampling)
    factor_suggestions = np.zeros(len(subset), dtype=int)
    
    existing_shm = shared_memory.SharedMemory(name=name)    
    raw_data = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    for ndx, rseed in enumerate(subset):
        rng_local = np.random.default_rng(rseed)
        new_data = resampling_method(rng_local, raw_data).reshape(n_items, -1)
        local_correlation = correlation_method(new_data)
        n_factors, _ = minimum_average_partial(local_correlation)
        factor_suggestions[ndx] = n_factors
        
    existing_shm.close()
    return factor_suggestions
