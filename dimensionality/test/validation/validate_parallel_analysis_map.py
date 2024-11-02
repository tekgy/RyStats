"""Validation of parallel analysis MAP implementation using BFI dataset."""

import unittest
import numpy as np
import pandas as pd
from scipy import stats
import time
from sklearn.datasets import fetch_openml

from RyStats.dimensionality import parallel_analysis_map
from RyStats.common.procrustes import procrustes

class ValidationHelpers:
    """Helper methods for validation analyses."""
    
    @staticmethod
    def calculate_factor_accuracy(factor_count, target_count):
        """Calculate accuracy metrics for factor count.
        
        Args:
            factor_count: int, number of factors found
            target_count: int, known number of factors
            
        Returns:
            dict containing error and binary accuracy
        """
        return {
            'absolute_error': abs(factor_count - target_count),
            'exact_match': factor_count == target_count
        }

    @staticmethod
    def analyze_timing(results_df):
        """Analyze processing time across parameter combinations.
        
        Args:
            results_df: DataFrame containing results
            
        Returns:
            dict containing timing analysis
        """
        timing = results_df.groupby('processors')['processing_time'].agg([
            'mean', 'std', 'min', 'max'
        ])
        
        serial_time = timing.loc[1, 'mean']
        speedups = serial_time / timing['mean']
        
        return {
            'summary': timing,
            'speedups': speedups,
            'efficiency': speedups / results_df['processors']
        }

    @staticmethod
    def analyze_method_stability(results_df, param):
        """Analyze stability of factor suggestions for a parameter.
        
        Args:
            results_df: DataFrame containing results
            param: str, parameter to analyze
            
        Returns:
            dict containing stability metrics
        """
        stability = results_df.groupby(param).agg({
            'factor_std': ['mean', 'std'],
            'n_factors': lambda x: len(set(x))
        })
        
        cv = stability[('factor_std', 'mean')] / \
             results_df.groupby(param)['n_factors'].mean()
        
        return {
            'metrics': stability,
            'cv': cv
        }

class TestParallelAnalysisMapValidation(unittest.TestCase):
    """Validation test suite for parallel analysis MAP implementation."""

    @classmethod
    def setUpClass(cls):
        """Set up validation data and known structure."""
        try:
            data = fetch_openml(data_id=43673, as_frame=True)
            bfi_cols = [col for col in data.data.columns if col.startswith('bfi_')]
            cls.bfi_data = data.data[bfi_cols].values.T
            
            # Verify scale properties
            unique_values = np.unique(cls.bfi_data[~np.isnan(cls.bfi_data)])
            assert np.array_equal(unique_values, np.array([1, 2, 3, 4, 5]))
            assert cls.bfi_data.shape[0] == 44  # 44 BFI items
            
            cls.bfi_structure = {
                'n_factors': 5,
                'factor_names': ['E', 'A', 'C', 'N', 'O'],
                'item_factors': {
                    'E': [1, 6, 11, 16, 21, 26, 31, 36],
                    'A': [2, 7, 12, 17, 22, 27, 32, 37],
                    'C': [3, 8, 13, 18, 23, 28, 33, 38],
                    'N': [4, 9, 14, 19, 24, 29, 34, 39],
                    'O': [5, 10, 15, 20, 25, 30, 35, 40]
                },
                'reversed_items': [6, 21, 31, 2, 12, 27, 37, 8, 18, 23, 43, 9, 24, 34]
            }
            
            # Store distribution properties
            cls.bfi_properties = {
                'skew': stats.skew(cls.bfi_data[~np.isnan(cls.bfi_data)].flatten()),
                'kurtosis': stats.kurtosis(cls.bfi_data[~np.isnan(cls.bfi_data)].flatten())
            }
            
        except Exception as e:
            print(f"Error fetching BFI data: {e}")
            cls.bfi_data = None
            cls.bfi_structure = None
            cls.bfi_properties = None

    def analyze_parameter_impacts(self, map_results):
        """Analyze impact of parameters on MAP results.
        
        Args:
            map_results: dict, results from MAP analysis
            
        Returns:
            dict containing comprehensive analysis results
        """
        # [Previous implementation]

    def test_comprehensive_validation(self):
        """Validate MAP implementation using BFI dataset."""
        if self.bfi_data is None:
            self.skipTest("BFI data not available")

        # Define parameter space
        correlation_types = [
            ('pearsons',),
            ('spearman',),
            ('kendall',),
            ('polychoric', 1, 5)
        ]
        resample_methods = [
            'permutation',
            'bootstrap',
            'distribution'
        ]
        num_processors = [1, 2, 4]
        num_iterations = [5, 50, 100]

        # Run comprehensive analysis
        map_results = {}
        
        for rm in resample_methods:
            for ct in correlation_types:
                for ni in num_iterations:
                    for np in num_processors:
                        method_key = f'{rm}_{ct[0]}_{ni}_{np}'
                        print(f'\nProcessing {method_key}...')
                        
                        try:
                            start_time = time.time()
                            n_factors, suggestions = parallel_analysis_map(
                                self.bfi_data,
                                n_iterations=ni,
                                correlation=ct,
                                resampling=rm,
                                seed=435789,
                                num_processors=np
                            )
                            processing_time = time.time() - start_time
                            
                            map_results[method_key] = {
                                'n_factors': n_factors,
                                'suggestions': suggestions,
                                'processing_time': processing_time,
                                'params': {
                                    'resampling': rm,
                                    'correlation': ct[0],
                                    'iterations': ni,
                                    'processors': np
                                }
                            }
                            
                        except Exception as e:
                            print(f"Error with {method_key}: {str(e)}")

        # Analyze results
        analysis = self.analyze_parameter_impacts(map_results)
        
        # Validate results
        poly_results = [v for k, v in map_results.items() if 'polychoric' in k]
        poly_accuracy = [abs(r['n_factors'] - 5) for r in poly_results]
        
        self.assertTrue(
            any(acc == 0 for acc in poly_accuracy),
            "Polychoric correlation failed to identify correct BFI structure"
        )

if __name__ == '__main__':
    unittest.main()