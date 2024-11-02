# RyStats Validation Tests

This directory contains validation tests that are separate from unit tests. While unit tests focus on testing specific functionality, validation tests:

1. Use real-world datasets with known properties
2. Validate against established psychometric structures
3. Compare performance across multiple methods
4. Provide comprehensive parameter impact analyses

## Current Validation Suites:
- parallel_analysis_map.py: Validates MAP implementation using BFI dataset
  - Tests correlation methods
  - Tests resampling approaches
  - Analyzes parameter impacts
  - Compares against known 5-factor structure

## Running Validation Tests
```python
python -m unittest RyStats.dimensionality.tests.validation.validation_parallel_analysis_map
