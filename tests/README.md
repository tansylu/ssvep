# SSVEP Analysis Tool Tests

This directory contains test modules for the SSVEP Analysis Tool.

## Running Tests

You can run the tests using the Python unittest framework:

```bash
# Run all tests
python -m unittest discover tests

# Run a specific test file
python -m unittest tests/test_signal_processing.py

# Run a specific test case
python -m unittest tests.test_signal_processing.TestSignalProcessing

# Run a specific test method
python -m unittest tests.test_signal_processing.TestSignalProcessing.test_perform_fourier_transform
```

## Test Files

- `test_signal_processing.py`: Tests for the signal processing module
- `test_db.py`: Tests for the database module

## Adding New Tests

When adding new tests, follow these guidelines:

1. Create a new test file with the naming convention `test_*.py`
2. Import the module you want to test
3. Create a test class that inherits from `unittest.TestCase`
4. Write test methods with names starting with `test_`
5. Use assertions to verify expected behavior

Example:

```python
import unittest
from src.module_to_test import function_to_test

class TestMyModule(unittest.TestCase):
    def test_my_function(self):
        result = function_to_test(input_value)
        self.assertEqual(result, expected_value)
```
