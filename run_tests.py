#!/usr/bin/env python3
"""
Test Pipeline Runner
Runs all unit tests and provides clear reporting for CI/CD pipeline
"""

import unittest
import sys
import os
import json

def run_test_pipeline():
    """Main test pipeline function"""
    print("Starting Event Data Validation Pipeline")
    print("=" * 60)
    
    # Add tests directory to path
    sys.path.insert(0, 'tests')
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='*test*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Generate pipeline report
    print("=" * 60)
    print("PIPELINE TEST REPORT")
    print("=" * 60)
    
    # Test summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors - skipped}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    
    # Detailed failure report
    if result.failures:
        print("\nFAILED TESTS:")
        for test, traceback in result.failures:
            print(f"   - {test}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"   - {test}")
    
    # Pipeline status
    if result.wasSuccessful():
        print("\nPIPELINE STATUS: SUCCESS")
        print("All tests passed! Data quality verified.")
        print("Ready for FAISS vectorization and deployment.")
        return 0
    else:
        print("\nPIPELINE STATUS: FAILED")
        print("Some tests failed! Please fix issues before deployment.")
        return 1

if __name__ == '__main__':
    exit_code = run_test_pipeline()
    sys.exit(exit_code)