#!/usr/bin/env python
"""Script to run tests with coverage reporting."""
import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_tests(args):
    """Run tests with specified arguments."""
    cmd = ["pytest"]
    
    
    # Add test markers if specified
    if args.unit_only:
        cmd.append("-m")
        cmd.append("unit")
    
    # Add specific tests if provided
    if args.tests:
        cmd.extend(args.tests)
    else:
        cmd.append("tests/")
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add fail fast option
    if args.fail_fast:
        cmd.append("-x")
    
    # Print command
    print(f"Running: {' '.join(cmd)}")
    
    # Run tests
    return subprocess.call(cmd)

def main():
    parser = argparse.ArgumentParser(description="Run tests with coverage report.")
    parser.add_argument(
        "--package", 
        type=str, 
        default="titans",
        help="Package to measure coverage for"
    )
    parser.add_argument(
        "--report-type", 
        type=str, 
        default="term-missing",
        choices=["term", "term-missing", "html", "xml"], 
        help="Coverage report type"
    )
    parser.add_argument(
        "--xml-path", 
        type=str, 
        default="coverage.xml",
        help="Path for XML coverage report"
    )
    parser.add_argument(
        "--unit-only", 
        action="store_true",
        help="Run only unit tests"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--fail-fast", "-x", 
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "tests", 
        nargs="*",
        help="Specific test files or directories to run"
    )
    
    args = parser.parse_args()
    
    
    # Run the tests
    return_code = run_tests(args)
    
    # Exit with the same code as pytest
    sys.exit(return_code)

if __name__ == "__main__":
    main()