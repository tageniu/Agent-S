#!/usr/bin/env python3
"""
Wrapper script to run GAIA benchmark from project root.
"""

import sys
from pathlib import Path

# Import and run the actual benchmark runner
sys.path.insert(0, str(Path(__file__).parent))
from benchmarks.gaia.run_gaia_benchmark import main

if __name__ == "__main__":
    main()