#!/bin/bash
# SF1 (3 seeds, 30 batches) and SF10 (two batch regimes) maintenance runs with the vectorised delta checks
cd "$(dirname "$0")/.."
.venv/bin/python code/maint.py tpch tpcds && .venv/bin/python code/maint_scale.py tpch tpcds && echo MAINT_ALL_DONE
