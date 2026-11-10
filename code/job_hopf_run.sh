#!/bin/bash
# HoPF and the exact profilers on JOB (memory heavy: run alone, at low priority)
cd "$(dirname "$0")/.."
OUT_SUFFIX=_job PROFILE_CACHE=<SCRATCH> nice -n 10 .venv/bin/python code/baseline_profilers.py job > results/baseline_profilers_job.log 2>&1
OUT_SUFFIX=_job PROFILE_CACHE=<SCRATCH> nice -n 10 .venv/bin/python code/baseline_hopf.py job > results/baseline_hopf_job.log 2>&1
echo JOB_HOPF_DONE >> results/baseline_hopf_job.log
