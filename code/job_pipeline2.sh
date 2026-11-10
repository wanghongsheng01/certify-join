#!/bin/bash
cd "$(dirname "$0")/.."
.venv/bin/python code/job_load.py || exit 1
.venv/bin/python code/prepare.py job || exit 1
.venv/bin/python code/stats.py || exit 1
TAG=job .venv/bin/python code/run_pilot.py job && echo JOB_PIPELINE_DONE
