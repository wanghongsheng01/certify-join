#!/bin/bash
# After the parallel download: assemble imdb.tgz, extract, load into DuckDB, then build the JOB pool and run the pilot.
J=<DATA_DIR>
cd "$(dirname "$0")/.."
until grep -q ALL_PARTS_DONE $J/logs/dl_parts.log; do sleep 20; done
cat $J/dl/parts/p* > $J/dl/imdb_full.tgz
ls -l $J/dl/imdb_full.tgz
[ "$(stat -f %z $J/dl/imdb_full.tgz)" = "1263193115" ] || { echo SIZE_MISMATCH; exit 1; }
mkdir -p $J/dl/imdb && tar -xzf $J/dl/imdb_full.tgz -C $J/dl/imdb && ls $J/dl/imdb | head -30
.venv/bin/python code/job_load.py || exit 1
.venv/bin/python code/prepare.py job || exit 1
.venv/bin/python code/stats.py || exit 1
TAG=job .venv/bin/python code/run_pilot.py job && echo JOB_PIPELINE_DONE
