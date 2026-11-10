# Certify-Join: Budgeted Certification and Incremental Maintenance of Aggregation-Safe Join Graphs

Artifact of the ICDE 2027 submission. It contains the implementation, the scripts that build every public instance,
the candidate pools with exact atom truths and cached LLM priors, the raw results behind every table and figure,
and the technical report with the full proofs (`techreport_proofs.pdf`).

## Layout

| Path | Content |
|---|---|
| `code/policies.py` | Certify-Join (adaptive bundle-greedy prober), its sampled variant, probing baselines (Full, Random, KeyCoverage, Profile) and the clairvoyant ILP |
| `code/atoms.py` | Exact key and inclusion probes (SQL on DuckDB), including temporal (SCD) atoms |
| `code/instances.py`, `code/prepare.py`, `code/stats.py` | Instance loaders, the candidate generator, defect injection, pool construction, catalog statistics |
| `code/llm.py`, `code/llm_prior.py` | Cached OpenAI-compatible client and the LLM priors (schema metadata only) |
| `code/run_pilot.py` | Coverage-budget experiments and zero-probe baselines (RQ1-RQ3) |
| `code/baseline_sji.py`, `code/baseline_autobi.py`, `code/baseline_hopf.py`, `code/baseline_profilers.py` | Reimplementations of SJI (arXiv 2603.04176), Auto-BI (PVLDB 2023), HoPF (JIIS 2019), and SPIDER+HyUCC via Desbordante |
| `code/calib_uniform.py` | Calibration of sampled inclusion certificates (RQ4) |
| `code/maint.py`, `code/maint_scale.py`, `code/sf10_augment.py` | Certificate maintenance at SF1 and SF10 (RQ5) |
| `code/synth_scale.py` | Planning time on synthetic schemas with 10^2-10^4 tables (RQ6) |
| `code/prior_vendor.py` | Certify-Join with priors from each LLM vendor (RQ3) |
| `code/paper_figs.py` | Regenerates every table (`paper/tables`) and figure (`paper/figs`) of the paper from `results/` |
| `results/pool/*.json` | Candidate edges, exact atom truths and LLM priors per instance (the cached LLM outputs) |
| `results/*.csv`, `results/maint/*.json` | Raw results |

## Reproducing

```
pip install -r requirements.txt
python code/gen_tpc.py                       # TPC-H / TPC-DS SF1 (DuckDB dbgen/dsdgen)
export BIRD_DEV=<path to BIRD dev_20240627>  # BIRD dev databases (https://bird-bench.github.io/)
python code/prepare.py tpch tpcds bird dirty # pools: candidates, exact atoms, LLM priors (needs llm_keys.json)
python code/stats.py
python code/run_pilot.py                     # RQ1-RQ3
python code/calib_uniform.py                 # RQ4
python code/maint.py tpch tpcds              # RQ5 (SF1)
python code/paper_figs.py                    # tables and figures
```

The shipped pools already contain the LLM priors, so every experiment except `prepare.py` runs without API keys.
JOB/IMDB: download `imdb.tgz` from the Join Order Benchmark (https://event.cwi.nl/da/job/imdb.tgz) and the schema
from https://github.com/gregrahn/join-order-benchmark, then run `code/job_load.py` and `python code/prepare.py job`.
SF10: generate with `code/gen_tpc.py`-style `CALL dbgen(sf=10)` / `dsdgen(sf=10)`, run `code/sf10_augment.py`, then
`code/maint_scale.py`. Paths of the large files are given by `<DATA_DIR>` placeholders.

## Not included

The production BI platform of Section V-H belongs to a company; its data, schema names and platform code are not
released. The paper reports only aggregate counts from it.

## License

MIT (see `LICENSE`).
