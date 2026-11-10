"""Maintenance at SF10 (instances on the external disk). Two batch regimes per instance:
  vol  - the same absolute change volume as the SF1 runs (fractions / 10), so |D| grows 10x and |dD| stays fixed;
  prop - batches proportional to the current table size (as in the SF1 runs), so |dD| grows with |D|.
Writes results/maint/<inst>_sf10<regime>_s<seed>.json via maint.run."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maint

SSD = "<DATA_DIR>/data"
if __name__ == "__main__":
    insts = sys.argv[1:] or ["tpch", "tpcds"]
    for nm in insts:
        for reg, af, df in [("vol", 2e-4, 1e-4), ("prop", 2e-3, 1e-3)]:
            for s in [0]:
                r = maint.run(nm, n_batch=10, seed=s, app_frac=af, dim_frac=df, data=os.path.join(SSD, f"{nm}_sf10.duckdb"),
                              tag=f"_sf10{reg}", keep=True)
                print("RESULT", json.dumps({k: r[k] for k in ["inst", "tag", "n_atoms", "cost", "secs", "exposure", "build_rows", "build_secs"]}), flush=True)
