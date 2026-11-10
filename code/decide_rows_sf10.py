"""Rows to decide the whole workload on the SF10 pools (same quantities as Table II's row columns at SF1)."""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policies as Pm
import extra_baselines as X
from run_pilot import load, ROOT, SAMPLE

with open(os.path.join(ROOT, "results", "decide_rows_sf10.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["inst", "keyfirst_rows", "wlfull_rows", "cj_rows", "cjs_rows", "n_answerable", "cj_coverage"])
    for name in ["tpch_sf10", "tpcds_sf10"]:
        P, st = load(name); W = P.workload()
        S1, _ = X.keyfirst(P, 1e30, X.catalog_edges(P)); S2, _ = X.keyfirst(P, 1e30, X.workload_edges(P, W))
        S3 = Pm.certify_join(P, W, 1e30); S4 = Pm.certify_join(P, W, 1e30, sample=SAMPLE)
        r = P.evaluate(W, S3.certified(), "cj", S3.spent)
        w.writerow([name] + [round(S.spent / P.full_cost, 4) for S in (S1, S2, S3, S4)] + [r["n_answerable"], r["coverage"]])
        print(name, [round(S.spent / P.full_cost, 4) for S in (S1, S2, S3, S4)], r["coverage"], r["fcr_card"])
