"""Run the reimplemented baselines on JOB/IMDB (added after the baseline runs) into results/baseline_<b>_job.csv."""
import csv, os, sys
import duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instances as I

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write(rows, name):
    with open(os.path.join(ROOT, "results", name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    which = sys.argv[1:] or ["sji"]
    if "sji" in which:
        import baseline_sji as B
        P, st = B.load("job")
        W = P.workload()
        con = duckdb.connect(I.JOB_DB, read_only=True, config={"memory_limit": "3GB", "threads": 4})
        acc, tree, det = B.sji(P, st, con, cache={})
        r = P.evaluate(W, acc, B.LABEL, det["rows_read"])
        print("sji job", {k: r[k] for k in ["n_certified", "coverage", "fcr_card", "fcr_sem"]}, det.get("llm_calls"), det.get("dollars"), flush=True)
        write([r], "baseline_sji_job.csv")
