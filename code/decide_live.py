"""Seconds to decide the whole workload, end to end (review fixes G1/G2).

For every method: planning time measured live (the algorithm run with probe outcomes taken from the pool), plus probe
time. Exact probes are charged the wall-clock time measured when the pool was built (atom field "secs", one exact SQL
probe per atom). Sampled inclusion probes of Certify-Join+S are executed live on the database: m uniform row ids of R
(with replacement) are drawn and their partners in S are checked, and that time is charged.
Writes results/decide_live.csv (one row per instance)."""
import csv, json, math, os, sys, time
import duckdb
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instances as I
import policies as Pm
import extra_baselines as X
from run_pilot import load, ROOT, SAMPLE

SSD = "<DATA_DIR>/data"


def db_path(name):
    if name.startswith("bird_"):
        return os.path.join(I.DATA, "bird", name[5:] + ".duckdb")
    if name == "job":
        return I.JOB_DB
    if name.endswith("_sf10"):
        return os.path.join(SSD, name + ".duckdb")
    return os.path.join(I.DATA, name + ".duckdb")


def sampled_probe_secs(con, a, m, rng):
    """Execute one sampled inclusion probe: m uniform row ids of R among rows with non-null X, partners in S."""
    _, R, X_, S, Y, Tm = json.loads(a)
    nn = " AND ".join(f'"{x}" IS NOT NULL' for x in X_)
    t0 = time.time()
    n = con.execute(f'SELECT count(*) FROM "{R}"').fetchone()[0]
    ids = rng.integers(0, n, m)
    con.execute("CREATE OR REPLACE TEMP TABLE _ids AS SELECT unnest(?::BIGINT[]) AS rid", [ids.tolist()])
    cond = " AND ".join(f'r."{x}" = s."{y}"' for x, y in zip(X_, Y))
    con.execute(f'SELECT count(*) FROM (SELECT * FROM "{R}" WHERE rowid IN (SELECT rid FROM _ids) AND {nn}) r '
                f'WHERE NOT EXISTS (SELECT 1 FROM "{S}" s WHERE {cond})').fetchone()
    return time.time() - t0


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(f[:-5] for f in os.listdir(os.path.join(ROOT, "results", "pool")) if not f.startswith("ent"))
    m = int(math.ceil(math.log(1 / SAMPLE[1]) / SAMPLE[0]))
    out_p = os.path.join(ROOT, "results", os.environ.get("OUT", "decide_live.csv"))
    REP = int(os.environ.get("REP", "1"))  # planning time = minimum over REP runs (robust to concurrent load)
    done = set()
    if os.path.exists(out_p):
        done = {r["inst"] for r in csv.DictReader(open(out_p))}
    fh = open(out_p, "a", newline="")
    w = None
    for name in names:
        if name in done:
            continue
        P, st = load(name)
        W = P.workload()
        all_atoms = sorted({a for e in P.edges for a in P.ea[e]})
        cat, wl = X.catalog_edges(P), X.workload_edges(P, W)
        row = dict(inst=name, full_probe_s=X.secs_of(P, [a for a in all_atoms if a not in P.free]))
        t0 = time.time(); S, pr = X.keyfirst(P, 1e30, cat); row["keyfirst_plan_s"] = time.time() - t0
        row["keyfirst_probe_s"] = X.secs_of(P, pr)
        t0 = time.time(); S, pr = X.keyfirst(P, 1e30, wl); row["wlfull_plan_s"] = time.time() - t0
        row["wlfull_probe_s"] = X.secs_of(P, pr)
        orig = Pm.State
        Pm.State = X.Rec
        try:
            ts = []
            for _ in range(REP):
                t0 = time.time(); S = Pm.certify_join(P, W, 1e30); ts.append(time.time() - t0)
            row["cj_plan_s"] = min(ts)
            row["cj_probe_s"] = X.secs_of(P, S.probed)
            t0 = time.time(); S = Pm.certify_join(P, W, 1e30, sample=SAMPLE); row["cjs_plan_s"] = time.time() - t0
        finally:
            Pm.State = orig
        samp = [a for a in S.probed if a.startswith('["I"') and json.loads(a)[5] is None and P.rows[json.loads(a)[1]] > 10 * m]
        if os.environ.get("LIVE_SAMPLED", "1") == "1":
            con = duckdb.connect(db_path(name), read_only=True)
            rng = np.random.default_rng(0)
            live = sum(sampled_probe_secs(con, a, m, rng) for a in samp)
            con.close()
        else:  # columnar engine: a uniform row-id sample still scans the column, so only exact probes are timed
            live = float("nan")
        row["cjs_probe_s"] = X.secs_of(P, [a for a in S.probed if a not in samp]) + live
        row["cjs_sampled_probes"] = len(samp)
        row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
        if w is None:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if not done:
                w.writeheader()
        w.writerow(row); fh.flush()
        print(row, flush=True)
