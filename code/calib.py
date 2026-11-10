"""Calibration of sampled inclusion certificates: for every I atom with orphans on a large table, draw m uniform rows
with DuckDB (reservoir sampling, fresh seed per repetition) and certify iff no orphan is seen. Compare the empirical
certification rate with (1-rho)^m and with delta for atoms whose orphan rate rho exceeds eps."""
import csv, json, math, os, sys
import duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instances as I, policies as Pm

EPS, DELTA, REPS = 1e-3, 0.05, 100
M = int(math.ceil(math.log(1 / DELTA) / EPS))
out = open(os.path.join(Pm.ROOT, "results", "calib.csv"), "w", newline="")
w = csv.writer(out)
w.writerow(["inst", "atom", "rows_R", "orphans", "rho", "m", "reps", "certified", "emp_rate", "theory", "rho_gt_eps"])
for f in sorted(os.listdir(os.path.join(Pm.ROOT, "results", "pool"))):
    rec = json.load(open(os.path.join(Pm.ROOT, "results", "pool", f)))
    name = rec["name"]
    path = (os.path.join(I.DATA, name + ".duckdb") if not name.startswith("bird_")
            else os.path.join(I.DATA, "bird", name[5:] + ".duckdb"))
    if name.startswith("ent") or not os.path.exists(path):
        continue
    con = duckdb.connect(path, read_only=True)
    for k, v in rec["atoms"].items():
        # only the informative regime: rho near eps (atoms with rho close to 1 are refuted by any sample)
        if not k.startswith('["I"') or v["orphans"] == 0 or v["rows_R"] <= 10 * M or v["orphans"] > 0.02 * v["nonnull"]:
            continue
        _, R, X, S, Y, Tm = json.loads(k)
        if Tm:
            continue
        cond = " AND ".join(f'r."{x}" = s."{y}"' for x, y in zip(X, Y))
        nn = " AND ".join(f'"{x}" IS NOT NULL' for x in X)
        rho = v["orphans"] / v["nonnull"]
        cert = 0
        for rep in range(REPS):
            n = con.execute(f'SELECT count(*) FROM (SELECT * FROM (SELECT * FROM "{R}" WHERE {nn}) USING SAMPLE {M} ROWS '
                            f'(reservoir, {rep + 1})) r WHERE NOT EXISTS (SELECT 1 FROM "{S}" s WHERE {cond})').fetchone()[0]
            cert += n == 0
        w.writerow([name, k, v["rows_R"], v["orphans"], rho, M, REPS, cert, cert / REPS, (1 - rho) ** M, rho > EPS])
        out.flush()
        print(name, R, X, "rho=%.2e" % rho, "emp=%.2f theory=%.2f" % (cert / REPS, (1 - rho) ** M), flush=True)
