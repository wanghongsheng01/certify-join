"""Calibration of sampled inclusion certificates with uniform sampling by row identifier (with replacement): for every
I atom with 0 < rho <= 2% on a table with more than 10m rows, draw m uniform row ids among the rows with non-null X
(numpy, fresh seed per repetition), check their partners in S, and certify iff no orphan is seen. This replaces the
engine's reservoir sample (calib.py), which over-represented row groups appended after the initial load."""
import csv, json, math, os, sys
import duckdb, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instances as I, policies as Pm

EPS, DELTA, REPS = 1e-3, 0.05, 100
M = int(math.ceil(math.log(1 / DELTA) / EPS))
out = open(os.path.join(Pm.ROOT, "results", "calib_uniform.csv"), "w", newline="")
w = csv.writer(out)
w.writerow(["inst", "atom", "rows_R", "orphans", "rho", "m", "reps", "certified", "emp_rate", "theory", "rho_gt_eps"])
for f in sorted(os.listdir(os.path.join(Pm.ROOT, "results", "pool"))):
    rec = json.load(open(os.path.join(Pm.ROOT, "results", "pool", f)))
    name = rec["name"]
    path = (I.JOB_DB if name == "job" else os.path.join(I.DATA, name + ".duckdb") if not name.startswith("bird_")
            else os.path.join(I.DATA, "bird", name[5:] + ".duckdb"))
    if name.startswith("ent") or not os.path.exists(path):
        continue
    con = duckdb.connect(path, read_only=True)
    for k, v in rec["atoms"].items():
        if not k.startswith('["I"') or v["orphans"] == 0 or v["rows_R"] <= 10 * M or v["orphans"] > 0.02 * v["nonnull"]:
            continue
        _, R, X, S, Y, Tm = json.loads(k)
        if Tm:
            continue
        nn = " AND ".join(f'"{x}" IS NOT NULL' for x in X)
        # orphan flag per non-null row, computed once; sampling then draws uniform positions of this vector
        flag = con.execute(f'SELECT (s.k IS NULL) AS o FROM (SELECT {", ".join(chr(34) + x + chr(34) for x in X)} FROM "{R}" WHERE {nn}) r '
                           f'LEFT JOIN (SELECT DISTINCT {", ".join(chr(34) + y + chr(34) for y in Y)}, 1 AS k FROM "{S}") s ON '
                           + " AND ".join(f'r."{x}" = s."{y}"' for x, y in zip(X, Y))).fetchnumpy()["o"]
        flag = np.asarray(flag, dtype=bool)
        rho = v["orphans"] / v["nonnull"]
        assert abs(flag.sum() - v["orphans"]) <= 0, (name, k, flag.sum(), v["orphans"])
        rng = np.random.default_rng(abs(hash((name, k))) % 2**32)
        cert = sum(not flag[rng.integers(0, len(flag), M)].any() for _ in range(REPS))
        w.writerow([name, k, v["rows_R"], v["orphans"], rho, M, REPS, cert, cert / REPS, (1 - rho) ** M, rho > EPS])
        out.flush()
        print(name, R, X, "rho=%.2e" % rho, "emp=%.2f theory=%.2f" % (cert / REPS, (1 - rho) ** M), flush=True)
