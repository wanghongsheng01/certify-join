"""Executed sampled inclusion probes (review fix E1). For every inclusion atom that Certify-Join+S samples (R has more
than 10m rows), draw K independent samples of m uniform row identifiers among the rows of R with non-null X (with
replacement; rejection on null X), look up their partners in S on the database and record whether an orphan was
seen. Certify-Join+S then takes the outcome of its sampled probes from these executed samples (draw = seed mod K)
instead of drawing it with probability (1-rho)^m. Writes results/sampled_exec/<instance>.json."""
import json, math, os, sys, time, zlib
import duckdb
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instances as I
from run_pilot import ROOT, SAMPLE
from decide_live import db_path

M = int(math.ceil(math.log(1 / SAMPLE[1]) / SAMPLE[0]))
K = int(os.environ.get("K", "5"))
OUT = os.path.join(ROOT, "results", "sampled_exec")


def q(c):
    return '"' + c + '"'


def one_sample(con, R, X, S, Y, T, n, nonnull, rng):
    """Returns (orphan seen, orphans among the m sampled rows, rows drawn)."""
    nn = " AND ".join(f"r.{q(x)} IS NOT NULL" for x in X)
    cond = " AND ".join(f"r.{q(x)} = s.{q(y)}" for x, y in zip(X, Y))
    if T:
        d, f, t = T
        cond += f" AND r.{q(d)} BETWEEN s.{q(f)} AND coalesce(s.{q(t)}, DATE '9999-12-31')"
    got, orph, drawn = 0, 0, 0
    while got < M:
        b = int(math.ceil((M - got) * n / max(1, nonnull) * 1.2)) + 64
        ids = rng.integers(0, n, b)
        drawn += b
        con.register("_ids", pd.DataFrame({"pos": np.arange(b), "rid": ids.astype(np.int64)}))
        res = con.execute(f'SELECT i.pos, ({nn}) AS nn, ({nn} AND NOT EXISTS (SELECT 1 FROM "{S}" s WHERE {cond})) AS o '
                          f'FROM _ids i JOIN "{R}" r ON r.rowid = i.rid ORDER BY i.pos').fetchnumpy()
        keep = np.asarray(res["nn"], dtype=bool)
        o = np.asarray(res["o"], dtype=bool)[keep][: M - got]
        got += len(o)
        orph += int(o.sum())
    return orph > 0, orph, drawn


def run(name):
    rec = json.load(open(os.path.join(ROOT, "results", "pool", name + ".json")))
    path = db_path(name)
    if not os.path.exists(path):
        print(name, "database missing", path, flush=True)
        return
    os.makedirs(OUT, exist_ok=True)
    out_p = os.path.join(OUT, name + ".json")
    done = json.load(open(out_p)) if os.path.exists(out_p) else {}
    con = duckdb.connect(path, read_only=True)
    t_all = time.time()
    for k, v in sorted(rec["atoms"].items()):
        if not k.startswith('["I"') or v["rows_R"] <= 10 * M or (k in done and len(done[k]["hit"]) >= K):
            continue
        _, R, X, S, Y, T = json.loads(k)
        n = con.execute(f'SELECT count(*) FROM "{R}"').fetchone()[0]
        rng = np.random.default_rng([zlib.crc32(name.encode()), zlib.crc32(k.encode())])
        hit, orph, secs = [], [], []
        for _ in range(K):
            t0 = time.time()
            if v["nonnull"] == 0:
                h, o = False, 0
            else:
                h, o, _ = one_sample(con, R, X, S, Y, T, n, v["nonnull"], rng)
            secs.append(round(time.time() - t0, 4)); hit.append(bool(h)); orph.append(o)
        done[k] = dict(hit=hit, orphans_seen=orph, secs=secs, m=M, rho=v["orphans"] / max(1, v["nonnull"]))
        json.dump(done, open(out_p + ".tmp", "w"))
        os.replace(out_p + ".tmp", out_p)
    con.close()
    print(name, len(done), "atoms", round(time.time() - t_all, 1), "s", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(f[:-5] for f in os.listdir(os.path.join(ROOT, "results", "pool")) if not f.startswith("ent"))
    for name in names:
        run(name)
