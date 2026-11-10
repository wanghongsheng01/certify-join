"""Catalog-style statistics for the non-probing baselines: HyperLogLog distinct counts, value ranges, null counts
(what an optimizer catalog keeps), plus a 1% Bernoulli sample containment test used by the Auto-BI-style scorer."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instances as I

OUT = os.path.join(I.ROOT, "results", "stats")


def colexpr(cols):
    return '"' + cols[0] + '"' if len(cols) == 1 else "concat_ws('|', " + ", ".join(f'"{c}"' for c in cols) + ")"


def compute(inst, edges):
    # enterprise statistics contain data values (min/max): keep them in the private folder
    out = os.path.join(I.PRIV, "stats") if inst["name"].startswith("ent") else OUT
    path = os.path.join(out, inst["name"] + ".json")
    st = json.load(open(path)) if os.path.exists(path) else {}
    con = inst["con"]
    for e in edges:
        R, X, S, Y, Tm = e
        for t, cols in ((R, X), (S, Y)):
            k = json.dumps([t, list(cols)])
            if k in st:
                continue
            ex = colexpr(cols)
            r = con.execute(f'SELECT approx_count_distinct({ex}), count({ex}), min({ex})::VARCHAR, max({ex})::VARCHAR, count(*) '
                            f'FROM "{t}"').fetchone()
            st[k] = dict(ndv=r[0], nonnull=r[1], min=r[2], max=r[3], rows=r[4])
        k = "C" + json.dumps(e)
        if k not in st:
            cond = " AND ".join(f'r."{x}" = s."{y}"' for x, y in zip(X, Y))
            nn = " AND ".join(f'r."{x}" IS NOT NULL' for x in X)
            r = con.execute(f'SELECT count(*), count(*) FILTER (WHERE EXISTS (SELECT 1 FROM "{S}" s WHERE {cond})) '
                            f'FROM (SELECT * FROM "{R}" USING SAMPLE 1% (bernoulli, 11)) r WHERE {nn}').fetchone()
            st[k] = dict(sample=r[0], contained=r[1])
    os.makedirs(out, exist_ok=True)
    json.dump(st, open(path, "w"))
    return st


if __name__ == "__main__":
    import policies as Pm
    for f in sorted(os.listdir(os.path.join(I.ROOT, "results", "pool"))):
        name = f[:-5]
        if (os.path.exists(os.path.join(OUT, f)) or os.path.exists(os.path.join(I.PRIV, "stats", f))) and "--force" not in sys.argv:
            continue
        rec = json.load(open(os.path.join(I.ROOT, "results", "pool", f)))
        if name == "tpch": inst = I.load_tpch()
        elif name == "tpcds": inst = I.load_tpcds()
        elif name == "job": inst = I.load_job()
        elif name.startswith("bird_"): inst = I.load_bird(name[5:])
        elif "_dirty" in name:
            import duckdb
            inst = dict(name=name, con=duckdb.connect(os.path.join(I.DATA, name + ".duckdb"), read_only=True))
        elif name.startswith("ent"):
            import enterprise
            inst = enterprise.load()
        compute(inst, [Pm.T(d["e"]) for d in rec["edges"]])
        print("stats", name, flush=True)
