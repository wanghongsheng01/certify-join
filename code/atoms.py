"""Verification atoms behind a candidate join edge e = R.X -> S.Y [@ temporal predicate].

  U(S, Y[, valid-from, valid-to])  key atom: no two rows of S share Y (for SCD tables: no two rows share Y with
                                   overlapping validity intervals). Fan-out freedom of e needs U.
  I(R, X, S, Y[, T])               inclusion atom: every R row with non-null X matches some S row (inside the
                                   validity interval if temporal). Fact preservation under INNER JOIN needs I.

Probe cost model (rows read): U costs |S|; exact I costs |R| + |S|; a sampled I probe on m rows of R costs m + |S|.
All truths are computed once with exact SQL and replayed by the policies (label once, replay many).
"""
import json, os, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def q(c):
    return '"' + c + '"'


def edge_atoms(e):
    R, X, S, Y, T = e
    u = ("U", S, Y, (T[1], T[2]) if T else None)
    i = ("I", R, X, S, Y, T)
    return u, i


def akey(a):
    return json.dumps(a)


def _cond(X, Y, ra="r", sa="s"):
    return " AND ".join(f"{ra}.{q(x)} = {sa}.{q(y)}" for x, y in zip(X, Y))


def probe_U(con, S, Y, T):
    nn = " AND ".join(f"{q(y)} IS NOT NULL" for y in Y)
    ys = ", ".join(q(y) for y in Y)
    if T is None:
        r = con.execute(f'SELECT count(*), coalesce(sum(c), 0) FROM (SELECT {ys}, count(*) c FROM "{S}" WHERE {nn} '
                        f'GROUP BY {ys} HAVING count(*) > 1)').fetchone()
        return dict(dup_keys=r[0], dup_rows=int(r[1]))
    f, t = T
    jc = " AND ".join(f"a.{q(y)} = b.{q(y)}" for y in Y)
    r = con.execute(f'''WITH s AS (SELECT row_number() OVER () rid, * FROM "{S}" WHERE {nn})
        SELECT count(DISTINCT a.rid) FROM s a JOIN s b ON {jc} AND a.rid < b.rid
        AND a.{q(f)} <= coalesce(b.{q(t)}, DATE '9999-12-31') AND b.{q(f)} <= coalesce(a.{q(t)}, DATE '9999-12-31')''').fetchone()
    return dict(dup_keys=r[0], dup_rows=r[0])


def probe_I(con, R, X, S, Y, T):
    nn = " AND ".join(f"r.{q(x)} IS NOT NULL" for x in X)
    cond = _cond(X, Y)
    if T:
        d, f, t = T
        cond += f" AND r.{q(d)} BETWEEN s.{q(f)} AND coalesce(s.{q(t)}, DATE '9999-12-31')"
    r = con.execute(f'SELECT count(*) FILTER (WHERE {nn}), count(*) FILTER (WHERE {nn} AND NOT EXISTS '
                    f'(SELECT 1 FROM "{S}" s WHERE {cond})) FROM "{R}" r').fetchone()
    return dict(nonnull=r[0], orphans=r[1])


def fanout(con, e):
    """Ground-truth inflation of COUNT(*) when R is inner-joined with S along e (|R join S| / |R|)."""
    R, X, S, Y, T = e
    cond = _cond(X, Y)
    if T:
        d, f, t = T
        cond += f" AND r.{q(d)} BETWEEN s.{q(f)} AND coalesce(s.{q(t)}, DATE '9999-12-31')"
        r = con.execute(f'SELECT count(*) FROM "{R}" r JOIN "{S}" s ON {cond}').fetchone()[0]
    else:
        ys = ", ".join(q(y) for y in Y)
        jc = _cond(X, Y, "r", "g")
        r = con.execute(f'SELECT coalesce(sum(g.c), 0) FROM "{R}" r JOIN (SELECT {ys}, count(*) c FROM "{S}" GROUP BY {ys}) g '
                        f'ON {jc}').fetchone()[0]
    return int(r)


def compute(inst, edges, cache=True):
    path = os.path.join(ROOT, "results", "atoms", inst["name"] + ".json")
    old = json.load(open(path)) if cache and os.path.exists(path) else {}
    con, T = inst["con"], inst["tables"]
    out = dict(old)
    for e in edges:
        u, i = edge_atoms(e)
        R, X, S, Y, Tm = e
        for a in (u, i):
            k = akey(a)
            if k in out:
                continue
            t0 = time.time()
            if a[0] == "U":
                r = probe_U(con, S, Y, a[3])
                r.update(truth=r["dup_keys"] == 0, cost=T[S]["rows"], rows_S=T[S]["rows"])
            else:
                r = probe_I(con, R, X, S, Y, Tm)
                r.update(truth=r["orphans"] == 0, cost=T[R]["rows"] + T[S]["rows"], rows_R=T[R]["rows"],
                         rows_S=T[S]["rows"])
            r["secs"] = round(time.time() - t0, 4)
            out[k] = r
        ek = "E" + json.dumps(e)
        if ek not in out:
            out[ek] = dict(join_rows=fanout(con, e), rows_R=T[R]["rows"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"))
    return out
