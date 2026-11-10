"""Exact profiling baseline: SPIDER (unary IND) + HyUCC (UCC) + MIND (n-ary IND, composite candidates only), all from
the Desbordante library, run on the tables/columns of each candidate pool.

An edge R.X -> S.Y is certified when U(S,Y) holds (some discovered minimal UCC of S is a subset of Y) and I(R,X,S,Y)
holds (the IND was discovered). Profiling is exact, so the certified set should equal the safe set; the point of the
baseline is its cost (every touched table is read in full) compared with full verification and with Certify-Join.

Semantics aligned with the atoms (code/atoms.py):
  * IND: NULLs on the dependent side are ignored. SPIDER's own first phase extracts the sorted distinct values of each
    attribute; we do this extraction in DuckDB (distinct non-null values cast to VARCHAR) and feed one single-column
    table per attribute to Desbordante's Spider, so NULLs never enter the IND test.
  * UCC: NULL != NULL (SQL UNIQUE semantics). Desbordante maps a pandas None to the string "None", so every NULL is
    replaced by a row-unique token before loading; this is equivalent to the atom's "unique among rows whose key
    columns are all non-null".
  * Temporal edges (SCD validity-interval predicates) are neither UCCs nor INDs: Desbordante cannot certify them;
    they are reported as unsupported and left uncertified.

Writes results/baseline_profilers.csv and the "profilers" section of results/baselines2_detail.json.
Other baselines (HoPF, Auto-BI) import extract()/profile() from here."""
import csv, json, os, sys, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duckdb
import numpy as np
import pandas as pd
import policies as Pm

ROOT = Pm.ROOT
RES = os.path.join(ROOT, "results")
DETAIL = os.path.join(RES, "baselines2_detail.json")
SEP = "\x1f"


FIXED = ["tpch", "tpch_dirty0", "tpch_dirty1", "tpch_dirty2", "tpcds", "tpcds_dirty0", "tpcds_dirty1"]


def instances():
    """The 18 pilot instances of this study (TPC-H x4, TPC-DS x3, 11 BIRD DBs); never enterprise pools."""
    fs = sorted(f[:-5] for f in os.listdir(os.path.join(RES, "pool")) if not f.startswith("ent"))
    return [n for n in fs if n in FIXED or n.startswith("bird_")]


def load_pool(name):
    assert not name.startswith("ent")
    rec = json.load(open(os.path.join(RES, "pool", name + ".json")))
    return Pm.Pool(rec)


def connect(name):
    assert not name.startswith("ent")
    if name == "job":  # JOB/IMDB lives on the external SSD (exFAT): read-only
        import instances as I
        con = duckdb.connect(I.JOB_DB, read_only=True)
        con.execute("SET memory_limit='3GB'")
        return con
    if name.startswith("bird_"):
        p = os.path.join(ROOT, "data", "bird", name[5:] + ".duckdb")
    else:
        p = os.path.join(ROOT, "data", name + ".duckdb")
    return duckdb.connect(p, read_only=True)


def q(c):
    return '"' + c.replace('"', '""') + '"'


def colexpr(cols):
    if len(cols) == 1:
        return f"CAST({q(cols[0])} AS VARCHAR)"
    return "concat_ws(chr(31), " + ", ".join(f"CAST({q(c)} AS VARCHAR)" for c in cols) + ")"


def pool_columns(P):
    """Per table: the columns used by any candidate edge (both roles), in schema order; plus composite column lists."""
    cols = defaultdict(set)
    comp = set()
    for (R, X, S, Y, Tm) in P.edges:
        cols[R] |= set(X)
        cols[S] |= set(Y)
        if len(X) > 1:
            comp.add((R, X))
            comp.add((S, Y))
    out = {}
    for t, cs in cols.items():
        order = list(P.rec["tables"][t]["cols"])
        out[t] = [c for c in order if c in cs]
    return out, sorted(comp)


def extract(con, table, cols):
    """Distinct non-null values (VARCHAR) of a column list; composite values joined with chr(31)."""
    nn = " AND ".join(f"{q(c)} IS NOT NULL" for c in cols)
    ex = colexpr(cols)
    r = con.execute(f"SELECT DISTINCT {ex} AS v FROM {q(table)} WHERE {nn}").fetchall()
    return [x[0] for x in r]


def is_int_col(P, t, c):
    return "INT" in str(P.rec["tables"][t]["cols"][c]).upper()


def extract_np(con, table, col):
    """Integer column: sorted distinct non-null values as an int64 array (memory-light path used for JOB/IMDB)."""
    a = con.execute(f"SELECT DISTINCT {q(col)} AS v FROM {q(table)} WHERE {q(col)} IS NOT NULL ORDER BY 1").fetchnumpy()["v"]
    return np.asarray(a, dtype=np.int64)


def ucc_frame_np(con, P, table, cols):
    """As ucc_frame, but integer columns stay integers; NULLs become row-unique negative integers (JOB ids are >= 0)."""
    sel = ", ".join((f"coalesce(CAST({q(c)} AS BIGINT), -1 - CAST(rowid AS BIGINT)) AS {q(c)}" if is_int_col(P, table, c)
                     else f"coalesce(CAST({q(c)} AS VARCHAR), '#NULL#' || CAST(rowid AS VARCHAR)) AS {q(c)}") for c in cols)
    return con.execute(f"SELECT {sel} FROM {q(table)}").fetchdf()


def ucc_frame(con, table, cols):
    """Table projection for UCC discovery; NULLs replaced by row-unique tokens (NULL != NULL)."""
    sel = ", ".join(f"coalesce(CAST({q(c)} AS VARCHAR), '#NULL#' || CAST(rowid AS VARCHAR)) AS {q(c)}" for c in cols)
    return con.execute(f"SELECT {sel} FROM {q(table)}").fetchdf()


def profile(name, P=None, ucc_algo="HyUCC", keep_values=True, numeric=None, cache=None):
    """Run SPIDER + UCC (+ MIND for composite candidates) on the pool's columns. Returns a dict with discovered
    INDs/UCCs, distinct value lists (reused by HoPF/Auto-BI), timings and rows read.
    numeric=True (default for JOB): integer columns are handled as int64 arrays instead of VARCHAR lists (same
    equality semantics for integer keys, far less memory). cache: pickle path for the discovery result (no values);
    when it exists the Desbordante run is not repeated and the recorded timings are returned."""
    import desbordante as D
    import pickle
    P = P or load_pool(name)
    numeric = (name == "job") if numeric is None else numeric
    cols, comp = pool_columns(P)
    rows = P.rows
    if cache and os.path.exists(cache):
        res = pickle.load(open(cache, "rb"))
        res["cached"] = True
        if keep_values:
            con = connect(name)
            vals = {}
            for t in sorted(cols):
                for c in cols[t]:
                    vals[(t, (c,))] = extract_np(con, t, c) if numeric and is_int_col(P, t, c) else extract(con, t, (c,))
            con.close()
            res["vals"] = vals
        return res
    con = connect(name)
    res = dict(inst=name, secs={}, rows_read={})
    # ---- SPIDER: per-attribute sorted distinct value extraction + Spider
    t0 = time.time()
    attrs, vals = [], {}
    for t in sorted(cols):
        for c in cols[t]:
            v = extract_np(con, t, c) if numeric and is_int_col(P, t, c) else extract(con, t, (c,))
            vals[(t, (c,))] = v
            attrs.append((t, c))
    t1 = time.time()
    frames = [pd.DataFrame({"v": vals[(t, (c,))]}) if isinstance(vals[(t, (c,))], np.ndarray)
              else pd.DataFrame({"v": pd.Series(vals[(t, (c,))], dtype=object)}) for (t, c) in attrs]
    nonempty = [i for i, f in enumerate(frames) if len(f)]
    alg = D.ind.algorithms.Spider()
    alg.load_data(tables=[frames[i] for i in nonempty])
    alg.execute()
    uinds = set()
    for ind in alg.get_inds():
        l, r = ind.get_lhs(), ind.get_rhs()
        a, b = attrs[nonempty[l.table_index]], attrs[nonempty[r.table_index]]
        if a[0] != b[0] or a[1] != b[1]:
            uinds.add((a[0], (a[1],), b[0], (b[1],)))
    # an all-NULL attribute has no non-null value and is vacuously included in every attribute (atom semantics;
    # Metanome's SPIDER treats empty columns the same way); Desbordante is not given empty tables
    for i, a in enumerate(attrs):
        if not len(frames[i]):
            for b in attrs:
                if a != b:
                    uinds.add((a[0], (a[1],), b[0], (b[1],)))
    del alg, frames
    if not keep_values:
        vals = {k: v for k, v in vals.items() if len(k[1]) > 1}
    t2 = time.time()
    res["secs"]["spider_extract"] = t1 - t0
    res["secs"]["spider_algo"] = t2 - t1
    res["rows_read"]["spider"] = sum(rows[t] for t in cols)
    # ---- n-ary INDs for composite candidates (MIND on the distinct composite tuples)
    t0 = time.time()
    ninds = set()
    comp_edges = sorted({(R, X, S, Y) for (R, X, S, Y, Tm) in P.edges if len(X) > 1})
    for (R, X, S, Y) in comp_edges:
        fr = []
        for t, cs in ((R, X), (S, Y)):
            if (t, cs) not in vals:
                vals[(t, cs)] = extract(con, t, cs)
            v = vals[(t, cs)]
            fr.append(pd.DataFrame([x.split(SEP) for x in v], columns=list(cs)) if v else pd.DataFrame(columns=list(cs)))
        if len(fr[0]) == 0:
            continue
        m = D.ind.algorithms.Mind()
        m.load_data(tables=fr)
        m.execute(max_arity=len(X))
        for ind in m.get_inds():
            l, r = ind.get_lhs(), ind.get_rhs()
            if l.table_index == 0 and r.table_index == 1 and len(l.column_indices) == len(X):
                lc = tuple(X[i] for i in l.column_indices)
                rc = tuple(Y[i] for i in r.column_indices)
                if lc == X and rc == Y:
                    ninds.add((R, X, S, Y))
    res["secs"]["mind"] = time.time() - t0
    res["rows_read"]["mind"] = sum(rows[R] + rows[S] for (R, X, S, Y) in comp_edges)
    # ---- UCC discovery per table
    t0 = time.time()
    uccs = {}
    for t in sorted(cols):
        df = ucc_frame_np(con, P, t, cols[t]) if numeric else ucc_frame(con, t, cols[t])
        if len(df) == 0:
            uccs[t] = [tuple(cols[t][:1])]
            continue
        u = getattr(D.ucc.algorithms, ucc_algo)()
        ta = time.time()
        u.load_data(table=df)
        tb = time.time()
        u.execute()
        res["secs"]["ucc_exec"] = res["secs"].get("ucc_exec", 0.0) + time.time() - tb
        uccs[t] = [tuple(cols[t][i] for i in x.indices) for x in u.get_uccs()]
        del u, df
    res["secs"]["ucc"] = time.time() - t0 - res["secs"].get("ucc_exec", 0.0)
    res["rows_read"]["ucc"] = sum(rows[t] for t in cols)
    con.close()
    res.update(cols=cols, uccs=uccs, uinds=uinds, ninds=ninds)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        pickle.dump({k: v for k, v in res.items() if k != "vals"}, open(cache, "wb"))
    if keep_values:
        res["vals"] = vals
    return res


def discovered(P, prof):
    """Atom outcomes implied by the profiling result (None = not decidable by UCC/IND discovery)."""
    out = {}
    for e in P.edges:
        R, X, S, Y, Tm = e
        ua, ia = P.ea[e]
        if Tm:
            out[ua] = out.get(ua)
            out[ia] = None
            continue
        out[ua] = any(set(u) <= set(Y) for u in prof["uccs"].get(S, []))
        if len(X) == 1:
            out[ia] = (R, X, S, Y) in prof["uinds"]
        else:
            out[ia] = (R, X, S, Y) in prof["ninds"]
    return out


def update_detail(section, inst, payload):
    if os.environ.get("NO_DETAIL"):
        return
    import fcntl
    import tempfile
    with open(os.path.join(tempfile.gettempdir(), "baselines2_detail.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = json.load(open(DETAIL)) if os.path.exists(DETAIL) else {}
        d.setdefault(section, {})[inst] = payload
        with open(DETAIL, "w") as f:
            json.dump(d, f, indent=1, default=str)


def run(names):
    out = os.path.join(RES, "baseline_profilers" + os.environ.get("OUT_SUFFIX", "") + ".csv")
    rows_out = []
    for name in names:
        P = load_pool(name)
        W = P.workload()
        cdir = os.environ.get("PROFILE_CACHE")
        prof = profile(name, P, keep_values=False, cache=os.path.join(cdir, name + ".pkl") if cdir else None)
        disc = discovered(P, prof)
        cert = [e for e in P.edges if disc.get(P.ea[e][0]) and disc.get(P.ea[e][1])]
        n_temporal = sum(1 for e in P.edges if e[4])
        dec = [a for a, v in disc.items() if v is not None]
        atom_agree = sum(disc[a] == P.truth[a] for a in dec)
        mism = [a for a in dec if disc[a] != P.truth[a]]
        edge_agree = sum((e in set(cert)) == P.safe[e] for e in P.edges)
        ev = P.evaluate(W, cert, "SPIDER+UCC (Desbordante)", P.full_scan, dict(cost_model="scan"))
        full_verify_secs = sum(P.A[a]["secs"] for a in {a for e in P.edges for a in P.ea[e]})
        s = prof["secs"]
        tot = sum(s.values())
        fused_rows = P.full_scan  # one fused pass over every touched table
        comps = [("Spider (unary IND)", s["spider_extract"] + s["spider_algo"], prof["rows_read"]["spider"]),
                 ("HyUCC (UCC)", s["ucc"] + s.get("ucc_exec", 0.0), prof["rows_read"]["ucc"]),
                 ("Mind (n-ary IND, composite cand.)", s["mind"], prof["rows_read"]["mind"])]
        base = dict(instance=name, full_cost=P.full_cost, full_scan=P.full_scan,
                    full_verify_secs=round(full_verify_secs, 3), n_edges=len(P.edges), n_safe=sum(P.safe.values()))
        for alg, sec, rr in comps:
            rows_out.append(dict(base, algorithm=alg, seconds=round(sec, 3), rows_read=rr,
                                 ratio_full_cost=rr / P.full_cost, ratio_full_scan=rr / P.full_scan))
        rows_out.append(dict(base, algorithm="SPIDER+UCC (Desbordante)", seconds=round(tot, 3),
                             extract_secs=round(s["spider_extract"], 3), rows_read=fused_rows,
                             rows_read_as_run=sum(prof["rows_read"].values()),
                             ratio_full_cost=fused_rows / P.full_cost, ratio_full_scan=fused_rows / P.full_scan,
                             secs_ratio_full_verify=tot / max(1e-9, full_verify_secs),
                             n_edges_certified=len(cert), n_temporal_unsupported=n_temporal,
                             agreement=edge_agree / len(P.edges), atom_agreement=atom_agree / max(1, len(dec)),
                             n_atom_mismatch=len(mism), coverage=ev["coverage"], fcr_card=ev["fcr_card"],
                             fcr_sem=ev["fcr_sem"], n_certified_q=ev["n_certified"], n_answerable=ev["n_answerable"],
                             n_ok=ev["n_ok"], n_card=ev["n_card"], n_sem=ev["n_sem"], n_queries=ev["n_queries"]))
        update_detail("profilers", name, dict(
            secs=s, rows_read=prof["rows_read"], n_uinds=len(prof["uinds"]), n_ninds=len(prof["ninds"]),
            uccs={t: [list(u) for u in v] for t, v in prof["uccs"].items()}, mismatched_atoms=mism[:50],
            certified=[P.ename[e] for e in cert], evaluate=ev))
        print(name, "secs", round(tot, 1), "cert", len(cert), "safe", sum(P.safe.values()), "atom mismatches", len(mism),
              "cov", round(ev["coverage"], 3), "fcr", ev["fcr_card"], flush=True)
    keys = []
    for r in rows_out:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_out)


def pool_families():
    """Pool the three re-implemented baselines by family exactly as summarize.pooled() pools the zero-probe baselines
    (sums of n_ok / n_answerable / n_certified / n_card / n_sem over instances), next to the existing zero-probe rows
    of results/main_zero.csv for reference. Writes the "pooled" section of the detail file and prints markdown."""
    import summarize as SM
    parts = []
    for f in ["baseline_autobi.csv", "baseline_hopf.csv"]:
        p = os.path.join(RES, f)
        if os.path.exists(p):
            parts.append(pd.read_csv(p))
    p = os.path.join(RES, "baseline_profilers.csv")
    if os.path.exists(p):
        z = pd.read_csv(p)
        z = z[z.algorithm == "SPIDER+UCC (Desbordante)"].rename(columns={"instance": "inst", "algorithm": "method",
                                                                         "n_certified_q": "n_certified"})
        z["n_card_eps"] = z["n_card"]
        z["cost_frac"] = z["ratio_full_scan"]
        parts.append(z)
    x = pd.concat(parts, ignore_index=True)
    x = x[~x.inst.str.startswith("ent")]
    x["family"] = x["inst"].map(SM.fam)
    g = SM.pooled(x, ["family", "method"]).reset_index()
    g["n_inst"] = g.apply(lambda r: x[(x.family == r.family) & (x.method == r.method)].inst.nunique(), axis=1)
    ref = None
    zp = os.path.join(RES, "main_zero.csv")
    if os.path.exists(zp):
        zz = pd.read_csv(zp)
        zz = zz[~zz.inst.str.startswith("ent")]
        zz["family"] = zz["inst"].map(SM.fam)
        ref = SM.pooled(zz, ["family", "method"]).reset_index()
    order = {f: i for i, f in enumerate(SM.ORDER)}
    g = g.sort_values(["family", "method"], key=lambda c: c.map(order) if c.name == "family" else c)
    cols = ["family", "method", "n_inst", "n_certified", "coverage", "fcr_card", "fcr_sem", "cost_frac"]
    txt = "## Re-implemented baselines (pooled)\n" + SM.md(g[cols])
    if ref is not None:
        ref = ref.sort_values(["family", "method"], key=lambda c: c.map(order) if c.name == "family" else c)
        txt += "\n\n## Existing zero-probe baselines (results/main_zero.csv, pooled, reference)\n" + \
            SM.md(ref[["family", "method", "n_certified", "coverage", "fcr_card", "fcr_sem"]])
    if not os.environ.get("NO_DETAIL"):
        import fcntl, tempfile
        with open(os.path.join(tempfile.gettempdir(), "baselines2_detail.lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = json.load(open(DETAIL)) if os.path.exists(DETAIL) else {}
            d["pooled"] = g[cols].to_dict(orient="records")
            if ref is not None:
                d["pooled_reference_main_zero"] = ref[["family", "method", "n_certified", "coverage", "fcr_card",
                                                       "fcr_sem"]].to_dict(orient="records")
            with open(DETAIL, "w") as f:
                json.dump(d, f, indent=1, default=str)
    print(txt)
    return g


if __name__ == "__main__":
    if sys.argv[1:2] == ["--pool"]:
        pool_families()
    else:
        run(sys.argv[1:] or instances())
