"""Certificate maintenance under a stream of real data changes.

A writable copy of the instance receives batches of SQL changes: fact appends (resampled rows), dimension inserts
with fresh keys, and rare defects (duplicate key insert -> U breaks; orphan foreign key -> I breaks; delete of a
referenced key -> I breaks; overlapping SCD version -> temporal U breaks). After every batch exact SQL probes give the
ground truth. Maintenance methods keep a belief per certified atom:
  full      re-run every certified probe after every batch (rows read = probe costs)
  periodic  full re-verification every k batches
  touched   re-run probes whose tables changed in this batch
  trust     never re-verify
  delta     Certify-Join incremental checks on the change set only (key multiset per U atom, refcount per I atom)
"""
import datetime, json, os, random, shutil, sys, time
from collections import Counter, defaultdict
import duckdb
import numpy as np
import pandas as pd

MAXD = datetime.date(9999, 12, 31)


def enc_cols(arrs):
    """Encode one or two integer key columns as int64 (composite = a * 2^32 + b); strings stay Python objects."""
    if all(np.issubdtype(np.asarray(a).dtype, np.integer) for a in arrs):
        v = np.asarray(arrs[0]).astype(np.int64)
        for a in arrs[1:]:
            v = v * (1 << 32) + np.asarray(a).astype(np.int64)
        return v
    return np.array([tuple(x) if len(arrs) > 1 else x[0] for x in zip(*arrs)], dtype=object)


def enc_one(k):
    if all(isinstance(x, (int, np.integer)) for x in k):
        v = int(k[0])
        for x in k[1:]:
            v = v * (1 << 32) + int(x)
        return v
    return k if len(k) > 1 else k[0]


class KeyCount:
    """Multiset of keys: sorted base array with counts + a small delta Counter (O(log n) lookup, O(1) update)."""

    def __init__(self, vals):
        if vals.dtype == object:
            self.base = Counter(vals.tolist())
            self.arr = None
        else:
            self.arr, self.cnt = np.unique(vals, return_counts=True)
        self.d = Counter()       # scalar updates (object keys)
        self.dk = self.dc = None  # vectorised updates (integer keys): sorted unique keys and their count deltas
        self.ndup = int((self.cnt > 1).sum()) if self.arr is not None else sum(1 for v in self.base.values() if v > 1)

    def get(self, k):
        if self.arr is None:
            return self.base.get(k, 0) + self.d[k]
        return int(self.get_many(np.array([k], dtype=np.int64))[0])

    def get_many(self, ks):
        if self.arr is None or ks.dtype == object:
            return np.array([self.get(k) for k in ks.tolist()])
        o = np.argsort(ks, kind="stable")  # sorted probes keep the binary searches cache-friendly on large arrays
        ks_s = ks[o]
        i2 = np.minimum(np.searchsorted(self.arr, ks_s), len(self.arr) - 1)
        b = np.zeros(len(ks), dtype=np.int64)
        if len(self.arr):
            b[o] = np.where(self.arr[i2] == ks_s, self.cnt[i2], 0)
        if self.dk is not None and len(self.dk):
            j = np.minimum(np.searchsorted(self.dk, ks), len(self.dk) - 1)
            b = b + np.where(self.dk[j] == ks, self.dc[j], 0)
        if self.d:
            b = b + np.array([self.d.get(int(k), 0) for k in ks])
        return b

    def add_many(self, ks, n, track=True):
        """Add n (+1 / -1) occurrences of every key in ks; track=True keeps the duplicate-key count exact (key atoms),
        reference multisets of inclusion atoms only need the counts."""
        if not len(ks):
            return
        if self.arr is None or ks.dtype == object:
            for k in ks.tolist():
                self.add(k, n)
            return
        u, c = np.unique(ks, return_counts=True)
        c = c.astype(np.int64) * n
        if track:
            before = self.get_many(u)
            after = before + c
            self.ndup += int(((after > 1) & (before <= 1)).sum()) - int(((before > 1) & (after <= 1)).sum())
        if self.dk is None:
            self.dk, self.dc = u, c
        else:  # merge the sorted batch into the sorted delta arrays (no full re-sort)
            j = np.searchsorted(self.dk, u)
            hit = (j < len(self.dk)) & (self.dk[np.minimum(j, len(self.dk) - 1)] == u)
            self.dc[j[hit]] += c[hit]
            self.dk = np.insert(self.dk, j[~hit], u[~hit])
            self.dc = np.insert(self.dc, j[~hit], c[~hit])

    def add(self, k, n=1):
        c = self.get(k)
        self.d[k] += n
        if c <= 1 < c + n:
            self.ndup += 1
        if c + n <= 1 < c:
            self.ndup -= 1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instances as I, atoms as A, policies as Pm

OUT = os.path.join(I.ROOT, "results", "maint")


def keyv(row, cols, idx):
    return tuple(row[idx[c]] for c in cols)


def run(name, n_batch=30, k_period=10, seed=0, app_frac=2e-3, dim_frac=1e-3, p_dup=0.04, p_orph=0.04, p_del=0.03,
        p_scd=0.1, data=None, fixed_fact=None, fixed_dim=None, tag="", keep=False):
    """data: alternative instance file (e.g. SF10 on the external disk) for the pool `name`; fixed_fact / fixed_dim:
    absolute batch sizes (rows per fact / dimension per batch) instead of fractions of the current table size."""
    rnd = random.Random(seed)
    rec = json.load(open(os.path.join(I.ROOT, "results", "pool", name + ".json")))
    P = Pm.Pool(rec)
    W = P.workload()
    # certified set = atoms of gold-safe edges used by answerable queries' plans
    cert_edges = set()
    safe_gold = [e for e in P.gold if P.safe[e]]
    for q in W:
        t = P.plan(q, safe_gold)
        if t:
            cert_edges |= t
    atoms = sorted({a for e in cert_edges for a in P.ea[e]})
    src = data or os.path.join(I.DATA, name + ".duckdb")
    dst = src[:-len(".duckdb")] + f"_maint{tag}_s{seed}.duckdb" if data else os.path.join(I.DATA, name + "_maint.duckdb")
    shutil.copy(src, dst)
    con = duckdb.connect(dst)
    cols = {t: [r[0] for r in con.execute(f'DESCRIBE "{t}"').fetchall()] for (t,) in con.execute("SHOW TABLES").fetchall()}
    ctype = {t: {r[0]: r[1] for r in con.execute(f'DESCRIBE "{t}"').fetchall()} for t in cols}
    rows = {t: con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in cols}
    tabs = sorted({t for a in atoms for t in Pm.atom_tables(a)})
    facts = sorted({e[0] for e in cert_edges} - {e[2] for e in cert_edges})
    dims = sorted({e[2] for e in cert_edges})
    A_ = {a: json.loads(a) for a in atoms}

    # ---------------- delta structures (built once; build cost recorded)
    t0 = time.time()
    build_rows = 0
    ukeys, iref, skeys, orph = {}, {}, {}, {}

    def fetch(t, cs, nn=True):
        w = (" WHERE " + " AND ".join(f'"{c}" IS NOT NULL' for c in cs)) if nn else ""
        d = con.execute(f'SELECT {", ".join(chr(34) + c + chr(34) for c in cs)} FROM "{t}"{w}').fetchnumpy()
        return [d[c] for c in cs]

    for a_, d in A_.items():
        if d[0] == "U":
            S, Y, Tm = d[1], tuple(d[2]), d[3]
            build_rows += rows[S]
            if Tm:
                iv = defaultdict(list)
                for r in con.execute(f'SELECT {", ".join(chr(34)+c+chr(34) for c in Y + tuple(Tm))} FROM "{S}"').fetchall():
                    iv[r[:len(Y)]].append((r[len(Y)], r[len(Y) + 1]))
                ukeys[a_] = iv
            else:
                ukeys[a_] = KeyCount(enc_cols(fetch(S, Y)))
    for a_, d in A_.items():
        if d[0] == "I":
            R, X, S, Y, Tm = d[1], tuple(d[2]), d[3], tuple(d[4]), d[5]
            if (S, Y) not in skeys:
                skeys[(S, Y)] = KeyCount(enc_cols(fetch(S, Y)))
                build_rows += rows[S]
            if Tm is None:
                iref[a_] = KeyCount(enc_cols(fetch(R, X)))
                build_rows += rows[R]
                rc = iref[a_]
                orph[a_] = int(rc.cnt[skeys[(S, Y)].get_many(rc.arr) <= 0].sum()) if rc.arr is not None else \
                    sum(c for k, c in rc.base.items() if skeys[(S, Y)].get(k) <= 0)
    build_secs = time.time() - t0

    belief = {m: {a: True for a in atoms} for m in ["full", "periodic", "touched", "trust", "delta"]}
    cost = defaultdict(list)
    secs = defaultdict(list)
    expo = defaultdict(int)
    log = []

    def truth_now():
        out, c, s = {}, 0, 0.0
        for a, d in A_.items():
            t1 = time.time()
            if d[0] == "U":
                r = A.probe_U(con, d[1], tuple(d[2]), d[3])
                out[a] = r["dup_keys"] == 0
                c += rows[d[1]]
            else:
                r = A.probe_I(con, d[1], tuple(d[2]), d[3], tuple(d[4]), d[5])
                out[a] = r["orphans"] == 0
                c += rows[d[1]] + rows[d[3]]
            s += time.time() - t1
        return out, c, s

    def queries_unsafe(bel, truth):
        n = 0
        for q in W:
            tree = P.plan(q, [e for e in cert_edges if all(bel[a] for a in P.ea[e])])
            if tree and any(not truth[a] for e in tree for a in P.ea[e] if a in truth):
                n += 1
        return n

    for b in range(n_batch):
        changed = set()
        delta = defaultdict(list)       # table -> inserted rows (tuples in column order)
        deleted = defaultdict(list)     # table -> deleted rows
        events = []
        # fact appends (valid foreign keys: resampled existing rows)
        for F in facts:
            k = fixed_fact or max(1, int(rows[F] * app_frac))
            new = con.execute(f'SELECT * FROM "{F}" USING SAMPLE {k} ROWS (reservoir, {seed * 1000 + b})').fetchall()
            for d in A_.values():
                if d[0] == "I" and d[1] == F and d[5] is None and rnd.random() < p_orph:
                    ix = cols[F].index(d[2][0])
                    r = list(new[0]); r[ix] = -(10 ** 8 + b) if I.kind(ctype[F][d[2][0]]) == "int" else f"__orph{b}"
                    new[0] = tuple(r); events.append(("orphan", F, d[2][0]))
                    break
            delta[F] += new
        # dimension inserts with fresh keys, rare duplicates / deletes / SCD overlaps
        scd = {d[1] for d in A_.values() if d[0] == "U" and d[3]}
        for S in dims:
            pk = rec["pk"].get(S)
            if not pk or len(pk) != 1 or S in scd or rnd.random() > 0.3:
                continue
            ix = cols[S].index(pk[0])
            k = fixed_dim or max(1, int(rows[S] * dim_frac))
            base = con.execute(f'SELECT * FROM "{S}" USING SAMPLE {k} ROWS (reservoir, {seed * 1000 + b})').fetchall()
            mx = con.execute(f'SELECT max("{pk[0]}") FROM "{S}"').fetchone()[0]
            new = []
            for j, r in enumerate(base):
                r = list(r)
                if isinstance(mx, int):
                    r[ix] = mx + 1 + j
                new.append(tuple(r))
            if rnd.random() < p_dup:
                new.append(base[0]); events.append(("dup", S))
            delta[S] += new
            if rnd.random() < p_del:
                # delete one key that is referenced by some fact
                refs = [d for d in A_.values() if d[0] == "I" and d[3] == S and d[5] is None and len(d[4]) == 1]
                if refs:
                    d = rnd.choice(refs)
                    v = con.execute(f'SELECT "{d[2][0]}" FROM "{d[1]}" WHERE "{d[2][0]}" IS NOT NULL LIMIT 1 OFFSET {b}').fetchone()
                    if v:
                        dr = con.execute(f'SELECT * FROM "{S}" WHERE "{d[4][0]}" = ?', [v[0]]).fetchall()
                        deleted[S] += dr; events.append(("delete", S))
        for a, d in A_.items():  # SCD: a new version of an existing business key, sometimes overlapping
            if d[0] == "U" and d[3] and rnd.random() < p_scd:
                S, Y, (f, t) = d[1], d[2], d[3]
                r = list(con.execute(f'SELECT * FROM (SELECT * FROM "{S}" WHERE "{t}" IS NULL) USING SAMPLE 1 ROWS (reservoir, {b})').fetchone())
                overlap = rnd.random() < 0.3
                fi, ti = cols[S].index(f), cols[S].index(t)
                r[fi] = r[fi] + datetime.timedelta(days=1 if overlap else 0)
                r[ti] = None
                if not overlap:
                    # close the current version first (an update = delete + insert)
                    old = con.execute(f'SELECT * FROM "{S}" WHERE "{Y[0]}" = ? AND "{t}" IS NULL', [r[cols[S].index(Y[0])]]).fetchall()
                    deleted[S] += old
                    for o in old:
                        o = list(o); o[ti] = r[fi] + datetime.timedelta(days=99)
                        delta[S].append(tuple(o))
                    r[fi] = r[fi] + datetime.timedelta(days=100)
                else:
                    events.append(("scd_overlap", S))
                delta[S].append(tuple(r))
        # apply to the database
        for t, rs in deleted.items():
            if not rs:
                continue
            con.executemany(f'DELETE FROM "{t}" WHERE ' + " AND ".join(f'"{c}" IS NOT DISTINCT FROM ?' for c in cols[t]),
                            [list(r) for r in rs])
            rows[t] -= len(rs); changed.add(t)
        for t, rs in delta.items():
            if not rs:
                continue
            df = pd.DataFrame([list(r) for r in rs], columns=cols[t])
            for c in df.columns:  # DuckDB infers DECIMAL width from the first value; insert as double instead
                nn = df[c].dropna()
                if len(nn) and isinstance(nn.iloc[0], __import__("decimal").Decimal):
                    df[c] = df[c].astype(float)
            con.register("df_ins", df)
            con.execute(f'INSERT INTO "{t}" SELECT * FROM df_ins')
            con.unregister("df_ins")
            rows[t] += len(rs); changed.add(t)
        truth, full_c, full_s = truth_now()
        # ---------------- delta method: inspect only the change set (vectorised per table and key list)
        # the change set is handed over in columnar form (as a CDC batch would be); building the frames is not timed,
        # just as loading the base tables is not timed for full re-verification
        frames, memo = {}, {}
        for src_ in (delta, deleted):
            for t, rs in src_.items():
                if rs:
                    frames[(id(src_), t)] = pd.DataFrame(rs, columns=cols[t])
        t1 = time.time()
        dc = 0
        for t in changed:
            dc += len(delta.get(t, [])) + len(deleted.get(t, []))
        idx = {t: {c: i for i, c in enumerate(cols[t])} for t in cols}

        def dkeys(src, t, cs):
            """Encoded non-null keys on columns cs of the inserted (src=delta) or deleted rows of table t."""
            mk = (id(src), t, cs)
            if mk not in memo:
                rs = src.get(t, [])
                if not rs:
                    memo[mk] = np.zeros(0, dtype=np.int64)
                else:
                    fk = (id(src), t)
                    if fk not in frames:
                        frames[fk] = pd.DataFrame(rs, columns=cols[t])
                    sub = frames[fk][list(cs)].dropna()
                    arrs = []
                    for c in cs:
                        v = sub[c].to_numpy()
                        if v.dtype == object:
                            v = np.array(v.tolist())
                        if v.dtype.kind == "f" and len(v) and np.all(np.mod(v, 1) == 0):
                            v = v.astype(np.int64)
                        arrs.append(v)
                    memo[mk] = enc_cols(arrs) if len(sub) else np.zeros(0, dtype=np.int64)
            return memo[mk]

        for a, d in A_.items():
            if d[0] == "U":
                S, Y, Tm = d[1], tuple(d[2]), d[3]
                if S not in changed:
                    continue
                if Tm:
                    iv = ukeys[a]
                    fi, ti = idx[S][Tm[0]], idx[S][Tm[1]]
                    for r in deleted.get(S, []):
                        k = keyv(r, Y, idx[S])
                        if (r[fi], r[ti]) in iv[k]:
                            iv[k].remove((r[fi], r[ti]))
                    ok = belief["delta"][a]
                    for r in delta.get(S, []):
                        k = keyv(r, Y, idx[S])
                        lo, hi = r[fi], r[ti]
                        for (f2, t2) in iv[k]:
                            if lo is not None and f2 is not None and lo <= (t2 or MAXD) and f2 <= (hi or MAXD):
                                ok = False
                        iv[k].append((lo, hi))
                    belief["delta"][a] = ok
                else:
                    cnt = ukeys[a]
                    cnt.add_many(dkeys(deleted, S, Y), -1)
                    cnt.add_many(dkeys(delta, S, Y), 1)
                    belief["delta"][a] = cnt.ndup == 0
        strans = {}  # (S, Y) -> keys whose count in S changed this batch, with counts before and after
        for (S, Y), cnt in skeys.items():
            if S in changed:
                dS, iS = dkeys(deleted, S, Y), dkeys(delta, S, Y)
                u = np.unique(np.concatenate([dS, iS])) if len(dS) + len(iS) else np.zeros(0, dtype=np.int64)
                before = cnt.get_many(u) if len(u) else u
                cnt.add_many(dS, -1)
                cnt.add_many(iS, 1)
                strans[(S, Y)] = (u, before, cnt.get_many(u) if len(u) else u)
        for a, d in A_.items():
            if d[0] != "I":
                continue
            R, X, S, Y, Tm = d[1], tuple(d[2]), d[3], tuple(d[4]), d[5]
            if R not in changed and S not in changed:
                continue
            sk = skeys[(S, Y)]
            if Tm:  # temporal inclusion: the business key must exist with a covering version (checked on new facts)
                iv = ukeys[Pm.atoms_of((R, X, S, Y, Tm))[0]]
                di = idx[R][Tm[0]]
                for r in delta.get(R, []):
                    k = keyv(r, X, idx[R])
                    if None in k:
                        continue
                    dt = r[di]
                    # as in the exact SQL probe: a NULL sale date or a NULL valid-from never matches
                    if dt is None or not any(f2 is not None and f2 <= dt <= (t2 or MAXD) for f2, t2 in iv.get(k, [])):
                        belief["delta"][a] = False
                continue
            # exact orphan counter: referencing rows whose key has no partner in S (Theorem 5)
            ref = iref[a]
            if (S, Y) in strans:
                u, b0, b1 = strans[(S, Y)]
                if len(u):
                    r = ref.get_many(u)
                    orph[a] += int(r[(b0 > 0) & (b1 <= 0)].sum()) - int(r[(b0 <= 0) & (b1 > 0)].sum())
            ks, kd = dkeys(delta, R, X), dkeys(deleted, R, X)
            if len(ks):
                orph[a] += int((sk.get_many(ks) <= 0).sum())
                ref.add_many(ks, 1, track=False)
            if len(kd):
                orph[a] -= int((sk.get_many(kd) <= 0).sum())
                ref.add_many(kd, -1, track=False)
            belief["delta"][a] = orph[a] == 0
        secs["delta"].append(time.time() - t1)
        cost["delta"].append(dc)
        # ---------------- other methods
        belief["full"] = dict(truth); cost["full"].append(full_c); secs["full"].append(full_s)
        if (b + 1) % k_period == 0:
            belief["periodic"] = dict(truth); cost["periodic"].append(full_c)
        else:
            cost["periodic"].append(0)
        tc = 0
        for a, d in A_.items():
            if Pm.atom_tables(a) & changed:
                belief["touched"][a] = truth[a]
                tc += rows[d[1]] + (rows[d[3]] if d[0] == "I" else 0)
        cost["touched"].append(tc)
        cost["trust"].append(0)
        for m in belief:
            expo[m] += queries_unsafe(belief[m], truth)
        mism = sum(belief["delta"][a] != truth[a] for a in atoms)
        log.append(dict(batch=b, events=events, n_false=sum(not v for v in truth.values()), delta_mismatch=mism,
                        full_rows=full_c, delta_rows=dc, full_secs=round(full_s, 3), delta_secs=round(secs["delta"][-1], 4)))
        print(name, b, events, "false atoms", log[-1]["n_false"], "delta mismatch", mism, flush=True)
    con.close()
    if not keep:
        os.remove(dst)
    res = dict(inst=name, data=src, tag=tag, rows=rows, fixed_fact=fixed_fact, fixed_dim=fixed_dim, app_frac=app_frac,
               n_atoms=len(atoms), n_queries=len(W), batches=n_batch, k_period=k_period,
               build_rows=build_rows, build_secs=round(build_secs, 2),
               cost={m: sum(v) for m, v in cost.items()}, secs={m: round(sum(v), 3) for m, v in secs.items()},
               exposure=dict(expo), log=log)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"{name}{tag}_s{seed}.json"), "w"), default=str)
    return res


if __name__ == "__main__":
    for nm in sys.argv[1:] or ["tpch", "tpcds"]:
        for s in [0, 1, 2]:
            r = run(nm, seed=s)
            print(json.dumps({k: r[k] for k in ["inst", "n_atoms", "cost", "secs", "exposure", "build_rows"]}))
