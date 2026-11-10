"""HoPF re-implementation (Jiang & Naumann, "Holistic primary key and foreign key detection", JIIS 54, 2020).

Input, as in the paper: exact minimal UCCs and exact INDs, here from the Desbordante profiling run of
baseline_profilers.profile() (SPIDER + HyUCC + MIND for composite candidates) on the columns of the candidate pool.

  PK score  (Sec. 4.1) mean of: cardinality 1/|X|; value length mean_c 1/max(1, maxlen(c) - 8);
            position 1/2 (1/(left(X)+1) + 1/(between(X)+1)); name suffix |{c: name ends with id/key/nr/no}|/|X|.
  FK score  (Sec. 4.2) mean of: fuzzy token similarity of table+column names (Chaudhuri et al. 2003: tokens of the
            RHS mapped to the most similar unmapped LHS token by Levenshtein similarity, weighted by ln(1/f));
            data distribution = Bhattacharyya coefficient of 20-bucket histograms over the RHS value range
            (n-ary candidates: per-dimension average).
  Pruning   (Sec. 5) RHS must be a minimal UCC; all-NULL columns ignored; PK<=PK filter (LHS is a chosen PK and its
            ordered values form a consecutive subsequence of the RHS values); cliff pruning of PK candidates;
            FK uniqueness (one RHS per LHS); no cyclic reference; stop when the schema graph is connected.
  Holistic  (Sec. 6, Alg. 1-4) enumerate PK combinations (Cartesian product of the per-table Upper sets), greedy FK
            selection per combination, PKReduce, keep the combination with the highest PK+FK score.

Accepted edges = predicted FKs that are candidate-pool edges (FKs outside the pool are counted, not evaluated).
Writes results/baseline_hopf.csv and the "hopf" section of results/baselines2_detail.json."""
import csv, itertools, json, math, os, re, sys, time
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import networkx as nx
import numpy as np
import baseline_profilers as BP

SUFFIX = ("id", "key", "nr", "no")
BUCKETS = 20
LEN_N = 8
MAX_PKCC = 5000


def toks(s):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def lev(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lsim(a, b):
    return 1 - lev(a, b) / max(1, len(a), len(b))


def fuzzy(TL, TR, freq, total):
    """sim(L <= R): each RHS token mapped to the most similar unmapped LHS token, IDF-like weights ln(1/f)."""
    free = list(TL)
    num = den = 0.0
    for t in TR:
        f = freq.get(t, 1) / total
        w = math.log(1 / f) if f < 1 else 1e-6
        s = 0.0
        if free:
            k = max(range(len(free)), key=lambda i: lsim(t, free[i]))
            s = lsim(t, free[k])
            free.pop(k)
        num += s * w
        den += w
    return num / den if den else 0.0


def isnum(vals):
    if isinstance(vals, np.ndarray):
        return vals.dtype.kind in "iuf"
    try:
        [float(v) for v in vals[:2000]]
        return all(_isf(v) for v in vals)
    except ValueError:
        return False


def _isf(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


def sort_vals(vals, num):
    return sorted(vals, key=float) if num else sorted(vals)


def bhatta(xv, yv):
    """Bhattacharyya coefficient between the histograms of X and Y on 20 buckets built over Y's value range."""
    if len(xv) == 0 or len(yv) == 0:
        return 0.0
    if isinstance(xv, np.ndarray) and isinstance(yv, np.ndarray) and isnum(xv) and isnum(yv):
        lo, hi = float(yv.min()), float(yv.max())
        width = (hi - lo) / BUCKETS or 1.0
        bx = np.clip(((xv.astype(float) - lo) / width).astype(np.int64), 0, BUCKETS - 1)
        by = np.clip(((yv.astype(float) - lo) / width).astype(np.int64), 0, BUCKETS - 1)
        hx, hy = np.bincount(bx, minlength=BUCKETS) / len(xv), np.bincount(by, minlength=BUCKETS) / len(yv)
        return float(np.sqrt(hx * hy).sum())
    xv, yv = [str(v) for v in xv], [str(v) for v in yv]
    num = isnum(yv) and isnum(xv)
    hx, hy = [0] * BUCKETS, [0] * BUCKETS
    if num:
        ys = [float(v) for v in yv]
        lo, hi = min(ys), max(ys)
        width = (hi - lo) / BUCKETS or 1.0
        b = lambda v: min(BUCKETS - 1, max(0, int((float(v) - lo) / width)))
        for v in yv: hy[b(v)] += 1
        for v in xv: hx[b(v)] += 1
    else:  # strings: buckets over the rank range of Y's sorted values
        ys = sorted(yv)
        rank = {v: i for i, v in enumerate(ys)}
        n = len(ys)
        import bisect
        for i in range(n): hy[min(BUCKETS - 1, i * BUCKETS // n)] += 1
        for v in xv:
            i = rank.get(v)
            if i is None:
                i = min(n - 1, bisect.bisect_left(ys, v))
            hx[min(BUCKETS - 1, i * BUCKETS // n)] += 1
    sx, sy = sum(hx), sum(hy)
    return sum(math.sqrt(a / sx * b / sy) for a, b in zip(hx, hy))


def consecutive_sub(xv, yv):
    if isinstance(xv, np.ndarray) and isinstance(yv, np.ndarray) and isnum(xv) and isnum(yv):
        xs, ys = np.unique(xv), np.unique(yv)
        if len(xs) == 0 or len(xs) > len(ys):
            return False
        i = int(np.searchsorted(ys, xs[0]))
        return i < len(ys) and ys[i] == xs[0] and i + len(xs) <= len(ys) and bool(np.array_equal(ys[i:i + len(xs)], xs))
    xv, yv = list(xv), list(yv)
    num = isnum(yv) and isnum(xv)
    xs, ys = sort_vals(xv, num), sort_vals(yv, num)
    if not xs or len(xs) > len(ys):
        return False
    key = (lambda v: float(v)) if num else (lambda v: v)
    try:
        i = [key(v) for v in ys].index(key(xs[0]))
    except ValueError:
        return False
    return [key(v) for v in ys[i:i + len(xs)]] == [key(v) for v in xs]


class HoPF:
    def __init__(self, P, prof):
        self.P, self.prof = P, prof
        self.tables = list(P.touched)
        self.vals = prof["vals"]
        allt = []
        for t in self.tables:
            allt += toks(t)
            for c in P.rec["tables"][t]["cols"]:
                allt += toks(c)
        self.freq, self.total = Counter(allt), len(allt)
        # PK candidates: minimal UCCs
        self.pkc = {}
        for t in self.tables:
            us = [tuple(u) for u in prof["uccs"].get(t, []) if u]
            if us:
                self.pkc[t] = sorted(((self.pk_score(t, u), u) for u in us), key=lambda z: -z[0])
        # FK candidates: INDs, prefiltered (RHS is a UCC, LHS not all-NULL)
        ucc_set = {(t, frozenset(u)) for t, us in self.pkc.items() for _, u in us}
        inds = [(R, X, S, Y) for (R, X, S, Y) in list(prof["uinds"]) + list(prof["ninds"]) if R != S]
        self.inds = []
        for (R, X, S, Y) in inds:
            if (S, frozenset(Y)) not in ucc_set:
                continue
            if not all(len(self.vals.get((R, (x,)), [])) for x in X):
                continue
            self.inds.append(((R, X, S, Y), self.fk_score(R, X, S, Y)))
        self.inds.sort(key=lambda z: -z[1])
        self._pkpk = {}

    # ---------------------------------------------------------------- features
    def pk_score(self, t, X):
        cols = list(self.P.rec["tables"][t]["cols"])
        card = 1 / len(X)
        ln = []
        for c in X:
            v = self.vals.get((t, (c,)), [])
            if isinstance(v, np.ndarray) and isnum(v):
                ml = max(len(str(v.min())), len(str(v.max()))) if len(v) else 0
            else:
                ml = max((len(s) for s in v), default=0)
            ln.append(1 / max(1, ml - LEN_N))
        length = sum(ln) / len(ln)
        pos = sorted(cols.index(c) for c in X)
        left = pos[0]
        between = sum(1 for i in range(pos[0], pos[-1] + 1) if cols[i] not in X)
        position = 0.5 * (1 / (left + 1) + 1 / (between + 1))
        suffix = sum(1 for c in X if c.lower().endswith(SUFFIX)) / len(X)
        return (card + length + position + suffix) / 4

    def fk_score(self, R, X, S, Y):
        ns, ds = [], []
        for x, y in zip(X, Y):
            TL = toks(R) + toks(x)
            TR = toks(S) + toks(y)
            ns.append(fuzzy(TL, TR, self.freq, self.total))
            ds.append(bhatta(self.vals.get((R, (x,)), []), self.vals.get((S, (y,)), [])))
        return (sum(ns) / len(ns) + sum(ds) / len(ds)) / 2

    def is_pkpk(self, ind, pkc):
        R, X, S, Y = ind
        if R not in pkc or set(pkc[R]) != set(X):
            return False
        if ind not in self._pkpk:
            xv = self.vals.get((R, X)) if len(X) > 1 else self.vals.get((R, X))
            yv = self.vals.get((S, Y))
            self._pkpk[ind] = bool(xv is not None and yv is not None and len(xv) and len(yv) and len(X) == 1
                                   and consecutive_sub(xv, yv))
        return self._pkpk[ind]

    # ---------------------------------------------------------------- Algorithms 1-4
    def pk_combinations(self):
        groups = []
        for t, cands in sorted(self.pkc.items()):
            if len(cands) == 1:
                groups.append([(t, cands[0])])
                continue
            sc = [s for s, _ in cands]
            sd = [sc[i] - sc[i + 1] for i in range(len(sc) - 1)]
            i = max(range(len(sd)), key=lambda k: sd[k])  # cliff
            groups.append([(t, c) for c in cands[:i + 1]])
        n = 1
        for g in groups:
            n *= len(g)
        self.n_pkcc = n
        if n > MAX_PKCC:  # safety cap (not reached on our instances is reported in the detail file)
            groups = [g[:1] if len(g) > 1 else g for g in groups]
        for combo in itertools.product(*groups):
            yield {t: u for t, (s, u) in combo}, sum(s for t, (s, u) in combo)

    def get_fks(self, pkc):
        cands = [(i, s) for (i, s) in self.inds if i[2] in pkc and set(pkc[i[2]]) == set(i[3])]
        G = nx.DiGraph()
        G.add_nodes_from(self.tables)
        U = nx.Graph()
        U.add_nodes_from(self.tables)
        fks, D, used_lhs = [], [], set()
        for ind, s in cands:
            R, X, S, Y = ind
            if (R, X) in used_lhs:
                continue
            if nx.has_path(G, S, R):  # circular reference
                continue
            if self.is_pkpk(ind, pkc):
                D.append((ind, s))
                continue
            fks.append((ind, s))
            used_lhs.add((R, X))
            G.add_edge(R, S)
            U.add_edge(R, S)
            if nx.is_connected(U):
                break
        return fks, D

    def pk_reduce(self, pkc, pks, fks):
        score = pks + sum(s for _, s in fks)
        for t in list(pkc):
            pkc2 = {k: v for k, v in pkc.items() if k != t}
            pks2 = pks - self.pk_score(t, pkc[t])
            fks2, _ = self.get_fks(pkc2)
            sc2 = pks2 + sum(s for _, s in fks2)
            if sc2 > score:
                pkc, pks, fks, score = pkc2, pks2, fks2, sc2
        return pkc, pks, fks, score

    def run(self):
        best = (None, None, -1)
        n_eval = 0
        for pkc, pks in self.pk_combinations():
            cands = [i for i, _ in self.inds if i[2] in pkc and set(pkc[i[2]]) == set(i[3])]
            if len(cands) < len(pkc) - 1:
                continue
            n_eval += 1
            fks, D = self.get_fks(pkc)
            pkc2, pks2, fks2, score = self.pk_reduce(pkc, pks, fks)
            if score > best[2]:
                best = (pkc2, fks2, score)
        self.n_eval = n_eval
        return best


def run(names):
    out = os.path.join(BP.RES, "baseline_hopf" + os.environ.get("OUT_SUFFIX", "") + ".csv")
    rows = []
    for name in names:
        P = BP.load_pool(name)
        W = P.workload()
        t0 = time.time()
        cdir = os.environ.get("PROFILE_CACHE")
        prof = BP.profile(name, P, cache=os.path.join(cdir, name + ".pkl") if cdir else None)
        t1 = time.time()
        if prof.get("cached"):  # reuse the recorded Desbordante run (same profiling the paper's HoPF needs)
            t0 = t1 - sum(prof["secs"].values())
        H = HoPF(P, prof)
        pkc, fks, score = H.run()
        pkc, fks = pkc or {}, fks or []  # no PK combination admitted (e.g. fewer FK candidates than |PKc|-1)
        t2 = time.time()
        pool = {(e[0], e[1], e[2], e[3]): e for e in P.edges if e[4] is None}
        acc = [pool[i] for i, _ in fks if i in pool]
        outside = [i for i, _ in fks if i not in pool]
        rr = P.full_scan  # one pass over every touched table (profiling); HoPF itself reads no further rows
        ev = P.evaluate(W, acc, "HoPF(reimpl.)", rr, dict(cost_model="scan"))
        gold_pk = P.rec["pk"]
        pk_ok = sum(1 for t, u in (pkc or {}).items() if gold_pk.get(t) and set(gold_pk[t]) == set(u))
        ev.update(rows_read=rr, secs_profile=round(t1 - t0, 3), secs_hopf=round(t2 - t1, 3),
                  n_fk=len(fks), n_fk_in_pool=len(acc), n_fk_outside_pool=len(outside),
                  n_fk_gold=sum(1 for e in acc if e in P.gold), n_gold=len(P.gold), n_pk=len(pkc or {}), n_pk_correct=pk_ok,
                  n_pkcc=H.n_pkcc, n_pkcc_eval=H.n_eval, n_inds=len(H.inds))
        rows.append(ev)
        BP.update_detail("hopf", name, dict(
            pks={t: list(u) for t, u in (pkc or {}).items()}, fks=[["%s.%s->%s.%s" % (i[0], "+".join(i[1]), i[2], "+".join(i[3])), round(s, 4)] for i, s in fks],
            outside_pool=[list(map(str, i)) for i in outside], evaluate=ev))
        print(name, {k: ev[k] for k in ["coverage", "fcr_card", "fcr_sem", "n_fk", "n_fk_in_pool", "n_fk_gold", "n_gold",
                                         "n_pk_correct", "n_pk", "n_pkcc", "secs_hopf"]}, flush=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    run(sys.argv[1:] or BP.instances())
