"""Auto-BI re-implementation (Lin, He, Chaudhuri, PVLDB 16(10) 2023, arXiv 2306.12515).

Local join classifier (Sec. 4.2, App. B, N:1 classifier). Features per candidate column pair (C_i = FK side R.X,
C_j = 1-side S.Y; composite pairs averaged over their column pairs, data features on the value tuples):
  metadata  Jaccard similarity / Jaccard containment / edit similarity / Jaro-Winkler of tokenised names, each as
            max(sim(C_i, C_j), sim(T_j + C_j, C_i)); name embedding similarity (SentenceBERT in the paper; here a
            character-3-gram cosine, no model download); token and character counts; column-name frequency in the
            training corpus; column position, relative position and unique-column position (both sides);
  data      left / right / max containment of distinct values; distinct-value ratio (both sides); numeric range
            overlap; EMD (rank-based, after Zhang et al. 2010); average value length (both sides); value type one-hot;
            row counts; row / column / cell ratios.
  Model: gradient-boosted trees (sklearn) with isotonic calibration (CalibratedClassifierCV, 5 folds), trained
  leave-one-family-out (families TPC-H, TPC-DS, BIRD; dirty copies belong to their base family) on the gold labels of
  the pool edges. The paper trains on >100K harvested real BI models, which are not available.
Candidates (Alg. 1): pool column pairs whose left containment >= TAU_IND (approximate IND). Auto-BI emits equi-joins
  between column lists, so a temporal (SCD) pool edge and its plain twin are the same Auto-BI candidate; the plain
  equi-join edge is what Auto-BI outputs.
Global step (Sec. 4.3): w(e) = -log P(e); k-MCA via an artificial root with penalty p = -log 0.5 and Chu-Liu/Edmonds
  (Alg. 2); k-MCA-CC (FK-once) by the branch-and-bound of Alg. 3 (precision mode, "Auto-BI-P"); recall mode EMS
  (Sec. 4.3.3): greedily add edges with P >= tau = 0.5 in descending P keeping FK-once and acyclicity.
Variants written: Auto-BI(k-MCA, reimpl.) = k-MCA-CC (precision mode); Auto-BI(k-MCA+EMS, reimpl.) = full Auto-BI;
  Auto-BI-S(k-MCA, reimpl.) = metadata-only features (paper's Auto-BI-S), no data read;
  Auto-BI-local(p>=0.5) = local classifier without the global step (ablation);
  "...LODO" = BIRD databases trained leave-one-database-out (other 10 BIRD DBs + TPC), a friendlier protocol.
Writes results/baseline_autobi.csv and the "autobi" section of results/baselines2_detail.json."""
import csv, math, os, re, sys, time
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import networkx as nx
import baseline_profilers as BP
from instances import kind

TAU_IND = 0.5
P_ROOT = -math.log(0.5)
TAU_EMS = 0.5
BB_LIMIT = 200000


def fam(n):
    if n == "job":
        return "job"
    if n.startswith("bird_"):
        return "bird"
    return "tpcds" if n.startswith("tpcds") else "tpch"


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


def jaro_winkler(s, t):
    if s == t:
        return 1.0
    ls, lt = len(s), len(t)
    if not ls or not lt:
        return 0.0
    md = max(ls, lt) // 2 - 1
    sm, tm = [False] * ls, [False] * lt
    m = 0
    for i in range(ls):
        for j in range(max(0, i - md), min(lt, i + md + 1)):
            if not tm[j] and s[i] == t[j]:
                sm[i] = tm[j] = True
                m += 1
                break
    if not m:
        return 0.0
    k = tr = 0
    for i in range(ls):
        if sm[i]:
            while not tm[k]:
                k += 1
            tr += s[i] != t[k]
            k += 1
    j = (m / ls + m / lt + (m - tr / 2) / m) / 3
    p = 0
    for a, b in zip(s[:4], t[:4]):
        if a != b:
            break
        p += 1
    return j + p * 0.1 * (1 - j)


def tri_cos(a, b):
    A = Counter(a[i:i + 3] for i in range(max(1, len(a) - 2)))
    B = Counter(b[i:i + 3] for i in range(max(1, len(b) - 2)))
    num = sum(A[k] * B[k] for k in A)
    den = math.sqrt(sum(v * v for v in A.values()) * sum(v * v for v in B.values()))
    return num / den if den else 0.0


def name_feats(x, y, S):
    a, b, bt = toks(x), toks(y), toks(S) + toks(y)
    out = []
    for f in (lambda u, v: len(set(u) & set(v)) / max(1, len(set(u) | set(v))),
              lambda u, v: len(set(u) & set(v)) / max(1, len(set(u))),
              lambda u, v: 1 - lev("_".join(u), "_".join(v)) / max(1, len("_".join(u)), len("_".join(v))),
              lambda u, v: jaro_winkler("_".join(u), "_".join(v)),
              lambda u, v: tri_cos("_".join(u), "_".join(v))):
        out.append(max(f(a, b), f(a, bt)))
    return out


def as_num(v):
    try:
        return np.asarray(v, dtype=float)
    except (ValueError, TypeError):
        return None


META = ["jac", "jac_cont", "edit", "jw", "emb3g", "tok_x", "tok_y", "chr_x", "chr_y", "freq_x", "freq_y", "pos_x", "pos_y",
        "relpos_x", "relpos_y", "upos_x", "upos_y", "t_int", "t_str", "t_num", "rows_R", "rows_S", "row_ratio",
        "col_ratio", "cell_ratio"]
DATA = ["cont_l", "cont_r", "cont_max", "dr_x", "dr_y", "range_ov", "emd", "len_x", "len_y"]
META_S = [f for f in META if f not in ("upos_x", "upos_y")]  # Auto-BI-S: no data read


class Inst:
    """Features of one instance (column-pair candidates)."""

    def __init__(self, name):
        t0 = time.time()
        self.name = name
        self.P = P = BP.load_pool(name)
        con = BP.connect(name)
        tabs = P.rec["tables"]
        # Auto-BI candidates are column pairs: dedupe temporal edges with their plain twins
        pairs = {}
        for e in P.edges:
            k = e[:4]
            if k not in pairs or (pairs[k][4] is not None and e[4] is None):
                pairs[k] = e
        self.pairs = pairs
        self.label = {k: any(g[:4] == k for g in P.gold) for k in pairs}
        # per-table uniqueness of every column (one pass per table; HLL distinct vs non-null count)
        self.uniq = {}
        for t in P.touched:
            cs = list(tabs[t]["cols"])
            sel = ", ".join(f"approx_count_distinct({BP.q(c)}), count({BP.q(c)})" for c in cs)
            r = con.execute(f"SELECT {sel} FROM {BP.q(t)}").fetchone()
            self.uniq[t] = [r[2 * i + 1] > 0 and r[2 * i] >= 0.99 * r[2 * i + 1] for i in range(len(cs))]
        self.vals, self.nn = {}, {}
        for (R, X, S, Y) in pairs:
            for t, cs in ((R, X), (S, Y)):
                if (t, cs) not in self.vals:
                    if name == "job" and len(cs) == 1 and BP.is_int_col(P, t, cs[0]):
                        self.vals[(t, cs)] = BP.extract_np(con, t, cs[0])  # memory-light path, same features
                    else:
                        self.vals[(t, cs)] = BP.extract(con, t, cs)
                    nn = " AND ".join(f"{BP.q(c)} IS NOT NULL" for c in cs)
                    self.nn[(t, cs)] = con.execute(f"SELECT count(*) FROM {BP.q(t)} WHERE {nn}").fetchone()[0]
        con.close()
        self.secs_read = time.time() - t0
        self.rows_read = P.full_scan  # every touched table is read (one fused pass)
        t0 = time.time()
        self.data = {k: self.data_feats(k) for k in pairs}
        self.secs_feat = time.time() - t0
        del self.vals  # raw values are not needed after featurisation

    def data_feats(self, k):
        R, X, S, Y = k
        xv, yv = self.vals[(R, X)], self.vals[(S, Y)]
        if isinstance(xv, np.ndarray) and isinstance(yv, np.ndarray):
            return self.data_feats_np(R, X, S, Y, xv, yv)
        if isinstance(xv, np.ndarray) or isinstance(yv, np.ndarray):
            xv = [str(v) for v in xv] if isinstance(xv, np.ndarray) else xv
            yv = [str(v) for v in yv] if isinstance(yv, np.ndarray) else yv
        sx, sy = set(xv), set(yv)
        inter = len(sx & sy)
        cl, cr = inter / max(1, len(sx)), inter / max(1, len(sy))
        drx, dry = len(sx) / max(1, self.nn[(R, X)]), len(sy) / max(1, self.nn[(S, Y)])
        nx_, ny_ = (as_num(xv), as_num(yv)) if len(X) == 1 else (None, None)
        if nx_ is not None and ny_ is not None and len(nx_) and len(ny_):
            lo, hi = max(nx_.min(), ny_.min()), min(nx_.max(), ny_.max())
            span = max(nx_.max(), ny_.max()) - min(nx_.min(), ny_.min())
            ov = max(0.0, hi - lo) / span if span > 0 else 1.0
            u = np.unique(np.concatenate([nx_, ny_]))
            xr, yr = np.searchsorted(u, nx_) / len(u), np.searchsorted(u, ny_) / len(u)
        else:
            ov = -1.0
            u = sorted(sx | sy)
            pos = {v: i for i, v in enumerate(u)}
            xr = np.array([pos[v] for v in xv]) / max(1, len(u))
            yr = np.array([pos[v] for v in yv]) / max(1, len(u))
        if len(xr) and len(yr):
            from scipy.stats import wasserstein_distance
            emd = wasserstein_distance(xr, yr)
        else:
            emd = 1.0
        lx = float(np.mean([len(v) for v in xv])) if xv else 0.0
        ly = float(np.mean([len(v) for v in yv])) if yv else 0.0
        return dict(cont_l=cl, cont_r=cr, cont_max=max(cl, cr), dr_x=drx, dr_y=dry, range_ov=ov, emd=emd, len_x=lx, len_y=ly)

    def data_feats_np(self, R, X, S, Y, xv, yv):
        """Same features as data_feats for sorted distinct int64 arrays (JOB/IMDB)."""
        inter = len(np.intersect1d(xv, yv, assume_unique=True))
        cl, cr = inter / max(1, len(xv)), inter / max(1, len(yv))
        drx, dry = len(xv) / max(1, self.nn[(R, X)]), len(yv) / max(1, self.nn[(S, Y)])
        if len(xv) and len(yv):
            lo, hi = max(xv.min(), yv.min()), min(xv.max(), yv.max())
            span = float(max(xv.max(), yv.max()) - min(xv.min(), yv.min()))
            ov = max(0.0, float(hi - lo)) / span if span > 0 else 1.0
            u = np.union1d(xv, yv)
            from scipy.stats import wasserstein_distance
            emd = wasserstein_distance(np.searchsorted(u, xv) / len(u), np.searchsorted(u, yv) / len(u))
        else:
            ov, emd = -1.0, 1.0

        def mlen(a):
            if not len(a):
                return 0.0
            ab = np.abs(a.astype(np.float64))
            d = np.where(ab < 1, 1, np.floor(np.log10(np.maximum(ab, 1))) + 1) + (a < 0)
            return float(d.mean())
        return dict(cont_l=cl, cont_r=cr, cont_max=max(cl, cr), dr_x=drx, dr_y=dry, range_ov=ov, emd=emd,
                    len_x=mlen(xv), len_y=mlen(yv))

    def meta_feats(self, k, colfreq):
        R, X, S, Y = k
        tabs = self.P.rec["tables"]
        cr, cs = list(tabs[R]["cols"]), list(tabs[S]["cols"])
        nf = np.mean([name_feats(x, y, S) for x, y in zip(X, Y)], axis=0)
        f = dict(zip(["jac", "jac_cont", "edit", "jw", "emb3g"], nf))
        f["tok_x"] = np.mean([len(toks(x)) for x in X]); f["tok_y"] = np.mean([len(toks(y)) for y in Y])
        f["chr_x"] = np.mean([len(x) for x in X]); f["chr_y"] = np.mean([len(y) for y in Y])
        f["freq_x"] = np.mean([colfreq.get("_".join(toks(x)), 0) for x in X])
        f["freq_y"] = np.mean([colfreq.get("_".join(toks(y)), 0) for y in Y])
        f["pos_x"] = np.mean([cr.index(x) for x in X]); f["pos_y"] = np.mean([cs.index(y) for y in Y])
        f["relpos_x"] = f["pos_x"] / len(cr); f["relpos_y"] = f["pos_y"] / len(cs)
        f["upos_x"] = np.mean([sum(self.uniq[R][:cr.index(x)]) for x in X])
        f["upos_y"] = np.mean([sum(self.uniq[S][:cs.index(y)]) for y in Y])
        kd = kind(tabs[R]["cols"][X[0]])
        f["t_int"], f["t_str"], f["t_num"] = float(kd == "int"), float(kd == "str"), float(kd == "num")
        nr, ns = self.P.rows[R], self.P.rows[S]
        f["rows_R"], f["rows_S"] = nr, ns
        f["row_ratio"] = nr / max(1, ns)
        f["col_ratio"] = len(cr) / max(1, len(cs))
        f["cell_ratio"] = nr * len(cr) / max(1, ns * len(cs))
        return f

    def matrix(self, colfreq, feats, only_ind=True):
        keys, Xm, y = [], [], []
        for k in self.pairs:
            d = self.data[k]
            if only_ind and d["cont_l"] < TAU_IND:
                continue
            f = dict(self.meta_feats(k, colfreq), **d)
            keys.append(k)
            Xm.append([f[n] for n in feats])
            y.append(int(self.label[k]))
        return keys, np.array(Xm, dtype=float).reshape(len(keys), len(feats)), np.array(y)


def colfreq_of(insts):
    c = Counter()
    for I in insts:
        for t, v in I.P.rec["tables"].items():
            for col in v["cols"]:
                c["_".join(toks(col))] += 1
    return c


def train(train_insts, feats):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    cf = colfreq_of(train_insts)
    Xs, ys = [], []
    for I in train_insts:
        _, Xm, y = I.matrix(cf, feats)
        Xs.append(Xm); ys.append(y)
    X, y = np.vstack(Xs), np.concatenate(ys)
    clf = CalibratedClassifierCV(GradientBoostingClassifier(random_state=0), method="isotonic", cv=5)
    clf.fit(X, y)
    return clf, cf, dict(n_train=len(y), n_pos=int(y.sum()))


# ------------------------------------------------------------------ global step
def kmca(cands, P):
    """cands: list of (edge, w, src). Alg. 2: artificial root r, edges r->v of weight p, Chu-Liu/Edmonds."""
    if not cands:
        return [], 0.0
    G = nx.DiGraph()
    nodes = {e[0] for e, _, _ in cands} | {e[2] for e, _, _ in cands}
    for v in nodes:
        G.add_edge("__root__", v, weight=P_ROOT, e=None)
    for e, w, s in cands:
        u, v = e[0], e[2]
        if G.has_edge(u, v) and G[u][v]["weight"] <= w:
            continue
        G.add_edge(u, v, weight=w, e=e)
    A = nx.minimum_spanning_arborescence(G, attr="weight", preserve_attrs=True)
    J = [d["e"] for u, v, d in A.edges(data=True) if d["e"] is not None]
    k = sum(1 for u, v in A.edges() if u == "__root__")
    cost = sum(d["weight"] for u, v, d in A.edges(data=True) if d["e"] is not None) + (k - 1) * P_ROOT
    return J, cost


def kmca_cc(cands):
    """Alg. 3: branch and bound on FK-once conflicts (same source column list, different targets)."""
    W = {e: w for e, w, s in cands}
    best = [None, float("inf")]
    calls = [0]

    def rec(allowed):
        calls[0] += 1
        if calls[0] > BB_LIMIT:
            return
        J, c = kmca([(e, W[e], e[:2]) for e in allowed], None)
        if c >= best[1] - 1e-12:
            return
        by_src = defaultdict(list)
        for e in J:
            by_src[e[:2]].append(e)
        conf = [g for g in by_src.values() if len(g) > 1]
        if not conf:
            best[0], best[1] = J, c
            return
        Cs = sorted(conf[0], key=lambda e: W[e])
        for keep in Cs:
            rec([e for e in allowed if e not in Cs or e == keep])

    rec([e for e, w, s in cands])
    return best[0] or [], best[1], calls[0]


def ems(J, cands, prob):
    """Recall mode: add edges with P >= tau greedily (descending P), keeping FK-once and acyclicity."""
    G = nx.DiGraph()
    G.add_edges_from((e[0], e[2]) for e in J)
    used = {e[:2]: e[2:4] for e in J}
    S = []
    for e in sorted((e for e, w, s in cands if e not in J and prob[e] >= TAU_EMS), key=lambda e: -prob[e]):
        src = e[:2]
        if src in used and used[src] != e[2:4]:
            continue
        if e[0] == e[2] or (e[2] in G and e[0] in G and nx.has_path(G, e[2], e[0])):
            continue
        S.append(e)
        used[src] = e[2:4]
        G.add_edge(e[0], e[2])
    return S


def predict(I, clf, cf, feats):
    t0 = time.time()
    keys, Xm, _ = I.matrix(cf, feats)
    pr = clf.predict_proba(Xm)[:, 1] if len(keys) else np.array([])
    prob = {I.pairs[k]: float(p) for k, p in zip(keys, pr)}
    cands = [(e, -math.log(min(1 - 1e-9, max(1e-9, p))), e[:2]) for e, p in prob.items() if p > 0.5]
    t1 = time.time()
    J, cost, calls = kmca_cc(cands)
    t2 = time.time()
    S = ems(J, [(e, -math.log(max(1e-9, p)), e[:2]) for e, p in prob.items()], prob)
    t3 = time.time()
    return dict(prob=prob, J=J, S=S, cost=cost, calls=calls, secs_local=t1 - t0, secs_kmca=t2 - t1, secs_ems=t3 - t2,
                n_cand=len(keys))


def run(names):
    allnames = BP.instances()
    t0 = time.time()
    insts = {}
    extra = [n for n in names if n not in allnames]
    cache = os.environ.get("AUTOBI_CACHE")  # optional pickle of computed features (keeps measured secs)
    for n in allnames + extra:
        cp = os.path.join(cache, n + ".pkl") if cache else None
        if cp and os.path.exists(cp):
            import pickle
            insts[n] = pickle.load(open(cp, "rb"))
        else:
            insts[n] = Inst(n)
            if cp:
                import pickle
                os.makedirs(cache, exist_ok=True)
                pickle.dump(insts[n], open(cp, "wb"))
        print("features", n, round(insts[n].secs_read + insts[n].secs_feat, 1), "s", flush=True)
    models = {}

    def model(key, train_names, feats):
        if key not in models:
            models[key] = train([insts[n] for n in train_names], feats)
        return models[key]

    feats_full, feats_s = META + DATA, META_S
    rows = []
    for name in names:
        I = insts[name]
        P = I.P
        W = P.workload()
        lofo = [n for n in allnames if fam(n) != fam(name)]
        configs = [("LOFO", lofo, feats_full, "full"), ("LOFO", lofo, feats_s, "S")]
        if fam(name) == "bird":
            configs.append(("LODO", [n for n in allnames if n != name], feats_full, "full"))
        det = {}
        for proto, tr, feats, kindf in configs:
            clf, cf, tinfo = model((proto, fam(name) if proto == "LOFO" else name, kindf), tr, feats)
            out = predict(I, clf, cf, feats)
            rr = I.rows_read if kindf == "full" else 0
            feat_secs = I.secs_read + I.secs_feat if kindf == "full" else 0.0
            secs = feat_secs + out["secs_local"] + out["secs_kmca"]
            ex = dict(cost_model="scan", protocol=proto, rows_read=rr, secs=round(secs, 3),
                      secs_features=round(feat_secs, 3), secs_kmca=round(out["secs_kmca"], 4), bb_calls=out["calls"],
                      n_cand=out["n_cand"], n_train=tinfo["n_train"], n_train_pos=tinfo["n_pos"])
            sfx = "" if proto == "LOFO" else ", LODO"
            if kindf == "full":
                variants = [(f"Auto-BI(k-MCA, reimpl.{sfx})", out["J"]),
                            (f"Auto-BI(k-MCA+EMS, reimpl.{sfx})", out["J"] + out["S"]),
                            (f"Auto-BI-local(p>=0.5{sfx})", [e for e, p in out["prob"].items() if p >= 0.5])]
            else:
                variants = [("Auto-BI-S(k-MCA, reimpl.)", out["J"])]
            for lab, acc in variants:
                r = P.evaluate(W, acc, lab, rr, dict(ex))
                r.update(n_accepted=len(acc), n_acc_gold=sum(1 for e in acc if e in P.gold),
                         n_acc_unsafe=sum(1 for e in acc if not P.safe[e]), n_gold=len(P.gold))
                rows.append(r)
                det[lab] = dict(accepted=[P.ename[e] for e in acc], evaluate=r)
            if kindf == "full" and proto == "LOFO":
                det["prob"] = {P.ename[e]: round(p, 4) for e, p in out["prob"].items()}
        BP.update_detail("autobi", name, det)
        for r in rows[-(len(configs) * 3):]:
            if r["inst"] == name:
                print(name, r["method"], "cov", round(r["coverage"], 3), "fcr_card", round(r["fcr_card"], 3),
                      "fcr_sem", round(r["fcr_sem"], 3), "acc", r["n_accepted"], "gold", r["n_acc_gold"], "/", r["n_gold"],
                      "unsafe", r["n_acc_unsafe"], flush=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(BP.RES, "baseline_autobi" + os.environ.get("OUT_SUFFIX", "") + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("total secs", round(time.time() - t0, 1))


if __name__ == "__main__":
    run(sys.argv[1:] or BP.instances())
