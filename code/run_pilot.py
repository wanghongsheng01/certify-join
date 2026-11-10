"""Main pilot: certified coverage vs probe budget, false-certification rates, under two cost models.

Writes results/main.csv (one row per instance x method x budget x cost model) and results/zero.csv
(non-probing baselines)."""
import csv, json, math, os, sys, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policies as Pm
from instances import name_sim, tokens

ROOT = Pm.ROOT
BUDGETS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
SAMPLE = (1e-3, 0.05)


def load(name, prior="ds-flash"):
    rec = json.load(open(os.path.join(ROOT, "results", "pool", name + ".json")))
    st_path = os.path.join(ROOT, "_private" if name.startswith("ent") else "results", "stats", name + ".json")
    st = json.load(open(st_path)) if os.path.exists(st_path) else None
    return Pm.Pool(rec, prior_model=prior), st


# ---------------------------------------------------------------------- zero-probe baselines
def ddl_declared(P):
    return [e for e in P.edges if e in P.declared]


def llm_judge(P, model, thr=0.5):
    pri = P.rec["prior"].get(model, {})
    return [e for e in P.edges if (pri.get(P.ename[e]) or 0) >= thr]


def llm_vote(P, thr=0.5):
    ms = [m for m in P.rec["prior"] if P.rec["prior"][m]]
    out = []
    for e in P.edges:
        v = [(P.rec["prior"][m].get(P.ename[e]) or 0) >= thr for m in ms]
        if sum(v) * 2 > len(v):
            out.append(e)
    return out


def _st(st, t, cols):
    return st.get(json.dumps([t, list(cols)]))


def card_est(P, st, tol=0.1):
    """Optimizer-statistics rule: Y looks unique if HLL ndv >= (1-tol) * non-null rows; X looks included if
    ndv(X) <= (1+tol) ndv(Y) and range(X) within range(Y). Temporal edges use the same rule on the business key."""
    out = []
    for e in P.edges:
        R, X, S, Y, Tm = e
        sy, sx = _st(st, S, Y), _st(st, R, X)
        if not sy or not sx or not sy["nonnull"]:
            continue
        uniq = sy["ndv"] >= (1 - tol) * sy["nonnull"]
        if Tm:  # catalog statistics cannot see validity intervals; trust declared temporal keys
            uniq = True
        inc = sx["ndv"] <= (1 + tol) * sy["ndv"]
        try:
            inc = inc and _cmp(sx["min"], sy["min"]) >= 0 and _cmp(sx["max"], sy["max"]) <= 0
        except Exception:
            pass
        if uniq and inc:
            out.append(e)
    return out


def _cmp(a, b):
    if a is None or b is None:
        return 0
    try:
        a, b = float(a), float(b)
    except ValueError:
        pass
    return (a > b) - (a < b)


def autobi(P, st, thr=0.5):
    """Auto-BI-style reimplementation: local join score from name similarity, uniqueness and 1%-sample
    containment, then a maximum-weight arborescence rooted at every fact (per-fact MCA); edges of the
    arborescences are accepted."""
    import networkx as nx
    score = {}
    for e in P.edges:
        R, X, S, Y, Tm = e
        sy = _st(st, S, Y)
        c = st.get("C" + json.dumps(list(e)))
        if not sy or not c:
            continue
        uniq = min(1.0, sy["ndv"] / max(1, sy["nonnull"]))
        cont = c["contained"] / max(1, c["sample"]) if c["sample"] else 0.5
        ns = sum(_ns(x, y, R, S) for x, y in zip(X, Y)) / len(X)
        score[e] = (ns + (1.0 if uniq >= 0.98 else uniq) + cont) / 3
    acc = set()
    for F in P.rec["facts"]:
        G = nx.DiGraph()
        for e, s in score.items():
            if s >= thr and e[2] != F:
                if G.has_edge(e[0], e[2]) and G[e[0]][e[2]]["weight"] >= s:
                    continue
                G.add_edge(e[0], e[2], weight=s, e=e)
        if F not in G:
            continue
        reach = nx.descendants(G, F) | {F}
        H = G.subgraph(reach).copy()
        B = nx.maximum_branching(H, attr="weight")
        for u, v in B.edges():
            acc.add(H[u][v]["e"])
    return list(acc)


def _ns(x, y, R, S):
    import difflib
    a, b = tokens(x, R), tokens(y, S)
    jac = len(set(a) & set(b)) / max(1, len(set(a) | set(b)))
    return max(jac, difflib.SequenceMatcher(None, "_".join(a), "_".join(b)).ratio())


# ---------------------------------------------------------------------- driver
def run(names, out_main, out_zero, with_opt=True):
    fm = open(out_main, "w", newline="")
    fz = open(out_zero, "w", newline="")
    wm = wz = None
    for name in names:
        P, st = load(name)
        W = P.workload()
        if os.environ.get("WORKLOAD") == "official":
            ow = json.load(open(os.path.join(ROOT, "results", "official_workload.json")))
            base = "tpcds" if name.startswith("tpcds") else "tpch"
            W = [(F, tuple(D)) for _, F, D in ow[base]]
        t0 = time.time()
        rows = []
        for cm in ["atom", "scan"]:
            full = P.full_cost if cm == "atom" else P.full_scan
            for f in BUDGETS:
                B = f * full
                ex = dict(cost_model=cm, budget=f)
                S = Pm.certify_join(P, W, B, cost_model=cm)
                rows.append(P.evaluate(W, S.certified(), "CertifyJoin", S.spent, dict(ex, probes=S.probes, inferred=S.inferred)))
                S = Pm.certify_join(P, W, B, cost_model=cm, sample=SAMPLE)
                rows.append(P.evaluate(W, S.certified(), "CertifyJoin+S", S.spent, dict(ex, probes=S.probes, inferred=S.inferred)))
                for lab, kw in [("CJ-noPrior", dict(prior=False)), ("CJ-noInfer", dict(inference=False)),
                                ("CJ-noFailFast", dict(failfast=False))]:
                    S = Pm.certify_join(P, W, B, cost_model=cm, **kw)
                    rows.append(P.evaluate(W, S.certified(), lab, S.spent, dict(ex, probes=S.probes)))
                S = Pm.run_order(P, Pm.full_order(P), B, cost_model=cm)
                rows.append(P.evaluate(W, S.certified(), "Full(catalog order)", S.spent, dict(ex, probes=S.probes)))
                for sd in range(5):
                    S = Pm.run_order(P, Pm.random_order(P, sd), B, cost_model=cm)
                    rows.append(P.evaluate(W, S.certified(), "Random", S.spent, dict(ex, probes=S.probes, seed=sd)))
                S = Pm.run_order(P, Pm.keycov_order(P, W), B, cost_model=cm)
                rows.append(P.evaluate(W, S.certified(), "KeyCoverage", S.spent, dict(ex, probes=S.probes)))
                if cm == "scan":
                    S = Pm.profile_tables(P, B)
                    rows.append(P.evaluate(W, S.certified(), "Profile(IND-discovery)", S.spent, dict(ex)))
                if with_opt and cm == "atom" and f <= 0.2:
                    try:
                        S = Pm.opt_ilp(P, W, B)
                        if S is not None:
                            rows.append(P.evaluate(W, S.certified(), "OPT(clairvoyant ILP)", S.spent, dict(ex)))
                    except Exception as exn:
                        print(name, "opt failed", repr(exn)[:120], flush=True)
            # unbounded run: cost to decide the whole workload
            S = Pm.certify_join(P, W, 1e30, cost_model=cm)
            rows.append(P.evaluate(W, S.certified(), "CertifyJoin", S.spent, dict(cost_model=cm, budget="inf", probes=S.probes,
                                                                                inferred=S.inferred)))
            S = Pm.certify_join(P, W, 1e30, cost_model=cm, sample=SAMPLE)
            rows.append(P.evaluate(W, S.certified(), "CertifyJoin+S", S.spent, dict(cost_model=cm, budget="inf", probes=S.probes)))
        for r in rows:
            if wm is None:
                keys = sorted({k for r2 in rows for k in r2})
                wm = csv.DictWriter(fm, fieldnames=keys + ["seed", "inferred"], extrasaction="ignore")
                wm.writeheader()
            wm.writerow(r)
        fm.flush()
        z = [P.evaluate(W, ddl_declared(P), "DDL-declared", 0)]
        for m in P.rec["prior"]:
            if P.rec["prior"][m]:
                z.append(P.evaluate(W, llm_judge(P, m), "LLM-judge:" + m, 0))
        z.append(P.evaluate(W, llm_vote(P), "LLM-judge:vote", 0))
        if st:
            z.append(P.evaluate(W, card_est(P, st), "CardEst(HLL stats)", 0))
            z.append(P.evaluate(W, autobi(P, st), "Auto-BI-style(reimpl.)", 0))
        for r in z:
            if wz is None:
                wz = csv.DictWriter(fz, fieldnames=list(r.keys()))
                wz.writeheader()
            wz.writerow(r)
        fz.flush()
        print(name, "queries", len(W), "answerable", rows[0]["n_answerable"], "secs", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(f[:-5] for f in os.listdir(os.path.join(ROOT, "results", "pool")) if not f.startswith("ent"))
    tag = os.environ.get("TAG", "main")
    run(names, os.path.join(ROOT, "results", f"{tag}.csv"), os.path.join(ROOT, "results", f"{tag}_zero.csv"),
        with_opt=os.environ.get("OPT", "1") == "1")
