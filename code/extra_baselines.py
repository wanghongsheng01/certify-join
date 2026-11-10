"""Stronger exhaustive baselines and wall-clock cost (review fixes F4, F2).

  Full-KeyFirst   probe every candidate edge in catalog order, key atom first, and skip its inclusion atom when the
                  key is false (the obvious improvement of per-edge validation)
  WL-Full         the same, restricted to candidate edges on some path of at most 3 hops from a query's fact table to
                  one of its targets (verification limited to the workload, without bundle selection)

For every method the cost to decide the whole workload is reported in rows (atom model) and in seconds, where seconds
are the measured wall-clock times of the exact SQL probes recorded when the pools were built (results/pool/*.json,
atom field "secs"); for Certify-Join+S a sampled inclusion probe is charged secs * (m + |S|) / (|R| + |S|).
Writes results/extra_baselines.csv (budget curves, atom model) and results/decide_seconds.csv."""
import csv, json, math, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policies as Pm
from run_pilot import load, ROOT, BUDGETS, SAMPLE


def keyfirst(P, B, edges, cost_model="atom"):
    S = Pm.State(P, use_inference=False, cost_model=cost_model)
    probed = []
    for e in edges:
        u, i = P.ea[e]
        for a in (u, i):
            if a == i and S.status(u) is False:
                break
            if S.status(a) is not None:
                continue
            c = S.cost_of(a)
            if S.spent + c > B:
                break
            S.probe(a); probed.append(a)
            if S.status(a) is False:
                break
    return S, probed


def catalog_edges(P):
    return sorted(P.edges, key=lambda e: (P.src[e] != "ddl", P.ename[e]))


def workload_edges(P, W, hops=3):
    adj = defaultdict(list)
    for e in P.edges:
        adj[e[0]].append(e)
    keep = set()
    for F, D in W:
        # all candidate edges on simple paths of <= hops from F that end in a target of the query
        def dfs(u, path, seen):
            if u in D and path:
                keep.update(path)
            if len(path) == hops:
                return
            for e in adj[u]:
                if e[2] not in seen:
                    dfs(e[2], path + [e], seen | {e[2]})
        dfs(F, [], {F})
    return [e for e in catalog_edges(P) if e in keep]


def secs_of(P, probed, sampled=(), m=0):
    t = 0.0
    for a in probed:
        s = P.A[a].get("secs") or 0.0
        if a in sampled:
            d = json.loads(a)
            R, S_ = d[1], d[3]
            s = s * (m + P.rows[S_]) / max(1, P.rows[R] + P.rows[S_])
        t += s
    return t


class Rec(Pm.State):
    """State that remembers which atoms were probed (for the wall-clock accounting of Certify-Join)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.probed = []

    def probe(self, a, outcome=None, cost=None):
        self.probed.append(a)
        return super().probe(a, outcome=outcome, cost=cost)


if __name__ == "__main__":
    names = sorted(f[:-5] for f in os.listdir(os.path.join(ROOT, "results", "pool")) if not f.startswith("ent"))
    m_samp = int(math.ceil(math.log(1 / SAMPLE[1]) / SAMPLE[0]))
    fc = open(os.path.join(ROOT, "results", "extra_baselines.csv"), "w", newline="")
    fs = open(os.path.join(ROOT, "results", "decide_seconds.csv"), "w", newline="")
    wc = ws = None
    orig_state = Pm.State
    for name in names:
        P, st = load(name)
        W = P.workload()
        cat, wl = catalog_edges(P), workload_edges(P, W)
        rows = []
        for f in BUDGETS + ["inf"]:
            B = 1e30 if f == "inf" else f * P.full_cost
            for lab, E in [("Full-KeyFirst", cat), ("WL-Full", wl)]:
                S, pr = keyfirst(P, B, E)
                r = P.evaluate(W, S.certified(), lab, S.spent, dict(cost_model="atom", budget=f, probes=len(pr),
                                                                     secs=round(secs_of(P, pr), 3)))
                rows.append(r)
        for r in rows:
            if wc is None:
                wc = csv.DictWriter(fc, fieldnames=list(r.keys())); wc.writeheader()
            wc.writerow(r)
        fc.flush()
        # seconds to decide the whole workload
        all_atoms = sorted({a for e in P.edges for a in P.ea[e]})
        out = dict(inst=name, full_secs=round(secs_of(P, all_atoms), 3))
        S, pr = keyfirst(P, 1e30, cat); out["keyfirst_secs"] = round(secs_of(P, pr), 3); out["keyfirst_rows"] = S.spent / P.full_cost
        S, pr = keyfirst(P, 1e30, wl); out["wlfull_secs"] = round(secs_of(P, pr), 3); out["wlfull_rows"] = S.spent / P.full_cost
        Pm.State = Rec
        try:
            S = Pm.certify_join(P, W, 1e30)
            out["cj_secs"] = round(secs_of(P, S.probed), 3); out["cj_rows"] = S.spent / P.full_cost
            S = Pm.certify_join(P, W, 1e30, sample=SAMPLE)
            samp = {a for a in S.probed if a.startswith('["I"') and P.rows[json.loads(a)[1]] > 10 * m_samp}
            out["cjs_secs"] = round(secs_of(P, S.probed, sampled=samp, m=m_samp), 3)
            out["cjs_rows"] = S.spent / P.full_cost
        finally:
            Pm.State = orig_state
        if ws is None:
            ws = csv.DictWriter(fs, fieldnames=list(out.keys())); ws.writeheader()
        ws.writerow(out); fs.flush()
        print(out, flush=True)
