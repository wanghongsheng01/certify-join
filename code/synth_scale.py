"""RQ6 planning scalability: synthetic snowflake/constellation pools with 1e2..1e4 tables.

Generates pool records in the results/pool/<inst>.json format (no data; atom truths are drawn), runs
certify_join (cost_model="atom") with an unbounded budget and with 5% of the full probing cost, and writes
results/synth_scale.csv. Probes are simulated from the stored truths, so the wall time of certify_join is the
planning overhead of Certify-Join.

usage: python code/synth_scale.py [--sizes 100,300,1000,3000,10000] [--seeds 0,1] [--out results/synth_scale.csv]
"""
import argparse, csv, heapq, json, math, os, random, sys, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policies as Pm

ROOT = Pm.ROOT
PRIOR_MODEL = "synth"


def logu(rnd, lo, hi):
    return int(round(math.exp(rnd.uniform(math.log(lo), math.log(hi)))))


def gen_pool(n_tables, seed=0, fact_frac=0.10, decoys_per_gold=3, p_gold_safe=0.85, p_decoy_safe=0.10):
    """Constellation of facts over conformed, snowflaked dimensions (depth <= 3) plus name-collision decoys."""
    rnd = random.Random(1000003 * seed + n_tables)
    n_f = max(1, int(round(fact_frac * n_tables)))
    n_d = n_tables - n_f
    # dimension levels: L1 referenced by facts, L2/L3 snowflaked sub-dimensions
    n1 = max(1, int(round(0.45 * n_d)))
    n2 = max(1, int(round(0.35 * n_d)))
    n3 = max(0, n_d - n1 - n2)
    facts = [f"f{i}" for i in range(n_f)]
    L = [[f"d1_{i}" for i in range(n1)], [f"d2_{i}" for i in range(n2)], [f"d3_{i}" for i in range(n3)]]
    rows = {}
    for F in facts:
        rows[F] = logu(rnd, 1e6, 1e8)
    for lev in L:
        for D in lev:
            rows[D] = logu(rnd, 1e2, 1e6)
    cols = {t: {"id": "BIGINT"} for t in rows}
    gold = []  # (R, col, S)

    # conformed dimensions: popularity ~ Zipf so a few dimensions are shared by many facts (date, customer, ...)
    w1 = [1.0 / (i + 1) ** 0.8 for i in range(n1)]
    for F in facts:
        k = min(n1, rnd.randint(3, 8))
        chosen = set()
        while len(chosen) < k:
            chosen.add(rnd.choices(range(n1), weights=w1)[0])
        for j in sorted(chosen):
            gold.append((F, f"{L[0][j]}_id", L[0][j]))
    # snowflake: each L1/L2 dimension references 0..2 dimensions of the next level; every sub-dimension has a parent
    for lv in (0, 1):
        par, ch = L[lv], L[lv + 1]
        if not ch:
            continue
        wch = [1.0 / (i + 1) ** 0.8 for i in range(len(ch))]
        has_parent = set()
        links = defaultdict(set)
        for D in par:
            for _ in range(rnd.choice([0, 1, 1, 2])):
                c = rnd.choices(range(len(ch)), weights=wch)[0]
                links[D].add(c)
                has_parent.add(c)
        for c in range(len(ch)):
            if c not in has_parent:
                links[rnd.choice(par)].add(c)
        for D in par:
            for c in sorted(links[D]):
                gold.append((D, f"{ch[c]}_id", ch[c]))
    for R, c, S in gold:
        cols[R][c] = "BIGINT"
    # attribute columns that look like keys (name-collision sources): code / type_id / status_id ...
    attr_names = ["code", "type_id", "status_id", "ref_no", "group_id", "parent_id", "seq"]
    for t in rows:
        for a in rnd.sample(attr_names, 2):
            cols[t][a] = "BIGINT"

    tables = list(rows)
    edges, seen = [], set()

    def add(R, X, S, src):
        e = (R, (X,), S, ("id",), None)
        if e in seen or R == S:
            return False
        seen.add(e)
        edges.append((e, src))
        return True

    for R, c, S in gold:
        add(R, c, S, "gold")
    gold_set = {(R, (c,), S, ("id",), None) for R, c, S in gold}
    # decoys: ~3 per gold edge; 2/3 reuse the gold FK column against a wrong (same-level) table, 1/3 an attribute
    # column of R against the gold target ("R.col -> S.id" name collisions)
    lvl_of = {}
    for i, lev in enumerate(L):
        for D in lev:
            lvl_of[D] = i
    for R, c, S in gold:
        pool_same = L[lvl_of[S]]
        made, tries = 0, 0
        while made < decoys_per_gold and tries < 20:
            tries += 1
            if made < 2 or rnd.random() < 0.3:
                S2 = rnd.choice(pool_same) if rnd.random() < 0.8 else rnd.choice(tables)
                made += add(R, c, S2, "decoy")
            else:
                a = rnd.choice([x for x in cols[R] if x in attr_names])
                made += add(R, a, S, "decoy")

    # truth: U per target key (shared by every edge into S), I per edge.
    # gold safe w.p. ~0.85: U(S.id) false w.p. pu, I false w.p. pi with (1-pu)(1-pi) = 0.85, failures ~ half/half
    pu = 1 - math.sqrt(p_gold_safe)
    pi_gold = 1 - p_gold_safe / (1 - pu)
    pi_decoy_true = min(1.0, p_decoy_safe / (1 - pu))
    atoms, prior = {}, {}
    utruth = {t: rnd.random() >= pu for t in tables}
    out_edges = []
    for e, src in edges:
        R, X, S, Y, _ = e
        ua, ia = Pm.atoms_of(e)
        if ua not in atoms:
            dup = 0 if utruth[S] else max(1, int(rows[S] * rnd.uniform(1e-4, 0.05)))
            atoms[ua] = dict(dup_keys=dup, dup_rows=2 * dup, truth=utruth[S], cost=rows[S], rows_S=rows[S])
        it = (rnd.random() >= pi_gold) if src == "gold" else (rnd.random() < pi_decoy_true)
        if it:
            orph = 0
        else:
            rate = math.exp(rnd.uniform(math.log(1e-4), math.log(0.05))) if src == "gold" else rnd.uniform(0.3, 1.0)
            orph = max(1, int(rows[R] * rate))
        atoms[ia] = dict(nonnull=rows[R], orphans=orph, truth=it, cost=rows[R] + rows[S], rows_R=rows[R], rows_S=rows[S])
        name = f"{R}.{X[0]}->{S}.{Y[0]}"
        p = rnd.betavariate(6, 2) if src == "gold" else rnd.betavariate(2, 6)
        prior[name] = round(p, 4)
        out_edges.append({"e": [R, list(X), S, list(Y), None], "name": name, "src": src})
    return {
        "name": f"synth{n_tables}_s{seed}",
        "tables": {t: {"rows": rows[t], "cols": cols[t]} for t in tables},
        "pk": {t: ["id"] for t in tables},
        "pk_enforced": [],
        "gold": [[R, list(X), S, list(Y), None] for (R, X, S, Y, _) in sorted(gold_set)],
        "declared": [],
        "facts": facts,
        "edges": out_edges,
        "atoms": atoms,
        "prior": {PRIOR_MODEL: prior},
    }


def coverage(P, W, cert):
    """Same quantity as Pool.evaluate(...)['coverage'] (n_ok / n_answerable), with the certified-edge adjacency
    and the per-fact Dijkstra built once instead of once per query."""
    cert = set(cert)
    adj = defaultdict(list)
    for e in cert:
        adj[e[0]].append(e)
    gadj = defaultdict(list)
    for e in P.gold:
        if P.safe[e]:
            gadj[e[0]].append(e)
    cache, rcache = {}, {}

    def reach(F):
        if F not in rcache:
            seen, st = {F}, [F]
            while st:
                u = st.pop()
                for e in gadj[u]:
                    if e[2] not in seen:
                        seen.add(e[2])
                        st.append(e[2])
            rcache[F] = seen
        return rcache[F]

    def sp(F):
        if F not in cache:
            dist, par, pq = {F: 0.0}, {}, [(0.0, F)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for e in adj[u]:
                    nd = d - math.log(P.p[e])
                    if nd < dist.get(e[2], 1e18):
                        dist[e[2]], par[e[2]] = nd, e
                        heapq.heappush(pq, (nd, e[2]))
            cache[F] = (dist, par)
        return cache[F]

    n_ans = n_ok = n_cert = 0
    for F, D in W:
        n_ans += all(t in reach(F) for t in D)
        dist, par = sp(F)
        if not all(t in dist for t in D):
            continue
        n_cert += 1
        tree = set()
        for t in D:
            while t != F:
                e = par[t]
                tree.add(e)
                t = e[0]
        if all(P.safe[e] for e in tree) and all(e in P.gold for e in tree):
            n_ok += 1
    return n_ok / max(1, n_ans), n_ans, n_cert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100,300,1000,3000,10000")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--per_fact", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "synth_scale.csv"))
    ap.add_argument("--save_pools", default="")  # optional directory for the generated pool records
    a = ap.parse_args()
    fields = ["n_tables", "seed", "n_edges", "n_gold", "n_atoms", "n_queries", "budget", "budget_rows", "wall_s",
              "iterations", "probes", "inferred", "spent_frac", "coverage", "n_answerable", "n_certified_q", "gen_s"]
    exists = os.path.exists(a.out)
    fo = open(a.out, "a" if exists else "w", newline="")
    w = csv.DictWriter(fo, fieldnames=fields)
    if not exists:
        w.writeheader()
    for n in [int(x) for x in a.sizes.split(",")]:
        for sd in [int(x) for x in a.seeds.split(",")]:
            t0 = time.time()
            rec = gen_pool(n, sd)
            if a.save_pools:
                os.makedirs(a.save_pools, exist_ok=True)
                json.dump(rec, open(os.path.join(a.save_pools, rec["name"] + ".json"), "w"))
            P = Pm.Pool(rec, prior_model=PRIOR_MODEL)
            W = P.workload(seed=sd, per_fact=a.per_fact)
            gen_s = time.time() - t0
            for lab, B in [("inf", 1e30), ("0.05", 0.05 * P.full_cost)]:
                t1 = time.time()
                S = Pm.certify_join(P, W, B, cost_model="atom")
                wall = time.time() - t1
                cov, n_ans, n_cq = coverage(P, W, S.certified())
                r = dict(n_tables=n, seed=sd, n_edges=len(P.edges), n_gold=len(P.gold),
                         n_atoms=len({x for e in P.edges for x in P.ea[e]}), n_queries=len(W), budget=lab,
                         budget_rows=B if lab != "inf" else "", wall_s=round(wall, 3),
                         iterations=getattr(S, "iterations", ""), probes=S.probes, inferred=S.inferred,
                         spent_frac=round(S.spent / P.full_cost, 5), coverage=round(cov, 4), n_answerable=n_ans,
                         n_certified_q=n_cq, gen_s=round(gen_s, 2))
                w.writerow(r)
                fo.flush()
                print(json.dumps(r), flush=True)
    fo.close()


if __name__ == "__main__":
    main()
