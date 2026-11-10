"""Ablation on deep chains of shared keys (review fix E2). Facts reference the leaf of subtype hierarchies
h_1 <- h_2 <- ... <- h_D whose levels share the key column (h_k.id is a subset of h_{k-1}.id, as in is-a and
vertical-partition tables of ERP schemas), and every level is also a direct candidate target of the fact column
(F.x -> h_k.id). Key sets are materialized symbolically (integer id sets), so every atom truth is exact and the
inference rules are sound on them; costs use the row counts of the generated tables. For depths 1..5 and five
seeds, runs Certify-Join and its ablations (no inference, no fail-first order, neither) with an unbounded budget and
with 1-20% budgets, and writes results/synth_deep.csv."""
import csv, json, math, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policies as Pm

ROOT = Pm.ROOT
PRIOR_MODEL = "synth"
BUDGETS = [0.01, 0.02, 0.05, 0.1, 0.2]


def logu(rnd, lo, hi):
    return int(round(math.exp(rnd.uniform(math.log(lo), math.log(hi)))))


def gen_pool(depth, seed, n_facts=20, n_chains=12, chains_per_fact=3, p_bad_link=0.15, p_bad_fact=0.15, p_dup=0.08):
    rnd = random.Random(7919 * seed + depth)
    rows, cols, ids, dup = {}, {}, {}, {}
    chains = []
    for c in range(n_chains):
        lv = [f"h{c}_{k}" for k in range(1, depth + 1)]
        base = 10 ** 6 * (c + 1)
        root_n = rnd.randint(400, 1200)
        ids[lv[0]] = set(range(base, base + root_n))
        rows[lv[0]] = logu(rnd, 1e4, 1e6)
        for k in range(1, depth):
            par = sorted(ids[lv[k - 1]])
            ch = set(rnd.sample(par, max(20, int(len(par) * rnd.uniform(0.5, 0.9)))))
            if rnd.random() < p_bad_link:  # subtype rows whose supertype row was deleted
                ch |= {base + 900000 + 1000 * k + j for j in range(rnd.randint(1, 5))}
            ids[lv[k]] = ch
            rows[lv[k]] = max(100, int(rows[lv[k - 1]] * len(ch) / len(par)))
        for t in lv:
            cols[t] = {"id": "BIGINT", "code": "BIGINT"}
            dup[t] = rnd.random() < p_dup
        chains.append(lv)
    facts = [f"f{i}" for i in range(n_facts)]
    fvals = {}
    for F in facts:
        rows[F] = logu(rnd, 1e6, 1e8)
        cols[F] = {"id": "BIGINT"}
        for c in rnd.sample(range(n_chains), chains_per_fact):
            leaf = sorted(ids[chains[c][-1]])
            v = set(rnd.choices(leaf, k=min(len(leaf), 300)))
            if rnd.random() < p_bad_fact:
                v |= {10 ** 6 * (c + 1) + 990000 + j for j in range(rnd.randint(1, 5))}
            col = f"h{c}_id"
            cols[F][col] = "BIGINT"
            fvals[(F, col)] = (c, v)

    edges, gold, seen = [], set(), set()

    def add(R, X, S, src, vals):
        e = (R, (X,), S, ("id",), None)
        if e in seen or R == S:
            return
        seen.add(e)
        edges.append((e, src, vals))
        if src == "gold":
            gold.add(e)

    for lv in chains:
        for k in range(1, depth):
            add(lv[k], "id", lv[k - 1], "gold", ids[lv[k]])
    for (F, col), (c, v) in sorted(fvals.items()):
        for t in chains[c]:
            add(F, col, t, "gold", v)  # leaf reference and the direct references to every supertype level
        for c2 in rnd.sample([x for x in range(n_chains) if x != c], 2):  # name-collision decoys
            add(F, col, rnd.choice(chains[c2]), "decoy", v)
    atoms, prior, out_edges = {}, {}, []
    for e, src, vals in edges:
        R, X, S, Y, _ = e
        ua, ia = Pm.atoms_of(e)
        if ua not in atoms:
            d = max(1, int(rows[S] * rnd.uniform(1e-4, 0.01))) if dup[S] else 0
            atoms[ua] = dict(dup_keys=d, dup_rows=2 * d, truth=not dup[S], cost=rows[S], rows_S=rows[S])
        orph_vals = len(vals - ids[S])
        it = orph_vals == 0
        orph = 0 if it else max(1, int(rows[R] * orph_vals / max(1, len(vals))))
        atoms[ia] = dict(nonnull=rows[R], orphans=orph, truth=it, cost=rows[R] + rows[S], rows_R=rows[R], rows_S=rows[S])
        name = f"{R}.{X[0]}->{S}.{Y[0]}"
        prior[name] = round(rnd.betavariate(6, 2) if src == "gold" else rnd.betavariate(2, 6), 4)
        out_edges.append({"e": [R, list(X), S, list(Y), None], "name": name, "src": src})
    tables = sorted(rows)
    return {"name": f"deep{depth}_s{seed}", "tables": {t: {"rows": rows[t], "cols": cols[t]} for t in tables},
            "pk": {t: ["id"] for t in tables}, "pk_enforced": [],
            "gold": [[R, list(X), S, list(Y), None] for (R, X, S, Y, _) in sorted(gold)], "declared": [],
            "facts": facts, "edges": out_edges, "atoms": atoms, "prior": {PRIOR_MODEL: prior}}


VARIANTS = [("CertifyJoin", {}), ("CJ-noInfer", dict(inference=False)), ("CJ-noFailFast", dict(failfast=False)),
            ("CJ-noInfer-noFailFast", dict(inference=False, failfast=False))]

if __name__ == "__main__":
    depths = [int(x) for x in os.environ.get("DEPTHS", "1,2,3,4,5").split(",")]
    seeds = range(int(os.environ.get("SEEDS", "5")))
    out = os.path.join(ROOT, "results", "synth_deep.csv")
    w = None
    with open(out, "w", newline="") as fh:
        for depth in depths:
            for sd in seeds:
                rec = gen_pool(depth, sd)
                P = Pm.Pool(rec, prior_model=PRIOR_MODEL)
                W = P.workload(seed=sd, per_fact=10)
                for lab, kw in VARIANTS:
                    for f in BUDGETS + ["inf"]:
                        B = 1e30 if f == "inf" else f * P.full_cost
                        S = Pm.certify_join(P, W, B, cost_model="atom", **kw)
                        r = P.evaluate(W, S.certified(), lab, S.spent, dict(depth=depth, seed=sd, budget=f, probes=S.probes,
                                                                             inferred=getattr(S, "inferred", 0)))
                        if w is None:
                            w = csv.DictWriter(fh, fieldnames=list(r.keys()))
                            w.writeheader()
                        w.writerow(r)
                fh.flush()
                print(depth, sd, len(P.edges), len(W), flush=True)
