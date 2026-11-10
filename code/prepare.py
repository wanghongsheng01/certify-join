"""Build every pilot instance once: candidate pool, exact atom truths, LLM priors (3 vendors). Writes results/pool/*.json."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instances as I, atoms as A, llm_prior as P

OUT = os.path.join(I.ROOT, "results", "pool")
MODELS = ["ds-flash", "glm", "kimi"]


def build(inst, cand=None, prior_inst=None):
    t0 = time.time()
    cand = cand or I.candidates(inst)
    edges = list(cand) + [g for g in inst["gold"] if g not in cand]
    for g in inst["gold"]:
        cand.setdefault(g, "ddl")
    at = A.compute(inst, edges)
    pri, usage = {}, {}
    for m in MODELS:
        try:
            pri[m], usage[m] = P.priors(prior_inst or inst, edges, model=m)
        except Exception as ex:
            print(inst["name"], m, "LLM failed", repr(ex)[:200], flush=True)
            pri[m], usage[m] = {}, []
    rec = dict(name=inst["name"], tables={t: {"rows": v["rows"], "cols": {c: str(ty) for c, ty in v["cols"].items()}}
                                           for t, v in inst["tables"].items()},
               pk=inst["pk"], pk_enforced=sorted(inst["pk_enforced"]), gold=[list(g) for g in inst["gold"]],
               declared=[list(g) for g in inst["declared"]], facts=inst["facts"],
               edges=[dict(e=list(e), name=I.ename(e), src=cand[e]) for e in edges], atoms=at, prior=pri, usage=usage,
               inject=inst.get("inject", []), secs=round(time.time() - t0, 1))
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(os.path.join(OUT, inst["name"] + ".json"), "w"))
    print(inst["name"], "edges", len(edges), "secs", rec["secs"], flush=True)
    return rec


if __name__ == "__main__":
    which = sys.argv[1:] or ["tpch", "tpcds", "bird", "dirty"]
    if "tpch" in which:
        build(I.load_tpch())
    if "tpcds" in which:
        build(I.load_tpcds())
    if "job" in which:
        build(I.load_job())
    if "bird" in which:
        I.bird_to_duckdb()
        for db in I.bird_ids():
            build(I.load_bird(db))
    if "dirty" in which:
        # TPC-H has only 10 FK edges: one duplicated dimension and two orphaned edges (~30% of edges broken)
        for ld, seeds, kw in [(I.load_tpch, [0, 1, 2], dict(n_dim=1, n_fact=2)), (I.load_tpcds, [0, 1], {})]:
            if ld is I.load_tpcds and "tpch_only" in which:
                continue
            base = ld()
            cand = I.candidates(base)
            for s in seeds:
                d = I.make_dirty(base, os.path.join(I.DATA, f"{base['name']}_dirty{s}.duckdb"), seed=s, **kw)
                d["name"] = f"{base['name']}_dirty{s}"
                build(d, cand=dict(cand), prior_inst=base)
