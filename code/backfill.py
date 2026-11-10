"""Fill missing LLM priors (e.g. after rate limits) serially; prompts are rebuilt from the base schema so cached
calls are reused. Usage: python backfill.py [model ...]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_prior as LP, policies as Pm
D = os.path.join(Pm.ROOT, "results", "pool")
force = "--force" in sys.argv
models = [a for a in sys.argv[1:] if a != "--force"] or ["glm", "kimi", "ds-flash"]
for f in sorted(os.listdir(D)):
    rec = json.load(open(os.path.join(D, f)))
    base = rec["name"].split("_dirty")[0]
    brec = json.load(open(os.path.join(D, base + ".json"))) if base != rec["name"] else rec
    inst = {"tables": brec["tables"]}
    edges = [Pm.T(d["e"]) for d in rec["edges"]]
    changed = False
    for m in models:
        pri = rec["prior"].get(m) or {}
        miss = [e for e in edges if pri.get(rec["edges"][edges.index(e)]["name"]) is None]
        if len(miss) <= 0.02 * len(edges) and not force:
            continue
        try:
            new, usage = LP.priors(inst, edges, model=m)
        except Exception as ex:
            print(f, m, "failed", repr(ex)[:150]); continue
        rec["prior"][m] = new
        rec["usage"][m] = usage
        changed = True
        print(f, m, "missing before", len(miss), "after", sum(v is None for v in new.values()), flush=True)
    if changed:
        json.dump(rec, open(os.path.join(D, f), "w"))
